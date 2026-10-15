import argparse
import csv
import hashlib
import json
import pickle
import random
import re
import shutil
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image


SPLITS = ("train", "valid", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        "Prepare balanced historical-book sources as page-disjoint vertical columns"
    )
    parser.add_argument(
        "--catalog",
        default="experiments/configs/curated_historical_sources_v1.json",
        type=Path,
    )
    parser.add_argument(
        "--output-root", default="data/curated_historical_v1_dtlr", type=Path
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Parse and summarize candidates without writing crops or pickle files.",
    )
    return parser.parse_args()


def safe_id(value):
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")


def make_sample_id(source_id, page_id, column_index):
    return f"{source_id}_{safe_id(page_id)}_c{column_index:03d}"


def stable_seed(seed, value):
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def split_pages(page_ids, ratios, seed, source_id):
    pages = sorted(page_ids)
    random.Random(stable_seed(seed, source_id)).shuffle(pages)
    total = len(pages)
    n_train = int(total * ratios["train"])
    n_valid = int(total * ratios["valid"])
    if total >= 3:
        n_train = max(1, min(n_train, total - 2))
        n_valid = max(1, min(n_valid, total - n_train - 1))
    return {
        "train": pages[:n_train],
        "valid": pages[n_train : n_train + n_valid],
        "test": pages[n_train + n_valid :],
    }


def split_weighted_pages(page_weights, ratios, seed, source_id):
    """Assign whole pages while balancing the resulting number of columns."""
    pages = sorted(page_weights)
    randomizer = random.Random(stable_seed(seed, source_id))
    tie_breakers = {page: randomizer.random() for page in pages}
    pages.sort(key=lambda page: (-page_weights[page], tie_breakers[page], page))
    target_total = sum(page_weights.values())
    targets = {split: max(1.0, target_total * ratios[split]) for split in SPLITS}
    totals = {split: 0 for split in SPLITS}
    assignments = {split: [] for split in SPLITS}
    for page in pages:
        split = min(
            SPLITS,
            key=lambda name: (
                totals[name] / targets[name],
                len(assignments[name]),
                SPLITS.index(name),
            ),
        )
        assignments[split].append(page)
        totals[split] += page_weights[page]
    return assignments


def parse_voc_page(xml_path):
    root = ET.parse(xml_path).getroot()
    image_path = xml_path.with_suffix(".png")
    filename = (root.findtext("filename") or "").strip()
    if not image_path.exists() and filename:
        image_path = xml_path.parent / filename
    if not image_path.exists():
        raise FileNotFoundError(f"No image for {xml_path}")

    characters = []
    for obj in root.findall("object"):
        text = (obj.findtext("name") or "").strip()
        if len(text) != 1:
            raise ValueError(f"Expected one character in {xml_path}, got {text!r}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {xml_path}")
        x1 = int(float(box.findtext("xmin")))
        y1 = int(float(box.findtext("ymin")))
        x2 = int(float(box.findtext("xmax")))
        y2 = int(float(box.findtext("ymax")))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid box {(x1, y1, x2, y2)} in {xml_path}")
        characters.append(
            {
                "text": text,
                "box": (x1, y1, x2, y2),
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
            }
        )
    return image_path, characters


def make_column(group):
    ordered = sorted(group, key=lambda item: (item["cy"], item["cx"]))
    x1 = min(item["box"][0] for item in ordered)
    y1 = min(item["box"][1] for item in ordered)
    x2 = max(item["box"][2] for item in ordered)
    y2 = max(item["box"][3] for item in ordered)
    return {
        "text": "".join(item["text"] for item in ordered),
        "box": (x1, y1, x2, y2),
        "characters": ordered,
        "center_x": statistics.median(item["cx"] for item in ordered),
        "median_char_width": statistics.median(item["width"] for item in ordered),
        "median_char_height": statistics.median(item["height"] for item in ordered),
    }


def group_vertical_columns(characters, center_gap_factor=0.6):
    if not characters:
        return []
    median_width = statistics.median(char["width"] for char in characters)
    max_gap = max(8.0, center_gap_factor * median_width)
    groups = []
    for char in sorted(characters, key=lambda item: item["cx"]):
        if not groups or char["cx"] - groups[-1][-1]["cx"] > max_gap:
            groups.append([char])
        else:
            groups[-1].append(char)
    columns = [make_column(group) for group in groups]
    return sorted(columns, key=lambda item: item["center_x"], reverse=True)


def drop_isolated_leading_char(column, minimum_gap_ratio):
    characters = column["characters"]
    if len(characters) < 3:
        return column, False
    center_gaps = [
        characters[index + 1]["cy"] - characters[index]["cy"]
        for index in range(len(characters) - 1)
    ]
    regular_gap = statistics.median(center_gaps[1:])
    if regular_gap <= 0 or center_gaps[0] / regular_gap < minimum_gap_ratio:
        return column, False
    return make_column(characters[1:]), True


def allocate_quota(total, ratios):
    train = round(total * ratios["train"])
    valid = round(total * ratios["valid"])
    return {"train": train, "valid": valid, "test": total - train - valid}


def sample_columns(candidates, quota, seed, source_id, split):
    if len(candidates) < quota:
        raise ValueError(
            f"{source_id}/{split} has {len(candidates)} candidates, below quota {quota}"
        )
    indices = list(range(len(candidates)))
    random.Random(stable_seed(seed, f"{source_id}:{split}")).shuffle(indices)
    selected = [candidates[index] for index in sorted(indices[:quota])]
    return selected


def crop_box(column, image_size, margin_ratio, min_margin):
    width, height = image_size
    x1, y1, x2, y2 = column["box"]
    margin_x = max(min_margin, round(column["median_char_width"] * margin_ratio))
    margin_y = max(min_margin, round(column["median_char_height"] * margin_ratio))
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(width, x2 + margin_x),
        min(height, y2 + margin_y),
    )


def length_bins(samples):
    lengths = [len(sample["text"]) for sample in samples]
    return {
        "1": sum(length == 1 for length in lengths),
        "2": sum(length == 2 for length in lengths),
        "3-5": sum(3 <= length <= 5 for length in lengths),
        "6-10": sum(6 <= length <= 10 for length in lengths),
        "11+": sum(length >= 11 for length in lengths),
    }


def column_is_eligible(column, catalog):
    if len(column["text"]) < catalog["min_column_chars"]:
        return False
    disallowed = set(catalog.get("disallowed_labels", []))
    return not any(character in disallowed for character in column["text"])


def load_catalog(path):
    catalog = json.loads(path.read_text(encoding="utf-8"))
    ratios = catalog["split_ratios"]
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split_ratios must contain train/valid/test and sum to 1")
    return catalog


def load_excluded_sample_ids(catalog, catalog_path):
    findings_path = catalog.get("exclude_sample_ids_file")
    if not findings_path:
        return set()
    findings_path = Path(findings_path)
    if not findings_path.is_absolute():
        findings_path = (catalog_path.parent.parent.parent / findings_path).resolve()
    with findings_path.open("r", encoding="utf-8") as handle:
        return {
            row["sample_id"]
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("status") == "issue"
        }


def build_candidates(catalog, catalog_path):
    raw_root = Path(catalog["raw_root"])
    if not raw_root.is_absolute():
        raw_root = (catalog_path.parent.parent.parent / raw_root).resolve()
    excluded_sample_ids = load_excluded_sample_ids(catalog, catalog_path)
    all_sources = {}
    for source in catalog["sources"]:
        source_id = source["id"]
        source_root = raw_root / source["directory"]
        xml_paths = sorted(source_root.rglob("*.xml"))
        page_map = {xml_path.stem: xml_path for xml_path in xml_paths}
        page_candidates = {}
        for page_id, xml_path in page_map.items():
            image_path, characters = parse_voc_page(xml_path)
            columns = group_vertical_columns(
                characters, catalog["center_gap_factor"]
            )
            page_candidates[page_id] = []
            for column_index, column in enumerate(columns):
                record_id = make_sample_id(source_id, page_id, column_index)
                if record_id in excluded_sample_ids:
                    continue
                dropped_leading_char = False
                if source.get("drop_isolated_leading_char", False):
                    column, dropped_leading_char = drop_isolated_leading_char(
                        column,
                        float(source.get("isolated_leading_gap_ratio", 2.5)),
                    )
                if not column_is_eligible(column, catalog):
                    continue
                page_candidates[page_id].append(
                    {
                        **column,
                        "id": record_id,
                        "source_id": source_id,
                        "page_id": page_id,
                        "column_index": column_index,
                        "image_path": image_path,
                        "xml_path": xml_path,
                        "dropped_isolated_leading_char": dropped_leading_char,
                    }
                )
        page_split = split_weighted_pages(
            {page_id: len(columns) for page_id, columns in page_candidates.items()},
            catalog["split_ratios"],
            catalog["split_seed"],
            source_id,
        )
        candidates = {split: [] for split in SPLITS}
        for split in SPLITS:
            for page_id in page_split[split]:
                candidates[split].extend(page_candidates[page_id])
        all_sources[source_id] = {
            "metadata": source,
            "page_splits": page_split,
            "candidates": candidates,
        }
    return all_sources


def main():
    args = parse_args()
    catalog_path = args.catalog.resolve()
    catalog = load_catalog(catalog_path)
    sources = build_candidates(catalog, catalog_path)
    quotas = {
        source["id"]: allocate_quota(source["quota"], catalog["split_ratios"])
        for source in catalog["sources"]
    }

    selected = {split: [] for split in SPLITS}
    for source_id, source_data in sources.items():
        for split in SPLITS:
            selected[split].extend(
                sample_columns(
                    source_data["candidates"][split],
                    quotas[source_id][split],
                    catalog["split_seed"],
                    source_id,
                    split,
                )
            )

    summary = {
        "dataset": catalog["dataset"],
        "catalog": str(catalog_path),
        "split_seed": catalog["split_seed"],
        "split_ratios": catalog["split_ratios"],
        "min_column_chars": catalog["min_column_chars"],
        "center_gap_factor": catalog["center_gap_factor"],
        "disallowed_labels": catalog.get("disallowed_labels", []),
        "manual_excluded_sample_ids": len(
            load_excluded_sample_ids(catalog, catalog_path)
        ),
        "sources": {},
        "splits": {split: len(selected[split]) for split in SPLITS},
        "length_bins": {split: length_bins(selected[split]) for split in SPLITS},
    }
    for source_id, source_data in sources.items():
        summary["sources"][source_id] = {
            "title": source_data["metadata"]["title"],
            "quota": sum(quotas[source_id].values()),
            "page_splits": {
                split: len(source_data["page_splits"][split]) for split in SPLITS
            },
            "candidate_columns": {
                split: len(source_data["candidates"][split]) for split in SPLITS
            },
            "selected_columns": quotas[source_id],
            "dropped_isolated_leading_chars": sum(
                int(record["dropped_isolated_leading_char"])
                for split in SPLITS
                for record in selected[split]
                if record["source_id"] == source_id
            ),
            "license_status": source_data["metadata"]["license_status"],
        }
    if args.audit_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    output_root = args.output_root.resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"{output_root} exists. Use --overwrite to replace it.")
    if output_root.exists():
        shutil.rmtree(output_root)
    lines_root = output_root / "lines"
    lines_root.mkdir(parents=True)

    labels = {"ground_truth": {split: [] for split in SPLITS}}
    charset = Counter()
    manifest_rows = []
    crop_sizes = []
    for split in SPLITS:
        for record in selected[split]:
            source_id = record["source_id"]
            sample_id = record["id"]
            with Image.open(record["image_path"]) as image:
                image = image.convert("RGB")
                bbox = crop_box(
                    record,
                    image.size,
                    catalog["margin_ratio"],
                    catalog["min_margin"],
                )
                crop = image.crop(bbox)
                crop.save(lines_root / f"{sample_id}.jpg", quality=95)
                crop_sizes.append(crop.size)
            text = record["text"]
            labels["ground_truth"][split].append(
                {"id": sample_id, "text": text, "direction": "vertical"}
            )
            charset.update(text)
            source_metadata = sources[source_id]["metadata"]
            manifest_rows.append(
                {
                    "id": sample_id,
                    "split": split,
                    "text": text,
                    "text_length": len(text),
                    "source_id": source_id,
                    "title": source_metadata["title"],
                    "page_id": record["page_id"],
                    "column_index_right_to_left": record["column_index"],
                    "dropped_isolated_leading_char": record[
                        "dropped_isolated_leading_char"
                    ],
                    "source_image": str(record["image_path"]),
                    "source_xml": str(record["xml_path"]),
                    "crop_box": list(bbox),
                    "text_creation_period": source_metadata[
                        "text_creation_period"
                    ],
                    "physical_edition_period": source_metadata[
                        "physical_edition_period"
                    ],
                    "reproduction_period": source_metadata[
                        "reproduction_period"
                    ],
                    "style": source_metadata["style"],
                    "carrier": source_metadata["carrier"],
                    "license_status": source_metadata["license_status"],
                }
            )

    with (output_root / "labels.pkl").open("wb") as handle:
        pickle.dump(labels, handle)
    with (output_root / "charset.pkl").open("wb") as handle:
        pickle.dump(sorted(charset), handle)
    with (output_root / "columns_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary.update(
        {
            "output_root": str(output_root),
            "total_columns": sum(summary["splits"].values()),
            "total_chars": sum(len(row["text"]) for row in manifest_rows),
            "unique_chars": len(charset),
            "crop_size": {
                "min_width": min(size[0] for size in crop_sizes),
                "max_width": max(size[0] for size in crop_sizes),
                "min_height": min(size[1] for size in crop_sizes),
                "max_height": max(size[1] for size in crop_sizes),
            },
            "loader_options": {
                "mth1000_root": output_root.name,
                "mth1000_labels_file": "labels.pkl",
                "mth1000_images_subdir": "lines",
                "mth1000_image_ext": "jpg",
            },
        }
    )
    (output_root / "meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
