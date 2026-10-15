#!/usr/bin/env python3
"""Convert selected CHI-KNOW-PO books into page-disjoint SAQT datasets."""

import argparse
import hashlib
import json
import pickle
import random
import re
import shutil
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


SPLITS = ("train", "valid", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/chiknowpo_transfer_v1.json"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/external/chiknowpo_hf"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/chiknowpo_transfer_v1_candidate"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(seed, value):
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def safe_id(value):
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")


def sample_id(domain_id, file_name):
    stem = safe_id(Path(file_name).stem)
    digest = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:10]
    return f"{domain_id}_{stem}_{digest}"


def split_weighted_pages(page_weights, ratios, seed, domain_id):
    pages = sorted(page_weights)
    randomizer = random.Random(stable_seed(seed, domain_id))
    tie_breakers = {page: randomizer.random() for page in pages}
    pages.sort(key=lambda page: (-page_weights[page], tie_breakers[page], page))
    targets = {
        split: max(1.0, sum(page_weights.values()) * ratios[split])
        for split in SPLITS
    }
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


def select_rows(rows, count, seed, domain_id, split):
    if len(rows) < count:
        raise ValueError(
            f"{domain_id}/{split} has {len(rows)} page-disjoint rows, below {count}"
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            stable_seed(seed, f"{domain_id}:{split}:{row['file_name']}"),
            row["file_name"],
        ),
    )
    return ranked[:count]


def row_fingerprint(rows):
    payload = "\n".join(
        json.dumps(
            {
                "id": row["id"],
                "split": row["split"],
                "text": row["text"],
                "source_page": row["source_page"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in sorted(rows, key=lambda item: (item["split"], item["id"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_root}")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)

    parquet_files = sorted((args.source_root / "data").glob("*.parquet"))
    if len(parquet_files) != 4:
        raise ValueError(f"Expected four source parquet shards, found {len(parquet_files)}")
    selected_docs = {row["doc_id"]: row for row in config["domains"]}
    excluded_files = set(config.get("exclude_file_names", []))
    metadata_columns = [
        "file_name",
        "transcription",
        "source_page",
        "doc_id",
        "title_zh",
        "title_en",
        "title_abbr",
        "author",
        "compiler",
        "main_text_creation",
        "edition",
        "type",
        "library",
        "call_number",
    ]
    rows_by_doc = defaultdict(list)
    seen_files = set()
    for parquet_path in parquet_files:
        for batch in pq.ParquetFile(parquet_path).iter_batches(
            batch_size=1024, columns=metadata_columns
        ):
            for row in batch.to_pylist():
                if row["doc_id"] not in selected_docs:
                    continue
                if row["file_name"] in excluded_files:
                    continue
                if row["file_name"] in seen_files:
                    raise ValueError(f"Duplicate source file_name: {row['file_name']}")
                seen_files.add(row["file_name"])
                text = row["transcription"]
                if not text or text != text.strip() or any(char.isspace() for char in text):
                    raise ValueError(f"Invalid transcription for {row['file_name']}")
                rows_by_doc[row["doc_id"]].append(row)

    selected_by_file = {}
    domain_rows = {}
    suite_summary = {
        "suite": config["suite"],
        "source_repository": config["source_repository"],
        "source_revision": config["source_revision"],
        "source_license": config["source_license"],
        "excluded_source_records": sorted(excluded_files),
        "exclusion_reason": config.get("exclusion_reason", ""),
        "domains": {},
    }
    for doc_id, domain_config in selected_docs.items():
        source_rows = rows_by_doc[doc_id]
        page_rows = defaultdict(list)
        for row in source_rows:
            page_rows[row["source_page"]].append(row)
        assignments = split_weighted_pages(
            {page: len(rows) for page, rows in page_rows.items()},
            config["page_ratios"],
            int(config["split_seed"]),
            domain_config["domain_id"],
        )

        selected_rows = []
        split_pages = {}
        available = {}
        for split in SPLITS:
            split_pages[split] = set(assignments[split])
            candidates = [
                row for page in assignments[split] for row in page_rows[page]
            ]
            available[split] = len(candidates)
            chosen = select_rows(
                candidates,
                int(config["selected_counts"][split]),
                int(config["split_seed"]),
                domain_config["domain_id"],
                split,
            )
            for row in chosen:
                converted = {
                    "id": sample_id(domain_config["domain_id"], row["file_name"]),
                    "split": split,
                    "text": row["transcription"],
                    "direction": "vertical",
                    "source_page": row["source_page"],
                    "source_file_name": row["file_name"],
                    "doc_id": row["doc_id"],
                    "title_zh": row["title_zh"],
                    "title_en": row["title_en"],
                    "title_abbr": row["title_abbr"],
                    "author": row["author"],
                    "compiler": row["compiler"],
                    "main_text_creation": row["main_text_creation"],
                    "edition": row["edition"],
                    "type": row["type"],
                    "library": row["library"],
                    "call_number": row["call_number"],
                }
                selected_rows.append(converted)
                selected_by_file[row["file_name"]] = (
                    domain_config["domain_id"],
                    converted,
                )
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            if split_pages[left] & split_pages[right]:
                raise AssertionError(f"Page leakage in {domain_config['domain_id']}")
        domain_rows[domain_config["domain_id"]] = selected_rows
        suite_summary["domains"][domain_config["domain_id"]] = {
            "doc_id": doc_id,
            "title_zh": domain_config["title_zh"],
            "role": domain_config["role"],
            "source_lines": len(source_rows),
            "source_pages": len(page_rows),
            "available_after_page_split": available,
            "assigned_pages": {
                split: len(split_pages[split]) for split in SPLITS
            },
        }

    written = set()
    for parquet_path in parquet_files:
        for batch in pq.ParquetFile(parquet_path).iter_batches(
            batch_size=128, columns=["file_name", "image"]
        ):
            for source_row in batch.to_pylist():
                file_name = source_row["file_name"]
                if file_name not in selected_by_file:
                    continue
                domain_id, row = selected_by_file[file_name]
                output_dir = args.output_root / domain_id / "lines"
                output_dir.mkdir(parents=True, exist_ok=True)
                with Image.open(BytesIO(source_row["image"]["bytes"])) as image:
                    image = image.convert("RGB").transpose(Image.Transpose.ROTATE_90)
                    image.save(output_dir / f"{row['id']}.jpg", quality=95)
                    row["width"] = image.width
                    row["height"] = image.height
                written.add(file_name)
    if written != set(selected_by_file):
        missing = sorted(set(selected_by_file) - written)
        raise ValueError(f"Failed to materialize {len(missing)} selected images")

    for domain_id, rows in domain_rows.items():
        domain_root = args.output_root / domain_id
        ground_truth = {
            split: [
                {"id": row["id"], "text": row["text"], "direction": "vertical"}
                for row in rows
                if row["split"] == split
            ]
            for split in SPLITS
        }
        charset = sorted({char for row in rows for char in row["text"]})
        with (domain_root / "labels.pkl").open("wb") as handle:
            pickle.dump({"ground_truth": ground_truth}, handle)
        with (domain_root / "charset.pkl").open("wb") as handle:
            pickle.dump(charset, handle)
        with (domain_root / "columns_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda item: (SPLITS.index(item["split"]), item["id"])):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts = {split: len(ground_truth[split]) for split in SPLITS}
        page_counts = {
            split: len({row["source_page"] for row in rows if row["split"] == split})
            for split in SPLITS
        }
        meta = {
            **suite_summary["domains"][domain_id],
            "selected": counts,
            "selected_pages": page_counts,
            "charset_size": len(charset),
            "length_buckets": {
                split: dict(
                    Counter(_length_bucket(len(sample["text"])) for sample in ground_truth[split])
                )
                for split in SPLITS
            },
            "dataset_sha256": row_fingerprint(rows),
            "source_revision": config["source_revision"],
            "license": config["source_license"],
            "orientation_transform": config["orientation_transform"],
        }
        (domain_root / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        suite_summary["domains"][domain_id].update(meta)

    (args.output_root / "suite_summary.json").write_text(
        json.dumps(suite_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(suite_summary, ensure_ascii=False, indent=2))


def _length_bucket(length):
    if length == 1:
        return "1"
    if length == 2:
        return "2"
    if length <= 5:
        return "3-5"
    if length <= 10:
        return "6-10"
    return "11+"


if __name__ == "__main__":
    main()
