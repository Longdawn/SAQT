import argparse
import hashlib
import json
import pickle
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prepare_curated_historical_columns import (
    SPLITS,
    build_candidates,
    crop_box,
    load_catalog,
    load_excluded_sample_ids,
    stable_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Prepare independent target domains for cross-style transfer"
    )
    parser.add_argument(
        "--suite-config",
        type=Path,
        default=Path("experiments/configs/cross_style_transfer_suite_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cross_style_transfer_suite_v1_candidate"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def fingerprint(sample_ids):
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def select_fixed_candidates(candidates, count, seed, domain_id, split):
    if len(candidates) < count:
        raise ValueError(
            f"{domain_id}/{split} has {len(candidates)} candidates, needs {count}"
        )
    indices = list(range(len(candidates)))
    random.Random(stable_seed(seed, f"transfer:{domain_id}:{split}")).shuffle(indices)
    return [candidates[index] for index in sorted(indices[:count])]


def nested_train_manifests(train_records, budgets, seed, domain_id):
    if not budgets or sorted(set(budgets)) != list(budgets):
        raise ValueError("nested_train_budgets must be sorted and unique")
    if budgets[-1] > len(train_records):
        raise ValueError(
            f"largest budget {budgets[-1]} exceeds train size {len(train_records)}"
        )
    order = list(range(len(train_records)))
    random.Random(stable_seed(seed, f"nested:{domain_id}")).shuffle(order)
    manifests = {}
    for budget in budgets:
        selected = sorted(order[:budget])
        sample_ids = [train_records[index]["id"] for index in selected]
        manifests[str(budget)] = {
            "count": budget,
            "ids": sample_ids,
            "sha256": fingerprint(sample_ids),
        }
    full_ids = [record["id"] for record in train_records]
    manifests["full"] = {
        "count": len(full_ids),
        "ids": full_ids,
        "sha256": fingerprint(full_ids),
    }
    return manifests


def first_round_verified_ids(suite, excluded_ids):
    review_path = suite.get("first_round_review_csv")
    if not review_path:
        return set()
    import csv

    with resolve_project_path(review_path).open("r", encoding="utf-8-sig") as handle:
        reviewed = {row["id"] for row in csv.DictReader(handle)}
    return reviewed - excluded_ids


def build_suite(suite_path, output_root, overwrite=False, audit_only=False):
    suite_path = resolve_project_path(suite_path)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    source_catalog_path = resolve_project_path(suite["source_catalog"])
    source_catalog = load_catalog(source_catalog_path)
    source_catalog["split_seed"] = int(suite["split_seed"])
    sources = build_candidates(source_catalog, source_catalog_path)
    excluded_ids = load_excluded_sample_ids(source_catalog, source_catalog_path)
    verified_ids = first_round_verified_ids(suite, excluded_ids)
    counts = dict(suite["per_domain_counts"])
    for split in ("valid", "test"):
        counts[split] = int(counts[split])
    budgets = [int(value) for value in suite["nested_train_budgets"]]

    selected_by_domain = {}
    audit = {
        "suite": suite["suite"],
        "status": "candidate_pending_qa",
        "split_seed": suite["split_seed"],
        "per_domain_counts": counts,
        "nested_train_budgets": budgets,
        "manual_exclusions": len(excluded_ids),
        "domains": {},
    }
    for domain_id in suite["main_target_domains"]:
        source_data = sources[domain_id]
        selected = {}
        for split in SPLITS:
            if split == "train" and counts[split] == "all":
                selected[split] = list(source_data["candidates"][split])
            else:
                selected[split] = select_fixed_candidates(
                    source_data["candidates"][split],
                    int(counts[split]),
                    suite["split_seed"],
                    domain_id,
                    split,
                )
        selected_by_domain[domain_id] = selected
        page_sets = {
            split: {record["page_id"] for record in selected[split]}
            for split in SPLITS
        }
        overlap = sum(
            len(page_sets[left] & page_sets[right])
            for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
        )
        qa_counts = Counter(
            "verified_good" if record["id"] in verified_ids else "pending"
            for split in ("valid", "test")
            for record in selected[split]
        )
        audit["domains"][domain_id] = {
            "title": source_data["metadata"]["title"],
            "available_candidates": {
                split: len(source_data["candidates"][split]) for split in SPLITS
            },
            "selected": {split: len(selected[split]) for split in SPLITS},
            "selected_pages": {split: len(page_sets[split]) for split in SPLITS},
            "page_overlap": overlap,
            "valid_test_qa": dict(qa_counts),
        }

    if audit_only:
        return audit
    output_root = resolve_project_path(output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    for domain_id, selected in selected_by_domain.items():
        domain_root = output_root / domain_id
        lines_root = domain_root / "lines"
        lines_root.mkdir(parents=True)
        labels = {"ground_truth": {split: [] for split in SPLITS}}
        charset = Counter()
        manifest_rows = []
        source_metadata = sources[domain_id]["metadata"]
        for split in SPLITS:
            for record in selected[split]:
                with Image.open(record["image_path"]) as image:
                    image = image.convert("RGB")
                    bbox = crop_box(
                        record,
                        image.size,
                        source_catalog["margin_ratio"],
                        source_catalog["min_margin"],
                    )
                    image.crop(bbox).save(
                        lines_root / f"{record['id']}.jpg", quality=95
                    )
                labels["ground_truth"][split].append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "direction": "vertical",
                    }
                )
                charset.update(record["text"])
                qa_status = (
                    "verified_good" if record["id"] in verified_ids else "pending"
                )
                manifest_rows.append(
                    {
                        "id": record["id"],
                        "split": split,
                        "text": record["text"],
                        "text_length": len(record["text"]),
                        "source_id": domain_id,
                        "title": source_metadata["title"],
                        "page_id": record["page_id"],
                        "column_index_right_to_left": record["column_index"],
                        "source_image": str(record["image_path"]),
                        "source_xml": str(record["xml_path"]),
                        "crop_box": list(bbox),
                        "qa_status": qa_status,
                    }
                )
        nested = nested_train_manifests(
            selected["train"], budgets, suite["split_seed"], domain_id
        )
        with (domain_root / "labels.pkl").open("wb") as handle:
            pickle.dump(labels, handle)
        with (domain_root / "charset.pkl").open("wb") as handle:
            pickle.dump(sorted(charset), handle)
        with (domain_root / "columns_manifest.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (domain_root / "nested_train_subsets.json").write_text(
            json.dumps(nested, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        domain_meta = {**audit["domains"][domain_id], "domain_id": domain_id}
        (domain_root / "meta.json").write_text(
            json.dumps(domain_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (output_root / "suite_meta.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def main():
    args = parse_args()
    audit = build_suite(
        args.suite_config,
        args.output_root,
        overwrite=args.overwrite,
        audit_only=args.audit_only,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
