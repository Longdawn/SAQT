#!/usr/bin/env python3
"""Audit line-ID and page-group disjointness for recognition datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = {
    "HDRC": "data/hdrc_dtlr/labels.pkl",
    "MTH1000": "data/tkhmth2200_mth1000_dtlr/labels.pkl",
    "MTH1200": "data/tkhmth2200_mth1200_dtlr/labels.pkl",
    "TKH": "data/tkhmth2200_tkh_dtlr/labels.pkl",
}
SPLITS = ("train", "valid", "test")
PAIRS = (("train", "valid"), ("train", "test"), ("valid", "test"))


def digest(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def final_suffix_page_group(sample_id: str) -> str:
    if "_" not in sample_id:
        raise ValueError(f"sample ID lacks a line suffix separator: {sample_id}")
    return sample_id.rsplit("_", 1)[0]


def source_prefix_page_group(sample_id: str, source_page_ids: set[str]) -> str:
    """Recover the longest source-page stem that prefixes a processed line ID."""
    parts = sample_id.split("_")
    matches = {
        "_".join(parts[:end])
        for end in range(1, len(parts))
        if "_".join(parts[:end]) in source_page_ids
    }
    if not matches:
        raise ValueError(f"no source-page prefix matches processed line ID: {sample_id}")
    longest = max(matches, key=len)
    if sum(len(item) == len(longest) for item in matches) != 1:
        raise ValueError(f"ambiguous source-page prefix for processed line ID: {sample_id}")
    return longest


def hdrc_source_page_ids(labels_path: Path) -> set[str]:
    meta_path = labels_path.with_name("meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    image_root = ROOT / meta["image_root"]
    source_page_ids = {path.stem for path in image_root.iterdir() if path.is_file()}
    if not source_page_ids:
        raise ValueError(f"no HDRC source pages found under {image_root}")
    return source_page_ids


def audit_dataset(name: str, labels_path: Path) -> dict:
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    with labels_path.open("rb") as handle:
        ground_truth = pickle.load(handle)["ground_truth"]

    if name == "HDRC":
        source_page_ids = hdrc_source_page_ids(labels_path)
        group_fn = lambda sample_id: source_prefix_page_group(sample_id, source_page_ids)
        group_rule = "longest source-image stem prefix"
    else:
        group_fn = final_suffix_page_group
        group_rule = "sample_id.rsplit('_', 1)[0]"

    split_data = {}
    failures = []
    for split in SPLITS:
        samples = ground_truth[split]
        ids_list = [str(sample["id"]) for sample in samples]
        counts = Counter(ids_list)
        duplicate_ids = sorted(sample_id for sample_id, count in counts.items() if count > 1)
        ids = set(ids_list)
        groups = {group_fn(sample_id) for sample_id in ids}
        if duplicate_ids:
            failures.append(f"{split} has {len(duplicate_ids)} duplicate line IDs")
        split_data[split] = {
            "samples": len(samples),
            "unique_ids": len(ids),
            "page_groups": len(groups),
            "id_sha256": digest(ids),
            "page_group_sha256": digest(groups),
            "duplicate_ids": duplicate_ids[:20],
            "_ids": ids,
            "_groups": groups,
        }

    overlaps = {}
    for left, right in PAIRS:
        id_overlap = sorted(split_data[left]["_ids"] & split_data[right]["_ids"])
        group_overlap = sorted(split_data[left]["_groups"] & split_data[right]["_groups"])
        key = f"{left}_vs_{right}"
        overlaps[key] = {
            "line_id_overlap": len(id_overlap),
            "page_group_overlap": len(group_overlap),
            "line_id_examples": id_overlap[:20],
            "page_group_examples": group_overlap[:20],
        }
        if id_overlap or group_overlap:
            failures.append(
                f"{key}: line_id_overlap={len(id_overlap)}, page_group_overlap={len(group_overlap)}"
            )

    for split in SPLITS:
        split_data[split].pop("_ids")
        split_data[split].pop("_groups")
    return {
        "dataset": name,
        "labels_file": str(labels_path.relative_to(ROOT)),
        "group_rule": group_rule,
        "splits": split_data,
        "overlaps": overlaps,
        "passed": not failures,
        "failures": failures,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Recognition Dataset Split Audit",
        "",
        "Line IDs and source-page groups must be disjoint across train, valid, and test.",
        "For HDRC, source pages are recovered by matching the longest source-image stem prefix; other datasets use their final-line-suffix convention.",
        "",
        "| Dataset | Train lines/pages | Valid lines/pages | Test lines/pages | ID overlaps | Page overlaps | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for dataset in report["datasets"]:
        split = dataset["splits"]
        id_overlaps = sum(item["line_id_overlap"] for item in dataset["overlaps"].values())
        page_overlaps = sum(item["page_group_overlap"] for item in dataset["overlaps"].values())
        lines.append(
            f"| {dataset['dataset']} | {split['train']['samples']}/{split['train']['page_groups']} | "
            f"{split['valid']['samples']}/{split['valid']['page_groups']} | "
            f"{split['test']['samples']}/{split['test']['page_groups']} | "
            f"{id_overlaps} | {page_overlaps} | {'PASS' if dataset['passed'] else 'FAIL'} |"
        )
    lines.extend(["", "## Split Fingerprints", ""])
    for dataset in report["datasets"]:
        lines.append(f"### {dataset['dataset']}")
        lines.append("")
        for split_name in SPLITS:
            split = dataset["splits"][split_name]
            lines.append(
                f"- `{split_name}` line IDs: `{split['id_sha256']}`; "
                f"page groups: `{split['page_group_sha256']}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "A passing result establishes that no processed line ID or recovered source-page group is shared across splits.",
            "It does not by itself establish corpus licensing, source-edition independence, or the absence of near-duplicate image content.",
            "",
            f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "experiments/notes/dataset_split_audit_0718.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "experiments/notes/dataset_split_audit_0718.md",
    )
    args = parser.parse_args()

    datasets = [audit_dataset(name, ROOT / path) for name, path in DEFAULT_DATASETS.items()]
    report = {"schema_version": 1, "datasets": datasets, "passed": all(item["passed"] for item in datasets)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(
        f"dataset_split_audit: datasets={len(datasets)}, "
        f"passed={sum(item['passed'] for item in datasets)}, failures={sum(len(item['failures']) for item in datasets)}"
    )
    print(args.output_json.relative_to(ROOT))
    print(args.output_md.relative_to(ROOT))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
