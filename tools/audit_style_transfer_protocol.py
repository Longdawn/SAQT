#!/usr/bin/env python3
"""Audit and fingerprint the three-domain low-resource transfer protocol."""

import argparse
import csv
import hashlib
import json
import pickle
import random
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image


DOMAINS = ("zhanguoce_song", "tangshi_xuanxuan", "shujiao_ming")
SPLITS = ("train", "valid", "test")
EXPECTED_COUNTS = {"train": 200, "valid": 30, "test": 60}
SUBSET_SEED = 20260803


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/style_stratified_transfer_pilot_v2_candidate"),
    )
    parser.add_argument(
        "--source-charset",
        type=Path,
        default=Path(
            "logs/mthv2_stage1_qbudget_from_mth1000mth1200_gpu0_0602/charset.pkl"
        ),
    )
    parser.add_argument(
        "--known-findings",
        type=Path,
        default=Path(
            "experiments/notes/curated_historical_qa_0803/manual_review_findings.tsv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/notes/style_transfer_protocol_audit_0803.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("experiments/notes/style_transfer_protocol_audit_0803.md"),
    )
    return parser.parse_args()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_rows(rows):
    payload = "\n".join(
        json.dumps(
            {"id": row["id"], "split": row["split"], "text": row["text"]},
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in sorted(rows, key=lambda item: (item["split"], item["id"]))
    )
    return sha256_bytes(payload.encode("utf-8"))


def fingerprint_ids(samples):
    return sha256_bytes(
        "\n".join(str(sample["id"]) for sample in samples).encode("utf-8")
    )


def length_bucket(length):
    if length == 1:
        return "len=1"
    if length == 2:
        return "len=2"
    if length <= 5:
        return "len=3-5"
    if length <= 10:
        return "len=6-10"
    return "len>=11"


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_known_ids(path):
    with path.open(encoding="utf-8-sig") as handle:
        return {
            row["sample_id"]
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("status") == "issue"
        }


def selected_50(samples):
    indices = sorted(random.Random(SUBSET_SEED).sample(range(len(samples)), 50))
    return [samples[index] for index in indices]


def audit_domain(root, domain, source_charset, known_ids):
    domain_root = root / domain
    with (domain_root / "labels.pkl").open("rb") as handle:
        labels = pickle.load(handle)["ground_truth"]
    manifest = load_jsonl(domain_root / "columns_manifest.jsonl")
    manifest_by_id = {row["id"]: row for row in manifest}
    failures = []
    warnings = []

    if len(manifest_by_id) != len(manifest):
        failures.append("duplicate IDs in columns_manifest.jsonl")

    label_rows = []
    split_ids = {}
    split_pages = {}
    split_report = {}
    for split in SPLITS:
        samples = labels[split]
        ids = [str(sample["id"]) for sample in samples]
        split_ids[split] = set(ids)
        if len(ids) != EXPECTED_COUNTS[split]:
            failures.append(
                f"{split} count {len(ids)} != expected {EXPECTED_COUNTS[split]}"
            )
        if len(set(ids)) != len(ids):
            failures.append(f"duplicate IDs in {split}")
        pages = set()
        for sample in samples:
            sample_id = str(sample["id"])
            row = manifest_by_id.get(sample_id)
            if row is None:
                failures.append(f"missing manifest row: {sample_id}")
                continue
            if row["split"] != split or row["text"] != sample["text"]:
                failures.append(f"label/manifest mismatch: {sample_id}")
            if sample.get("direction") != "vertical":
                failures.append(f"non-vertical direction: {sample_id}")
            pages.add(row["page_id"])
            label_rows.append(row)
        split_pages[split] = pages
        split_report[split] = {
            "samples": len(samples),
            "pages": len(pages),
            "id_sha256": sha256_bytes("\n".join(sorted(ids)).encode("utf-8")),
            "page_sha256": sha256_bytes(
                "\n".join(sorted(pages)).encode("utf-8")
            ),
            "length_buckets": dict(
                Counter(length_bucket(len(sample["text"])) for sample in samples)
            ),
        }

    overlaps = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        id_overlap = split_ids[left] & split_ids[right]
        page_overlap = split_pages[left] & split_pages[right]
        overlaps[f"{left}_vs_{right}"] = {
            "id_overlap": len(id_overlap),
            "page_overlap": len(page_overlap),
        }
        if id_overlap or page_overlap:
            failures.append(
                f"{left}/{right} overlap: ids={len(id_overlap)} pages={len(page_overlap)}"
            )

    selected_ids = set().union(*split_ids.values())
    known_overlap = selected_ids & known_ids
    if known_overlap:
        failures.append(f"contains {len(known_overlap)} known issue IDs")

    image_hashes = {}
    invalid_images = []
    dimension_mismatches = []
    aspect_ratios = []
    whitespace_ids = []
    control_ids = []
    empty_ids = []
    missing_source_files = []
    all_chars = set()
    for row in label_rows:
        sample_id = row["id"]
        text = row["text"]
        all_chars.update(text)
        if not text:
            empty_ids.append(sample_id)
        if any(char.isspace() for char in text):
            whitespace_ids.append(sample_id)
        if any(unicodedata.category(char).startswith("C") for char in text):
            control_ids.append(sample_id)
        if not Path(row["source_image"]).is_file() or not Path(row["source_xml"]).is_file():
            missing_source_files.append(sample_id)

        image_path = domain_root / "lines" / f"{sample_id}.jpg"
        if not image_path.is_file():
            invalid_images.append(sample_id)
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
        except Exception:
            invalid_images.append(sample_id)
            continue
        if width <= 0 or height <= 0:
            invalid_images.append(sample_id)
            continue
        aspect_ratios.append(height / width)
        x1, y1, x2, y2 = row["crop_box"]
        if (width, height) != (x2 - x1, y2 - y1):
            dimension_mismatches.append(sample_id)
        image_hashes[sample_id] = sha256_file(image_path)

    for name, values in (
        ("empty labels", empty_ids),
        ("whitespace labels", whitespace_ids),
        ("control-character labels", control_ids),
        ("invalid/missing images", invalid_images),
        ("crop dimension mismatches", dimension_mismatches),
        ("missing source files", missing_source_files),
    ):
        if values:
            failures.append(f"{name}: {len(values)}")

    duplicate_image_groups = [
        sorted(ids)
        for digest, ids in _invert(image_hashes).items()
        if len(ids) > 1
    ]
    if duplicate_image_groups:
        failures.append(f"duplicate image groups: {len(duplicate_image_groups)}")

    with (domain_root / "charset.pkl").open("rb") as handle:
        saved_charset = set(pickle.load(handle))
    if saved_charset != all_chars:
        failures.append("charset.pkl does not equal characters from all splits")

    mmocr_path = root / "mmocr_exports" / domain / "textrecog_all.json"
    mmocr = json.loads(mmocr_path.read_text(encoding="utf-8"))["data_list"]
    mmocr_map = {
        Path(row["img_path"]).stem: row["instances"][0]["text"] for row in mmocr
    }
    expected_map = {row["id"]: row["text"] for row in label_rows}
    if mmocr_map != expected_map:
        failures.append("MMOCR all-manifest differs from canonical labels")

    target_50 = selected_50(labels["train"])
    target_50_chars = {char for sample in target_50 for char in sample["text"]}
    eval_chars = {
        char
        for split in ("valid", "test")
        for sample in labels[split]
        for char in sample["text"]
    }
    source_overlap = all_chars & source_charset
    eval_unseen_target = eval_chars - target_50_chars
    eval_supported_only_by_source = eval_unseen_target & source_charset
    eval_unseen_both = eval_unseen_target - source_charset

    if min(aspect_ratios) < 1.0:
        warnings.append("at least one crop is wider than tall")
    return {
        "domain": domain,
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "splits": split_report,
        "overlaps": overlaps,
        "known_issue_overlap": len(known_overlap),
        "dataset_sha256": fingerprint_rows(label_rows),
        "image_manifest_sha256": sha256_bytes(
            "\n".join(
                f"{sample_id}\t{image_hashes[sample_id]}"
                for sample_id in sorted(image_hashes)
            ).encode("utf-8")
        ),
        "selected_50": {
            "seed": SUBSET_SEED,
            "samples": len(target_50),
            "ordered_id_sha256": fingerprint_ids(target_50),
            "ids": [sample["id"] for sample in target_50],
        },
        "charset": {
            "target_chars": len(all_chars),
            "source_overlap_chars": len(source_overlap),
            "source_overlap_percent": round(100 * len(source_overlap) / len(all_chars), 4),
            "target_50_chars": len(target_50_chars),
            "eval_chars": len(eval_chars),
            "eval_unseen_in_target_50": len(eval_unseen_target),
            "eval_unseen_target_but_in_source": len(eval_supported_only_by_source),
            "eval_unseen_in_target_and_source": len(eval_unseen_both),
        },
        "images": {
            "verified": len(image_hashes),
            "duplicate_groups": duplicate_image_groups,
            "min_aspect_ratio": round(min(aspect_ratios), 4),
            "median_aspect_ratio": round(
                sorted(aspect_ratios)[len(aspect_ratios) // 2], 4
            ),
            "max_aspect_ratio": round(max(aspect_ratios), 4),
        },
    }, image_hashes, expected_map


def _invert(mapping):
    output = {}
    for key, value in mapping.items():
        output.setdefault(value, []).append(key)
    return output


def render_markdown(report):
    lines = [
        "# Style-Transfer Protocol Audit",
        "",
        "This audit freezes the three-domain Gate-A/Gate-B candidate protocol.",
        "It verifies engineering integrity, not transcript correctness or licensing.",
        "",
        "| Domain | Train/Val/Test | Pages | Charset | Source overlap | Eval unseen in 50 / source-supported / unseen-both | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain in report["domains"]:
        split = domain["splits"]
        charset = domain["charset"]
        lines.append(
            f"| {domain['domain']} | {split['train']['samples']}/{split['valid']['samples']}/{split['test']['samples']} | "
            f"{split['train']['pages']}/{split['valid']['pages']}/{split['test']['pages']} | "
            f"{charset['target_chars']} | {charset['source_overlap_percent']:.2f}% | "
            f"{charset['eval_unseen_in_target_50']}/{charset['eval_unseen_target_but_in_source']}/{charset['eval_unseen_in_target_and_source']} | "
            f"{'PASS' if domain['passed'] else 'FAIL'} |"
        )
    lines.extend(["", "## Frozen 50-Column Fingerprints", ""])
    for domain in report["domains"]:
        lines.append(
            f"- `{domain['domain']}`: `{domain['selected_50']['ordered_id_sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Cross-Domain Checks",
            "",
            f"- Repeated IDs: `{report['cross_domain']['repeated_ids']}`",
            f"- Repeated image hashes: `{report['cross_domain']['repeated_image_hashes']}`",
            f"- Repeated exact transcripts (reported, not a failure): `{report['cross_domain']['repeated_texts']}`",
            "",
            "## Scope",
            "",
            "PASS proves deterministic split/data consistency, page disjointness, file readability, and frozen subset identity.",
            "It does not prove that every XML transcript is visually correct. Gate-A manual review remains mandatory.",
            "",
            f"Overall engineering audit: **{'PASS' if report['passed'] else 'FAIL'}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    with args.source_charset.open("rb") as handle:
        source_charset = set(pickle.load(handle))
    known_ids = load_known_ids(args.known_findings)

    domains = []
    all_image_hashes = {}
    all_texts = {}
    all_ids = {}
    for domain in DOMAINS:
        result, image_hashes, texts = audit_domain(
            args.dataset_root, domain, source_charset, known_ids
        )
        domains.append(result)
        for sample_id, digest in image_hashes.items():
            all_image_hashes[f"{domain}/{sample_id}"] = digest
        for sample_id, text in texts.items():
            all_texts[f"{domain}/{sample_id}"] = text
            all_ids.setdefault(sample_id, []).append(domain)

    repeated_ids = [key for key, values in all_ids.items() if len(values) > 1]
    repeated_images = [
        values for values in _invert(all_image_hashes).values() if len(values) > 1
    ]
    repeated_texts = [
        values
        for text, values in _invert(all_texts).items()
        if text and len(values) > 1
    ]
    cross_failures = []
    if repeated_ids:
        cross_failures.append(f"cross-domain repeated IDs: {len(repeated_ids)}")
    if repeated_images:
        cross_failures.append(
            f"cross-domain repeated image hashes: {len(repeated_images)}"
        )

    report = {
        "protocol": "style_stratified_transfer_pilot_v2_candidate",
        "evidence_type": "newly_run",
        "source_charset": str(args.source_charset),
        "known_findings": str(args.known_findings),
        "domains": domains,
        "cross_domain": {
            "repeated_ids": len(repeated_ids),
            "repeated_image_hashes": len(repeated_images),
            "repeated_texts": len(repeated_texts),
            "failures": cross_failures,
        },
        "passed": all(domain["passed"] for domain in domains) and not cross_failures,
        "manual_transcript_review_required": True,
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
