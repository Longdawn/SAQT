#!/usr/bin/env python3
"""Validate an exported Gate-A review and apply its pre-registered decision rule."""

import argparse
import json
from collections import Counter
from pathlib import Path


DOMAINS = ("zhanguoce_song", "tangshi_xuanxuan", "shujiao_ming")
VALID_STATUSES = {"correct", "issue", "uncertain"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument(
        "--canonical-review-csv",
        type=Path,
        default=Path(
            "experiments/notes/style_transfer_annotation_audit_v3_0803/review.csv"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_canonical(path):
    import csv

    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["id"]: {
            "domain": row["source_id"],
            "split": row["split"],
            "selection_type": row["selection_type"],
        }
        for row in rows
    }


def decision(random_issues, random_uncertain, incomplete):
    if incomplete or random_uncertain:
        return "INCOMPLETE"
    if random_issues <= 1:
        return "GO_AFTER_RISK_CLEANING"
    if random_issues == 2:
        return "EXPAND_AUDIT"
    return "REJECT_OR_REBUILD"


def render_markdown(report):
    lines = [
        "# Style-Transfer Gate-A Review Decision",
        "",
        "| Domain | Reviewed | Random issues / 40 | Random uncertain | Risk issues | Decision |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain in DOMAINS:
        row = report["domains"][domain]
        lines.append(
            f"| {domain} | {row['reviewed']}/60 | {row['random_issues']}/40 | "
            f"{row['random_uncertain']} | {row['risk_issues']} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            f"Overall: **{report['overall_decision']}**.",
            "",
            "Risk-mined errors must be corrected or excluded and the engineering audit rerun before training.",
            "Only stratified-random rows determine the sampled annotation-noise gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    canonical = load_canonical(args.canonical_review_csv)
    review_rows = json.loads(args.review_json.read_text(encoding="utf-8"))
    if not isinstance(review_rows, list):
        raise ValueError("Review JSON must be a list")

    ids = [str(row.get("id", "")) for row in review_rows]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count > 1
    )
    unknown_ids = sorted(set(ids) - set(canonical))
    missing_ids = sorted(set(canonical) - set(ids))
    if duplicate_ids or unknown_ids or missing_ids:
        raise ValueError(
            f"Review identity mismatch: duplicates={len(duplicate_ids)} "
            f"unknown={len(unknown_ids)} missing={len(missing_ids)}"
        )

    by_domain = {domain: [] for domain in DOMAINS}
    normalized_rows = []
    for row in review_rows:
        sample_id = str(row["id"])
        expected = canonical[sample_id]
        for key in ("domain", "split", "selection_type"):
            if str(row.get(key, "")) != expected[key]:
                raise ValueError(f"Review metadata mismatch for {sample_id}: {key}")
        status = str(row.get("status", ""))
        if status and status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r} for {sample_id}")
        normalized = {
            **expected,
            "id": sample_id,
            "status": status,
            "corrected_text": str(row.get("corrected_text", "")),
            "review_notes": str(row.get("review_notes", "")),
        }
        normalized_rows.append(normalized)
        by_domain[expected["domain"]].append(normalized)

    domain_reports = {}
    for domain, rows in by_domain.items():
        random_rows = [
            row for row in rows if row["selection_type"] == "stratified_random"
        ]
        risk_rows = [row for row in rows if row["selection_type"] == "risk_mined"]
        if len(random_rows) != 40 or len(risk_rows) != 20:
            raise ValueError(
                f"Unexpected audit quotas for {domain}: random={len(random_rows)} risk={len(risk_rows)}"
            )
        incomplete = sum(not row["status"] for row in rows)
        random_issues = sum(row["status"] == "issue" for row in random_rows)
        random_uncertain = sum(row["status"] == "uncertain" for row in random_rows)
        risk_issues = sum(row["status"] == "issue" for row in risk_rows)
        domain_reports[domain] = {
            "reviewed": len(rows) - incomplete,
            "incomplete": incomplete,
            "random_issues": random_issues,
            "random_uncertain": random_uncertain,
            "risk_issues": risk_issues,
            "risk_uncertain": sum(
                row["status"] == "uncertain" for row in risk_rows
            ),
            "decision": decision(random_issues, random_uncertain, incomplete),
        }

    decisions = [row["decision"] for row in domain_reports.values()]
    if "INCOMPLETE" in decisions:
        overall = "INCOMPLETE"
    elif "REJECT_OR_REBUILD" in decisions:
        overall = "REJECT_OR_REBUILD"
    elif "EXPAND_AUDIT" in decisions:
        overall = "EXPAND_AUDIT"
    else:
        overall = "GO_AFTER_RISK_CLEANING"

    report = {
        "evidence_type": "human_review",
        "review_file": str(args.review_json),
        "canonical_review_csv": str(args.canonical_review_csv),
        "domains": domain_reports,
        "overall_decision": overall,
        "reviewed_rows": sum(row["reviewed"] for row in domain_reports.values()),
        "total_rows": len(canonical),
        "issue_rows": [
            row for row in normalized_rows if row["status"] == "issue"
        ],
        "uncertain_rows": [
            row for row in normalized_rows if row["status"] == "uncertain"
        ],
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if overall == "GO_AFTER_RISK_CLEANING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
