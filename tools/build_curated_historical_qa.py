import argparse
import csv
import html
import json
import pickle
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        "Build a balanced high-risk review set for CuratedHistoricalColumns"
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/curated_historical_v1_dtlr")
    )
    parser.add_argument(
        "--source-charset",
        type=Path,
        default=Path(
            "logs/mthv2_stage1_qbudget_from_mth1000mth1200_gpu0_0602/charset.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/notes/curated_historical_qa_0803"),
    )
    parser.add_argument("--per-source-split", type=int, default=10)
    parser.add_argument("--selection", choices=["risk", "random"], default="risk")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--exclude-review-csv",
        type=Path,
        default=None,
        help="Exclude ids already present in a previous review CSV.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def risk_features(row, image_size, source_charset, char_frequency):
    width, height = image_size
    text = row["text"]
    oov = [char for char in text if char not in source_charset]
    rare = [char for char in text if char_frequency[char] <= 2]
    ascii_chars = [char for char in text if char.isascii()]
    aspect = height / max(width, 1)
    score = 0.0
    score += 8.0 * len(oov) / max(len(text), 1)
    score += 2.0 * len(rare) / max(len(text), 1)
    score += 4.0 if ascii_chars else 0.0
    score += 3.0 if len(text) <= 3 else 0.0
    score += 2.0 if len(text) >= 30 else 0.0
    score += 2.0 if aspect < 2.0 else 0.0
    score += 1.0 if width >= 300 or height >= 3000 else 0.0
    return {
        "risk_score": round(score, 6),
        "oov_chars": "".join(dict.fromkeys(oov)),
        "rare_chars": "".join(dict.fromkeys(rare)),
        "ascii_chars": "".join(dict.fromkeys(ascii_chars)),
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect, 4),
    }


def main():
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    thumbs_dir = args.output_dir / "thumbs"
    thumbs_dir.mkdir(parents=True)

    manifest_path = args.dataset_root / "columns_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    rows = [row for row in rows if row["split"] in {"valid", "test"}]
    char_frequency = Counter(char for row in rows for char in row["text"])
    with args.source_charset.open("rb") as handle:
        source_charset = set(pickle.load(handle))

    enriched = []
    for row in rows:
        image_path = args.dataset_root / "lines" / f"{row['id']}.jpg"
        with Image.open(image_path) as image:
            features = risk_features(row, image.size, source_charset, char_frequency)
        enriched.append({**row, **features, "image_path": image_path})

    groups = defaultdict(list)
    for row in enriched:
        groups[(row["source_id"], row["split"])].append(row)

    excluded_ids = set()
    if args.exclude_review_csv is not None:
        with args.exclude_review_csv.open("r", encoding="utf-8-sig") as handle:
            excluded_ids = {row["id"] for row in csv.DictReader(handle)}

    selected = []
    for key in sorted(groups):
        candidates = [row for row in groups[key] if row["id"] not in excluded_ids]
        if args.selection == "risk":
            ranked = sorted(
                candidates, key=lambda row: (-row["risk_score"], row["id"])
            )
        else:
            ranked = sorted(candidates, key=lambda row: row["id"])
            group_seed = args.seed + sum(ord(char) for char in ":".join(key))
            random.Random(group_seed).shuffle(ranked)
        selected.extend(ranked[: args.per_source_split])
    selected.sort(key=lambda row: (row["source_id"], row["split"], -row["risk_score"], row["id"]))

    for row in selected:
        thumb_path = thumbs_dir / f"{row['id']}.jpg"
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((240, 1200), Image.Resampling.LANCZOS)
            image.save(thumb_path, quality=90)
        row["thumbnail"] = f"thumbs/{thumb_path.name}"

    columns = [
        "review_status",
        "id",
        "source_id",
        "title",
        "split",
        "text",
        "text_length",
        "risk_score",
        "oov_chars",
        "rare_chars",
        "ascii_chars",
        "width",
        "height",
        "aspect_ratio",
        "page_id",
        "column_index_right_to_left",
        "thumbnail",
        "review_notes",
    ]
    with (args.output_dir / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow({**row, "review_status": "pending", "review_notes": ""})

    cards = []
    for row in selected:
        risk = []
        if row["oov_chars"]:
            risk.append(f"MTHv2-OOV: {html.escape(row['oov_chars'])}")
        if row["rare_chars"]:
            risk.append(f"rare: {html.escape(row['rare_chars'])}")
        if row["ascii_chars"]:
            risk.append(f"ASCII: {html.escape(row['ascii_chars'])}")
        cards.append(
            "<article>"
            f"<img src='{html.escape(row['thumbnail'])}' alt='{html.escape(row['id'])}'>"
            "<div class='meta'>"
            f"<h2>{html.escape(row['title'])} · {row['split']}</h2>"
            f"<p class='text'>{html.escape(row['text'])}</p>"
            f"<p><code>{html.escape(row['id'])}</code></p>"
            f"<p>risk={row['risk_score']:.2f}; length={row['text_length']}; "
            f"size={row['width']}×{row['height']}</p>"
            f"<p class='risk'>{'; '.join(risk) or 'geometry/length review'}</p>"
            "<p>□ 顺序正确　□ 无串列　□ 标签合理</p>"
            "</div></article>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>CuratedHistoricalColumns QA</title>
<style>
body {{ font-family: sans-serif; margin: 20px; color: #161616; }}
header {{ max-width: 1000px; margin-bottom: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill,minmax(360px,1fr)); gap: 12px; }}
article {{ display: grid; grid-template-columns: 120px 1fr; gap: 12px; border: 1px solid #bbb; padding: 10px; }}
img {{ width: 120px; max-height: 600px; object-fit: contain; object-position: top; background: #eee; }}
h2 {{ font-size: 15px; margin: 0 0 6px; }} p {{ margin: 5px 0; }}
.text {{ font-size: 18px; line-height: 1.5; overflow-wrap: anywhere; }}
.risk {{ color: #9a3412; }} code {{ font-size: 11px; }}
</style></head><body>
<header><h1>CuratedHistoricalColumns v1 {html.escape(args.selection)} 抽检</h1>
<p>共 {len(selected)} 项；每个来源的 valid/test 各取
{args.per_source_split} 项。检查从上到下字符顺序、相邻列串入和标签错误。</p></header>
<main class="grid">{''.join(cards)}</main></body></html>"""
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")

    summary = {
        "candidate_valid_test": len(rows),
        "selected_for_review": len(selected),
        "per_source_split": args.per_source_split,
        "selection_mode": args.selection,
        "seed": args.seed,
        "excluded_previous_ids": len(excluded_ids),
        "selection": dict(Counter(f"{row['source_id']}:{row['split']}" for row in selected)),
        "status": "pending_manual_review",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
