import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from make_qualitative_comparison_figures import DATASETS, get_original_image_and_boxes, load_font, load_saqt, wrap_cjk


def draw_vertical_text(draw, x, y, text, font, fill, step):
    for i, char in enumerate(text):
        draw.text((x, y + i * step), char, fill=fill, font=font)


def render_prediction_only(out_path, dataset_title, idx, image, pred):
    scale = min(3.8, 940 / max(image.height, 1), 250 / max(image.width, 1))
    scale = max(scale, 1.0)
    image_w = int(image.width * scale)
    image_h = int(image.height * scale)
    scaled = image.resize((image_w, image_h), Image.Resampling.LANCZOS)

    left_pad = 42
    top_pad = 112
    panel_gap = 120
    image_area_w = max(image_w, 430)
    text_panel_w = 680
    canvas_w = left_pad + image_area_w + panel_gap + text_panel_w
    canvas_h = max(top_pad + image_h + 100, 720)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    image_x = left_pad + (image_area_w - image_w) // 2
    canvas.paste(scaled, (image_x, top_pad))

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30)
    label_font = load_font(22)
    pred_font = load_font(28)
    small_font = load_font(17)

    panel_x = left_pad + image_area_w + panel_gap
    draw.text((left_pad, 28), f"{dataset_title} sample #{idx}", fill=(20, 31, 46), font=title_font)
    draw.text((image_x, top_pad - 30), "Input line image", fill=(55, 65, 81), font=small_font)
    draw.text((panel_x, 28), "SAQT recognition output", fill=(20, 31, 46), font=title_font)
    draw.text((panel_x, 86), "Predicted sequence", fill=(22, 124, 82), font=label_font)

    if len(pred) <= 24:
        draw_vertical_text(draw, panel_x + 18, 132, pred, pred_font, (22, 124, 82), 34)
    else:
        y = 132
        for line in wrap_cjk(pred, 24):
            draw.text((panel_x + 18, y), line, fill=(22, 124, 82), font=pred_font)
            y += 38

    draw.text(
        (panel_x, canvas_h - 44),
        "Model prediction only; GT text and GT boxes are not overlaid.",
        fill=(90, 90, 90),
        font=small_font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="docs/paper-drafts/figures/qualitative_examples/manifest.json")
    parser.add_argument("--output_dir", default="docs/paper-drafts/figures/saqt_recognition_outputs")
    parser.add_argument("--limit_per_dataset", type=int, default=6)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()

    datasets = {}
    written = []
    counts = {}
    for item in manifest:
        dataset_title = item["dataset"]
        dataset_key = "mthv2" if dataset_title == "MTHv2-combo" else "hdrc"
        counts.setdefault(dataset_key, 0)
        if counts[dataset_key] >= args.limit_per_dataset:
            continue
        if dataset_key not in datasets:
            spec = DATASETS[dataset_key]
            _, dataset, _, _ = load_saqt(spec["config"], spec["dataset_file"], spec["checkpoint"], "cpu")
            datasets[dataset_key] = dataset

        idx = int(item["index"])
        image, _, _ = get_original_image_and_boxes(datasets[dataset_key], idx)
        pred = item["saqt"]["pred"]
        counts[dataset_key] += 1
        out_path = output_dir / f"{dataset_key}_{counts[dataset_key]:02d}_idx{idx}_saqt_pred.png"
        render_prediction_only(out_path, dataset_title, idx, image, pred)
        written.append({"dataset": dataset_title, "index": idx, "prediction": pred, "figure": str(out_path)})

    (output_dir / "manifest.json").write_text(json.dumps(written, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(written)} SAQT recognition output figures to {output_dir}")


if __name__ == "__main__":
    main()
