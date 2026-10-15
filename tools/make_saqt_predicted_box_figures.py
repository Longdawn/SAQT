import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DTLR_MSDA_FORCE_PYTORCH", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

import util.misc as utils
from make_qualitative_comparison_figures import DATASETS, load_font, load_saqt
from util.box_ops import box_cxcywh_to_xyxy
from util.ctc_decoding import apply_ctc_calibration


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def charset_to_str(item):
    return item if isinstance(item, str) else chr(int(item))


def denormalize_pil(tensor, size_hw):
    h, w = [int(v) for v in size_hw]
    image = tensor.detach().cpu()[:, :h, :w] * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((image * 255).astype("uint8"))


def direction_name(target):
    direction = target.get("direction", 0)
    if torch.is_tensor(direction):
        direction = int(direction.reshape(-1)[0].item())
    return "vertical" if int(direction) == 1 else "horizontal"


def decoded_positions(token_ids):
    positions = []
    prev = None
    for pos, token in enumerate(token_ids):
        token = int(token)
        if token != prev and token != 0:
            positions.append((pos, token - 1))
        prev = token
    return positions


def boxes_to_pixels(boxes_cxcywh, size_hw):
    h, w = [float(v) for v in size_hw]
    scale = torch.tensor([w, h, w, h], dtype=boxes_cxcywh.dtype, device=boxes_cxcywh.device)
    return box_cxcywh_to_xyxy(boxes_cxcywh) * scale


def run_predictions(dataset, model, criterion, indices, device, blank_bias, nonblank_bias):
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
    outputs_by_index = {}
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for source_idx, (samples, targets) in zip(indices, loader):
            samples = samples.to(device)
            targets = [
                {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in target.items()}
                for target in targets
            ]
            target = targets[0]
            outputs = model(samples)
            _, pred_probs, _ = criterion.loss_CTC(outputs, targets, None, None, return_preds=True)
            scores = apply_ctc_calibration(
                pred_probs[0],
                target=target,
                blank_bias=blank_bias,
                nonblank_bias=nonblank_bias,
            )
            token_ids = scores.argmax(-1).detach().cpu().tolist()
            positions = decoded_positions(token_ids)

            pred_boxes = outputs["pred_boxes"][0]
            sort_axis = 1 if direction_name(target) == "vertical" else 0
            order_idx = torch.sort(pred_boxes[:, sort_axis], descending=False)[1]
            sorted_boxes = pred_boxes.index_select(0, order_idx)
            size_hw = target["size"].reshape(-1).detach().cpu().tolist()
            boxes_xyxy = boxes_to_pixels(sorted_boxes, size_hw).detach().cpu()
            image = denormalize_pil(samples.tensors[0], size_hw)

            decoded = []
            for pos, cls_idx in positions:
                if 0 <= cls_idx < len(dataset.charset) and pos < boxes_xyxy.shape[0]:
                    decoded.append(
                        {
                            "pos": pos,
                            "char": charset_to_str(dataset.charset[cls_idx]),
                            "box": [float(v) for v in boxes_xyxy[pos].tolist()],
                        }
                    )
            outputs_by_index[source_idx] = {"image": image, "decoded": decoded}
    return outputs_by_index


def render_case(out_path, dataset_title, idx, image, decoded):
    scale = min(4.0, 1200 / max(image.height, 1), 340 / max(image.width, 1))
    scale = max(scale, 1.0)
    image_w = int(image.width * scale)
    image_h = int(image.height * scale)
    scaled = image.resize((image_w, image_h), Image.Resampling.LANCZOS)

    left_pad = 42
    top_pad = 112
    label_pad = 220
    canvas_w = left_pad + image_w + label_pad + 28
    canvas_h = max(top_pad + image_h + 72, 720)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    canvas.paste(scaled, (left_pad, top_pad))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(28)
    label_font = load_font(18)
    char_font = load_font(20)
    note_font = load_font(14)
    colors = [
        (227, 74, 51),
        (49, 130, 189),
        (49, 163, 84),
        (117, 107, 177),
        (230, 171, 2),
        (166, 86, 40),
    ]

    draw.text((left_pad, 28), f"{dataset_title} sample #{idx}", fill=(20, 31, 46), font=title_font)
    draw.text((left_pad, top_pad - 30), "SAQT predicted boxes + chars", fill=(55, 65, 81), font=label_font)
    text_x = left_pad + image_w + 14
    for i, item in enumerate(decoded):
        color = colors[i % len(colors)]
        x0, y0, x1, y1 = [v * scale for v in item["box"]]
        x0 += left_pad
        x1 += left_pad
        y0 += top_pad
        y1 += top_pad
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        label_y = max(top_pad, min(canvas_h - 76, (y0 + y1) / 2 - 12))
        draw.text((text_x, label_y), item["char"], fill=color, font=char_font)

    draw.text(
        (left_pad, canvas_h - 40),
        "Predicted query boxes; no GT overlay.",
        fill=(90, 90, 90),
        font=note_font,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="docs/paper-drafts/figures/qualitative_examples/manifest.json")
    parser.add_argument("--output_dir", default="docs/paper-drafts/figures/saqt_predicted_boxes")
    parser.add_argument("--limit_per_dataset", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()

    by_dataset = {"mthv2": [], "hdrc": []}
    for item in manifest:
        key = "mthv2" if item["dataset"] == "MTHv2-combo" else "hdrc"
        if len(by_dataset[key]) < args.limit_per_dataset:
            by_dataset[key].append(item)

    written = []
    device = torch.device("cpu")
    for dataset_key, items in by_dataset.items():
        if not items:
            continue
        spec = DATASETS[dataset_key]
        _, dataset, model, criterion = load_saqt(spec["config"], spec["dataset_file"], spec["checkpoint"], device)
        indices = [int(item["index"]) for item in items]
        preds = run_predictions(
            dataset,
            model,
            criterion,
            indices,
            device,
            spec["blank_bias"],
            spec["nonblank_bias"],
        )
        for rank, item in enumerate(items, start=1):
            idx = int(item["index"])
            out_path = output_dir / f"{dataset_key}_{rank:02d}_idx{idx}_pred_boxes.png"
            render_case(out_path, spec["title"], idx, preds[idx]["image"], preds[idx]["decoded"])
            written.append(
                {
                    "dataset": spec["title"],
                    "index": idx,
                    "prediction": "".join(x["char"] for x in preds[idx]["decoded"]),
                    "figure": str(out_path),
                }
            )

    (output_dir / "manifest.json").write_text(json.dumps(written, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(written)} SAQT predicted-box figures to {output_dir}")


if __name__ == "__main__":
    main()
