import argparse
import json
import pickle
import sys
from bisect import bisect_right
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import util.misc as utils
from datasets import build_dataset
from finetuning import build_model_main
from util.ctc_decoding import apply_ctc_calibration, decode_greedy
from util.slconfig import SLConfig


FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


DATASETS = {
    "mthv2": {
        "title": "MTHv2-combo",
        "dataset_file": "mth_combo",
        "config": "config/SAQT_MTHV2.py",
        "checkpoint": "logs/mthv2_qbudgetstage1pre_mthv2_full_ctc_count001_0608/checkpoint_best_regular.pth",
        "blank_bias": -2.0,
        "nonblank_bias": 0.8,
        "models": {
            "MMOCR CRNN": "/home/ubuntu/mmocr/work_dirs/eval_mthv2_crnn_k3_epoch28_test_0530/best_recog_1-N.E.D_epoch_28.pth_predictions.pkl",
            "MMOCR SVTR-T": "/home/ubuntu/mmocr/work_dirs/eval_mthv2_svtr_tiny_k3_epoch29_test_0530/best_recog_1-N.E.D_epoch_29.pth_predictions.pkl",
            "MMOCR SVTR-L": "/home/ubuntu/mmocr/work_dirs/eval_mthv2_svtr_large_k3_officialgeom_epoch29_test_0601/best_recog_1-N.E.D_epoch_29.pth_predictions.pkl",
        },
    },
    "hdrc": {
        "title": "HDRC",
        "dataset_file": "mth1000",
        "config": "config/SAQT_HDRC.py",
        "checkpoint": "logs/hdrc_qbudget_full_0607/checkpoint_best_regular.pth",
        "blank_bias": -2.0,
        "nonblank_bias": 1.0,
        "models": {
            "MMOCR CRNN": "/home/ubuntu/mmocr/work_dirs/eval_hdrc_crnn_k3_best_epoch29_test_0603/best_recog_1-N.E.D_epoch_29.pth_predictions.pkl",
            "MMOCR SVTR-T": "/home/ubuntu/mmocr/work_dirs/eval_hdrc_svtr_tiny_k3_epoch30_test_0603/best_recog_1-N.E.D_epoch_30.pth_predictions.pkl",
            "MMOCR SVTR-L": "/home/ubuntu/mmocr/work_dirs/eval_hdrc_svtr_large_k3_officialgeom_epoch29_test_0603/best_recog_1-N.E.D_epoch_29.pth_predictions.pkl",
        },
    },
}


def load_font(size):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_args(config, dataset_file):
    cfg = SLConfig.fromfile(config)
    values = cfg._cfg_dict.to_dict()
    values.update(
        {
            "dataset_file": dataset_file,
            "mode_chr": True,
            "amp": False,
            "eval": True,
            "distributed": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        }
    )
    return argparse.Namespace(**values)


def adapt_class_head(model, charset_size, device):
    features_dim = model.class_embed[0].weight.data.shape[1]
    new_class_embed = nn.Linear(features_dim, charset_size)
    new_decoder_class_embed = nn.Linear(features_dim, charset_size)
    new_enc_out_class_embed = nn.Linear(features_dim, charset_size)
    if model.dec_pred_class_embed_share:
        class_embed_layerlist = [new_class_embed for _ in range(model.transformer.num_decoder_layers)]
    else:
        class_embed_layerlist = [
            nn.Linear(features_dim, charset_size)
            for _ in range(model.transformer.num_decoder_layers)
        ]
    model.class_embed = nn.ModuleList(class_embed_layerlist).to(device)
    model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
    model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
    model.label_enc = nn.Embedding(charset_size + 1, features_dim).to(device)


def load_saqt(config, dataset_file, checkpoint, device):
    args = load_args(config, dataset_file)
    dataset = build_dataset(image_set="test", args=args)
    args.charset = dataset.charset
    model, criterion, _ = build_model_main(args)
    adapt_class_head(model, len(dataset.charset), device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    clean_state = utils.clean_state_dict(state)
    model_state = model.state_dict()
    filtered = {
        key: value
        for key, value in clean_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(filtered, strict=False)
    model.to(device).eval()
    criterion.to(device).eval()
    return args, dataset, model, criterion


def text_cer(gt, pred):
    if not gt:
        return 0.0 if not pred else 1.0
    a = list(gt)
    b = list(pred)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    dist = prev[-1]
    return dist / max(len(gt), 1)


def pkl_text(row, key):
    value = row.get(key, {})
    if isinstance(value, dict):
        return str(value.get("item", ""))
    return str(value)


def load_mmocr_predictions(paths):
    loaded = {}
    for name, path in paths.items():
        loaded[name] = pickle.load(open(path, "rb"))
    return loaded


def select_candidates(mmocr_preds, limit):
    first_name = next(iter(mmocr_preds))
    n = len(mmocr_preds[first_name])
    rows = []
    for idx in range(n):
        gt = pkl_text(mmocr_preds[first_name][idx], "gt_text")
        if len(gt) < 5 or len(gt) > 24:
            continue
        if any(pkl_text(preds[idx], "gt_text") != gt for preds in mmocr_preds.values()):
            continue
        cers = {
            name: text_cer(gt, pkl_text(preds[idx], "pred_text"))
            for name, preds in mmocr_preds.items()
        }
        if max(cers.values()) < 0.12:
            continue
        rows.append((sum(cers.values()) / len(cers), max(cers.values()), idx, gt, cers))
    rows.sort(reverse=True)
    return rows[:limit]


def labels_to_text(labels, charset):
    return "".join(str(charset[int(i)]) for i in labels)


def run_saqt_predictions(dataset, model, criterion, candidate_rows, device, blank_bias, nonblank_bias):
    indices = [row[2] for row in candidate_rows]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=4, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
    outputs_by_index = {}
    cursor = 0
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = [
                {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in target.items()}
                for target in targets
            ]
            outputs = model(samples)
            _, pred_probs, _ = criterion.loss_CTC(outputs, targets, None, None, return_preds=True)
            for batch_i, target in enumerate(targets):
                source_idx = indices[cursor]
                cursor += 1
                scores = apply_ctc_calibration(
                    pred_probs[batch_i],
                    target=target,
                    blank_bias=blank_bias,
                    nonblank_bias=nonblank_bias,
                )
                pred_labels = decode_greedy(scores, len(dataset.charset))
                gt_labels = [int(x) for x in target["labels"].detach().cpu().tolist()]
                pred = labels_to_text(pred_labels, dataset.charset)
                gt = labels_to_text(gt_labels, dataset.charset)
                outputs_by_index[source_idx] = {
                    "gt": gt,
                    "pred": pred,
                    "cer": text_cer(gt, pred),
                }
    return outputs_by_index


def resolve_child_dataset(dataset, idx):
    if hasattr(dataset, "datasets") and hasattr(dataset, "cumulative_sizes"):
        dataset_idx = bisect_right(dataset.cumulative_sizes, idx)
        local_idx = idx - (dataset.cumulative_sizes[dataset_idx - 1] if dataset_idx else 0)
        return dataset.datasets[dataset_idx], local_idx
    return dataset, idx


def get_original_image_and_boxes(dataset, idx):
    child, local_idx = resolve_child_dataset(dataset, idx)
    sample = child.samples[local_idx]
    image_path = Path(child.images_path) / f"{sample['id']}.{child.im_extension}"
    image = Image.open(image_path).convert("RGB")
    direction = child._normalize_direction(sample, image=image)

    old_use = child.use_char_boxes
    has_raw = Path(child.textline_path).is_dir() and Path(child.char_path).is_dir()
    child.use_char_boxes = has_raw
    boxes = child._maybe_real_char_boxes(sample, image, direction) if has_raw else None
    child.use_char_boxes = old_use

    if boxes is not None and len(boxes) == len(sample["text"]):
        return image, boxes.tolist(), "raw"

    width, height = image.size
    n = max(len(sample["text"]), 1)
    step = height / n
    margin_x = max(1.0, width * 0.08)
    boxes = []
    for i in range(n):
        y1 = max(0.0, i * step)
        y2 = min(float(height), (i + 1) * step)
        boxes.append([margin_x, y1, width - margin_x, y2])
    return image, boxes, "estimated"


def wrap_cjk(text, width):
    if len(text) <= width:
        return [text]
    return [text[i : i + width] for i in range(0, len(text), width)]


def draw_text_block(draw, xy, label, text, font_label, font_text, color, max_chars=28):
    x, y = xy
    draw.text((x, y), label, fill=color, font=font_label)
    y += 30
    for line in wrap_cjk(text, max_chars):
        draw.text((x, y), line, fill=(25, 25, 25), font=font_text)
        y += 28
    return y + 10


def render_case(out_path, dataset_title, idx, image, boxes, box_source, gt, saqt, baselines):
    scale = min(3.2, 900 / max(image.height, 1), 220 / max(image.width, 1))
    scale = max(scale, 1.0)
    image_w = int(image.width * scale)
    image_h = int(image.height * scale)
    scaled = image.resize((image_w, image_h), Image.Resampling.LANCZOS)

    left_pad = 34
    top_pad = 112
    label_pad = 56
    panel_gap = 42
    image_area_w = max((image_w * 2) + panel_gap + label_pad, 620)
    panel_w = 760
    canvas_w = left_pad + image_area_w + panel_w
    canvas_h = max(top_pad + image_h + 64, 720)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    original_x = left_pad
    boxed_x = left_pad + image_w + panel_gap
    canvas.paste(scaled, (original_x, top_pad))
    canvas.paste(scaled, (boxed_x, top_pad))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(30)
    label_font = load_font(22)
    small_font = load_font(18)
    cjk_font = load_font(24)
    tiny_font = load_font(14)

    x_panel = left_pad + image_area_w
    draw.text((left_pad, 28), f"{dataset_title} sample #{idx}", fill=(20, 31, 46), font=title_font)
    draw.text((x_panel, 28), "Recognition comparison", fill=(20, 31, 46), font=title_font)
    draw.text((original_x, top_pad - 28), "Original", fill=(55, 65, 81), font=small_font)
    draw.text((boxed_x, top_pad - 28), "Structure cues", fill=(55, 65, 81), font=small_font)

    colors = [
        (227, 74, 51),
        (49, 130, 189),
        (49, 163, 84),
        (117, 107, 177),
        (230, 171, 2),
        (166, 86, 40),
    ]
    for i, (box, ch) in enumerate(zip(boxes, gt)):
        x1, y1, x2, y2 = [v * scale for v in box]
        x1 += boxed_x
        x2 += boxed_x
        y1 += top_pad
        y2 += top_pad
        color = colors[i % len(colors)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((boxed_x + image_w + 8, max(top_pad, y1 - 2)), ch, fill=color, font=small_font)

    draw.text(
        (left_pad, top_pad + image_h + 8),
        f"Character structure cues for visualization only: {box_source}",
        fill=(80, 80, 80),
        font=tiny_font,
    )

    y = 84
    y = draw_text_block(draw, (x_panel, y), "GT", gt, label_font, cjk_font, (40, 40, 40))
    y = draw_text_block(
        draw,
        (x_panel, y),
        f"SAQT  CER={saqt['cer']:.2f}",
        saqt["pred"],
        label_font,
        cjk_font,
        (22, 124, 82),
    )
    for name, pred in baselines.items():
        y = draw_text_block(
            draw,
            (x_panel, y),
            f"{name}  CER={text_cer(gt, pred):.2f}",
            pred,
            label_font,
            cjk_font,
            (150, 57, 57),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_dataset_figures(dataset_key, output_dir, candidates, examples):
    spec = DATASETS[dataset_key]
    mmocr = load_mmocr_predictions(spec["models"])
    candidate_rows = select_candidates(mmocr, candidates)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, dataset, model, criterion = load_saqt(spec["config"], spec["dataset_file"], spec["checkpoint"], device)
    saqt_outputs = run_saqt_predictions(
        dataset,
        model,
        criterion,
        candidate_rows,
        device,
        spec["blank_bias"],
        spec["nonblank_bias"],
    )

    selected = []
    for _, _, idx, gt, _ in candidate_rows:
        saqt = saqt_outputs[idx]
        baseline_texts = {name: pkl_text(preds[idx], "pred_text") for name, preds in mmocr.items()}
        baseline_cers = [text_cer(gt, pred) for pred in baseline_texts.values()]
        if (
            saqt["cer"] <= 0.05
            and min(baseline_cers) > saqt["cer"] + 1e-6
            and max(baseline_cers) >= 0.15
        ):
            selected.append((idx, gt, saqt, baseline_texts))
        if len(selected) >= examples:
            break
    if len(selected) < examples:
        ranked = sorted(
            (
                (
                    min(
                        text_cer(row[3], pkl_text(preds[row[2]], "pred_text"))
                        for preds in mmocr.values()
                    )
                    - saqt_outputs[row[2]]["cer"],
                    -saqt_outputs[row[2]]["cer"],
                    row[0],
                    row[2],
                    row[3],
                    saqt_outputs[row[2]],
                    {name: pkl_text(preds[row[2]], "pred_text") for name, preds in mmocr.items()},
                )
                for row in candidate_rows
            ),
            key=lambda item: (item[0], item[1], item[2]),
            reverse=True,
        )
        for _, _, _, idx, gt, saqt, baseline_texts in ranked:
            if idx not in {x[0] for x in selected}:
                selected.append((idx, gt, saqt, baseline_texts))
            if len(selected) >= examples:
                break

    metadata = []
    for rank, (idx, gt, saqt, baselines) in enumerate(selected, start=1):
        image, boxes, box_source = get_original_image_and_boxes(dataset, idx)
        out_path = output_dir / f"{dataset_key}_{rank:02d}_idx{idx}.png"
        render_case(out_path, spec["title"], idx, image, boxes, box_source, gt, saqt, baselines)
        metadata.append(
            {
                "dataset": spec["title"],
                "index": idx,
                "figure": str(out_path),
                "box_source": box_source,
                "gt": gt,
                "saqt": saqt,
                "mmocr": baselines,
            }
        )
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="docs/paper-drafts/figures/qualitative_examples")
    parser.add_argument("--datasets", nargs="+", default=["mthv2", "hdrc"], choices=sorted(DATASETS))
    parser.add_argument("--candidates", type=int, default=360)
    parser.add_argument("--examples_per_dataset", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("mthv2_*.png", "hdrc_*.png"):
        for old_png in output_dir.glob(pattern):
            old_png.unlink()

    all_metadata = []
    for dataset_key in args.datasets:
        all_metadata.extend(
            build_dataset_figures(
                dataset_key,
                output_dir,
                candidates=args.candidates,
                examples=args.examples_per_dataset,
            )
        )

    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(all_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(all_metadata)} qualitative figures to {output_dir}")
    print(f"Saved manifest to {manifest}")


if __name__ == "__main__":
    main()
