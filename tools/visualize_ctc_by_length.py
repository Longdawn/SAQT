import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
from matplotlib.font_manager import FontProperties
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_dataset
from finetuning import build_model_main
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from util.slconfig import DictAction, SLConfig


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def parse_args():
    parser = argparse.ArgumentParser("Visualize CTC decoded boxes by GT text length")
    parser.add_argument("--config_file", "-c", required=True)
    parser.add_argument("--dataset_file", default="mth1000")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--samples_per_bin", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate_topk", type=int, default=20)
    parser.add_argument("--new_class_embedding", action="store_true")
    parser.add_argument("--show_gt_boxes", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--options", nargs="+", action=DictAction)
    return parser.parse_args()


def load_cfg_to_args(cli):
    cfg = SLConfig.fromfile(cli.config_file)
    if cli.options is not None:
        cfg.merge_from_dict(cli.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    base = vars(cli).copy()
    for key, value in cfg_dict.items():
        if key not in base:
            base[key] = value
    defaults = {
        "mode_chr": True,
        "amp": False,
        "eval": True,
        "distributed": False,
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "fix_size": False,
        "coco_path": "/comp_robot/cv_public_dataset/COCO2017/",
    }
    for key, value in defaults.items():
        if key not in base:
            base[key] = value
    return argparse.Namespace(**base)


def adapt_class_head(model, charset_size, device):
    features_dim = model.class_embed[0].weight.data.shape[1]
    if model.dec_pred_class_embed_share:
        class_embed_layerlist = [nn.Linear(features_dim, charset_size) for _ in range(model.transformer.num_decoder_layers)]
        shared = class_embed_layerlist[0]
        class_embed_layerlist = [shared for _ in range(model.transformer.num_decoder_layers)]
    else:
        class_embed_layerlist = [nn.Linear(features_dim, charset_size) for _ in range(model.transformer.num_decoder_layers)]
    model.class_embed = nn.ModuleList(class_embed_layerlist).to(device)
    model.transformer.decoder.class_embed = nn.Linear(features_dim, charset_size).to(device)
    model.transformer.enc_out_class_embed = nn.Linear(features_dim, charset_size).to(device)
    model.label_enc = nn.Embedding(charset_size + 1, features_dim).to(device)


def load_compatible_state(model, checkpoint_model):
    clean_state = utils.clean_state_dict(checkpoint_model)
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in clean_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(filtered, strict=False)
    return skipped


def length_bin(length):
    if length == 1:
        return "1"
    if length == 2:
        return "2"
    if length <= 5:
        return "3-5"
    if length <= 10:
        return "6-10"
    return "11+"


def charset_to_str(item, mode_chr=True):
    if isinstance(item, str):
        return item
    return chr(int(item)) if mode_chr else str(item)


def denormalize_image(tensor):
    image = tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def direction_name(target):
    direction = target.get("direction", 0)
    if torch.is_tensor(direction):
        direction = int(direction.reshape(-1)[0].item())
    else:
        direction = int(direction)
    return "vertical" if direction == 1 else "horizontal"


def sort_boxes_by_direction(boxes, direction):
    axis = 1 if direction == "vertical" else 0
    return torch.sort(boxes[:, axis], descending=False)[1]


def boxes_to_xyxy_pixels(boxes_cxcywh, size_hw):
    h, w = int(size_hw[0]), int(size_hw[1])
    scale = torch.tensor([w, h, w, h], dtype=boxes_cxcywh.dtype, device=boxes_cxcywh.device)
    return box_cxcywh_to_xyxy(boxes_cxcywh) * scale


def target_boxes_to_xyxy_pixels(target_boxes, size_hw):
    if target_boxes.numel() == 0:
        return target_boxes
    h, w = int(size_hw[0]), int(size_hw[1])
    if float(target_boxes.min()) >= 0.0 and float(target_boxes.max()) <= 1.5:
        boxes = target_boxes
    else:
        boxes = box_xyxy_to_cxcywh(target_boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=target_boxes.dtype, device=target_boxes.device)
    return boxes_to_xyxy_pixels(boxes, size_hw)


def collapse_ctc(tokens):
    collapsed = []
    prev = None
    for pos, token in enumerate(tokens):
        if token != prev:
            collapsed.append((pos, token))
        prev = token
    return [(pos, token) for pos, token in collapsed if token != 0]


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sample_indices_by_length(dataset, samples_per_bin, seed):
    buckets = {key: [] for key in ["1", "2", "3-5", "6-10", "11+"]}
    if hasattr(dataset, "samples"):
        for idx, sample in enumerate(dataset.samples):
            text = sample.get("text", "")
            buckets[length_bin(len(text))].append(idx)
    else:
        for idx in range(len(dataset)):
            _, target = dataset[idx]
            buckets[length_bin(len(target["labels"]))].append(idx)

    rng = random.Random(seed)
    selected = []
    for key in ["1", "2", "3-5", "6-10", "11+"]:
        candidates = buckets[key]
        rng.shuffle(candidates)
        selected.extend((key, idx) for idx in candidates[:samples_per_bin])
    return selected


def render(image_np, gt_text, pred_text, cer, decoded_items, candidate_items, gt_boxes, save_path):
    font = FontProperties(fname=CJK_FONT, size=8) if Path(CJK_FONT).exists() else None
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(image_np)
    ax.axis("off")

    if gt_boxes is not None and gt_boxes.numel() > 0:
        for box in gt_boxes.cpu().numpy():
            x0, y0, x1, y1 = box.tolist()
            ax.add_patch(patches.Rectangle((x0, y0), max(x1 - x0, 1), max(y1 - y0, 1), linewidth=1.0, edgecolor="lime", facecolor="none", alpha=0.45))

    for item in candidate_items:
        x0, y0, x1, y1 = item["box"]
        ax.add_patch(patches.Rectangle((x0, y0), max(x1 - x0, 1), max(y1 - y0, 1), linewidth=0.8, edgecolor="tomato", facecolor="none", alpha=0.25))

    for rank, item in enumerate(decoded_items, 1):
        x0, y0, x1, y1 = item["box"]
        ax.add_patch(patches.Rectangle((x0, y0), max(x1 - x0, 1), max(y1 - y0, 1), linewidth=1.6, edgecolor="dodgerblue", facecolor="none", alpha=0.95))
        label = f"{rank}:{item['char']}"
        ax.text(x0, y0, label, fontsize=8, color="white", fontproperties=font, bbox={"facecolor": "dodgerblue", "alpha": 0.75, "pad": 1, "edgecolor": "none"})

    title = f"GT({len(gt_text)}): {gt_text}\nPred({len(pred_text)}): {pred_text}\nCER={cer:.3f}"
    ax.set_title(title, fontsize=10, fontproperties=font)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def main():
    cli = parse_args()
    args = load_cfg_to_args(cli)
    split = "valid" if cli.split == "val" else cli.split
    device = torch.device(cli.device if torch.cuda.is_available() and "cuda" in cli.device else "cpu")
    output_dir = Path(cli.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(image_set=split, args=args)
    args.charset = dataset.charset
    model, criterion, _ = build_model_main(args)
    model.to(device).eval()
    criterion.eval()

    if cli.new_class_embedding:
        adapt_class_head(model, len(dataset.charset), device)

    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    ckpt_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    skipped = load_compatible_state(model, ckpt_model)

    records = []
    selected = sample_indices_by_length(dataset, cli.samples_per_bin, cli.seed)
    with torch.no_grad():
        for record_idx, (bucket, dataset_idx) in enumerate(selected):
            sample, target = dataset[dataset_idx]
            samples, targets = utils.collate_fn([(sample, target)])
            samples = samples.to(device)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            target = targets[0]
            direction = direction_name(target)

            outputs = model(samples)
            _, pred_probs, _ = criterion.loss_CTC(outputs, targets, None, None, return_preds=True)
            probs = pred_probs[0]
            tokens = probs.argmax(-1).detach().cpu().tolist()
            decoded = collapse_ctc(tokens)

            pred_boxes = outputs["pred_boxes"][0]
            order = sort_boxes_by_direction(pred_boxes, direction)
            sorted_boxes = pred_boxes.index_select(0, order)
            size_hw = target["size"].reshape(-1).detach().cpu().tolist()
            sorted_boxes_xyxy = boxes_to_xyxy_pixels(sorted_boxes, size_hw).detach().cpu()

            gt_labels = [int(x) for x in target["labels"].detach().cpu().tolist()]
            gt_text = "".join(charset_to_str(dataset.charset[x], args.mode_chr) for x in gt_labels)

            pred_labels = []
            decoded_items = []
            for pos, token in decoded:
                class_id = token - 1
                if 0 <= class_id < len(dataset.charset):
                    char = charset_to_str(dataset.charset[class_id], args.mode_chr)
                    pred_labels.append(class_id)
                    decoded_items.append({
                        "pos": pos,
                        "char": char,
                        "box": [float(x) for x in sorted_boxes_xyxy[pos].tolist()],
                    })
            pred_text = "".join(item["char"] for item in decoded_items)
            cer = levenshtein(gt_labels, pred_labels) / max(len(gt_labels), 1)

            nonblank_score = probs[:, 1:].max(-1).values.detach().cpu()
            candidate_count = min(cli.candidate_topk, len(nonblank_score))
            candidate_pos = torch.topk(nonblank_score, k=candidate_count).indices.sort().values.tolist()
            candidate_items = [
                {"pos": pos, "box": [float(x) for x in sorted_boxes_xyxy[pos].tolist()]}
                for pos in candidate_pos
            ]

            gt_boxes = None
            if cli.show_gt_boxes and "boxes" in target and target["boxes"].numel() > 0 and bool(target["boxes"].abs().sum().item() > 0):
                gt_boxes = target_boxes_to_xyxy_pixels(target["boxes"].detach().cpu(), size_hw)

            name = f"{record_idx:03d}_{bucket.replace('+', 'plus')}_idx{dataset_idx:05d}_len{len(gt_labels)}.png"
            save_path = image_dir / name
            render(denormalize_image(samples.tensors[0]), gt_text, pred_text, cer, decoded_items, candidate_items, gt_boxes, save_path)

            records.append({
                "bucket": bucket,
                "dataset_index": dataset_idx,
                "gt_len": len(gt_labels),
                "pred_len": len(pred_labels),
                "cer": cer,
                "gt": gt_text,
                "pred": pred_text,
                "image_file": str(save_path),
                "decoded": decoded_items,
            })

    summary = {
        "checkpoint": cli.checkpoint,
        "config_file": cli.config_file,
        "split": split,
        "samples": len(records),
        "samples_per_bin": cli.samples_per_bin,
        "skipped_checkpoint_keys": len(skipped),
        "records_file": str(output_dir / "records.json"),
    }
    (output_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved visualizations to {image_dir}")


if __name__ == "__main__":
    main()
