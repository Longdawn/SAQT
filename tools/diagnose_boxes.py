import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_dataset
from finetuning import build_model_main
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from util.slconfig import DictAction, SLConfig


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\) DejaVu Sans.")


def parse_args():
    parser = argparse.ArgumentParser("Diagnose DTLR predicted boxes on MTH1000-style data")
    parser.add_argument("--config_file", "-c", required=True, type=str)
    parser.add_argument("--dataset_file", default="mth1000", type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "val", "test"])
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--num_samples", default=50, type=int)
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument("--viz_topk", default=40, type=int)
    parser.add_argument("--new_class_embedding", action="store_true")
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--show_gt_boxes", action="store_true")
    parser.add_argument("--force_pytorch_msda", action="store_true")
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="Override config values, same format as finetuning.py --options.",
    )
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


def resolve_checkpoint(path_str):
    path = Path(path_str)
    if path.is_file():
        return path
    candidates = [
        path / "checkpoint_best_regular.pth",
        path / "checkpoint_best.pth",
        path / "checkpoint.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve checkpoint from {path_str}")


def make_output_dir(cli, checkpoint_path):
    if cli.output_dir:
        return Path(cli.output_dir)
    stamp = datetime.now().strftime("%m%d-%H%M")
    return Path("debug_vis") / f"{checkpoint_path.stem}_boxdiag_{stamp}"


def adapt_class_head(model, charset_size, device):
    features_dim = model.class_embed[0].weight.data.shape[1]
    new_class_embed = nn.Linear(features_dim, charset_size)
    new_decoder_class_embed = nn.Linear(features_dim, charset_size)
    new_enc_out_class_embed = nn.Linear(features_dim, charset_size)

    if model.dec_pred_class_embed_share:
        class_embed_layerlist = [new_class_embed for _ in range(model.transformer.num_decoder_layers)]
    else:
        class_embed_layerlist = [
            nn.Linear(features_dim, charset_size) for _ in range(model.transformer.num_decoder_layers)
        ]

    model.class_embed = nn.ModuleList(class_embed_layerlist).to(device)
    model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
    model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
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


def direction_name(target):
    direction = target.get("direction", 0)
    if torch.is_tensor(direction):
        direction = int(direction.reshape(-1)[0].item())
    else:
        direction = int(direction)
    return "vertical" if direction == 1 else "horizontal"


def sort_indices_by_direction(pred_boxes, direction):
    axis = 1 if direction == "vertical" else 0
    return torch.sort(pred_boxes[:, axis], descending=False)[1]


def charset_to_str(charset_item, mode_chr=True):
    if isinstance(charset_item, str):
        return charset_item
    if mode_chr:
        return chr(int(charset_item))
    return str(charset_item)


def decode_top1_strings(sorted_probs, charset, mode_chr):
    token_ids = sorted_probs.argmax(-1)
    blank_probs = sorted_probs[:, 0]
    nonblank_scores, nonblank_ids = sorted_probs[:, 1:].max(-1)
    text = []
    for class_id in nonblank_ids.tolist():
        if 0 <= class_id < len(charset):
            text.append(charset_to_str(charset[class_id], mode_chr))
        else:
            text.append(f"<oov:{class_id}>")
    return token_ids, blank_probs, nonblank_scores, nonblank_ids, text


def denormalize_image(tensor):
    image = tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def tensor_boxes_to_xyxy_pixels(boxes_cxcywh, size_hw):
    h, w = int(size_hw[0]), int(size_hw[1])
    scale = torch.tensor([w, h, w, h], dtype=boxes_cxcywh.dtype, device=boxes_cxcywh.device)
    return box_cxcywh_to_xyxy(boxes_cxcywh) * scale


def target_boxes_to_xyxy_pixels(target_boxes, size_hw):
    h, w = int(size_hw[0]), int(size_hw[1])
    if target_boxes.numel() == 0:
        return target_boxes
    max_val = float(target_boxes.max().item())
    min_val = float(target_boxes.min().item())
    if min_val >= 0.0 and max_val <= 1.5:
        cxcywh = target_boxes
    else:
        cxcywh = box_xyxy_to_cxcywh(target_boxes)
        cxcywh = cxcywh / torch.tensor([w, h, w, h], dtype=target_boxes.dtype, device=target_boxes.device)
    return tensor_boxes_to_xyxy_pixels(cxcywh, size_hw)


def has_real_gt_boxes(target_boxes):
    if target_boxes.numel() == 0:
        return False
    return bool(target_boxes.abs().sum().item() > 0)


def render_sample(
    image_np,
    pred_boxes_xyxy,
    pred_order_ids,
    pred_chars,
    blank_probs,
    nonblank_scores,
    gt_text,
    pred_nonblank_count,
    save_path,
    gt_boxes_xyxy=None,
):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_np)
    ax.axis("off")

    if gt_boxes_xyxy is not None and gt_boxes_xyxy.numel() > 0:
        for box in gt_boxes_xyxy.cpu().numpy():
            x0, y0, x1, y1 = box.tolist()
            rect = patches.Rectangle(
                (x0, y0),
                max(x1 - x0, 1.0),
                max(y1 - y0, 1.0),
                linewidth=1.0,
                edgecolor="lime",
                facecolor="none",
                alpha=0.45,
            )
            ax.add_patch(rect)

    for i in range(pred_boxes_xyxy.shape[0]):
        x0, y0, x1, y1 = pred_boxes_xyxy[i].cpu().numpy().tolist()
        rect = patches.Rectangle(
            (x0, y0),
            max(x1 - x0, 1.0),
            max(y1 - y0, 1.0),
            linewidth=1.2,
            edgecolor="red",
            facecolor="none",
            alpha=0.85,
        )
        ax.add_patch(rect)
        label = f"{int(pred_order_ids[i])}:{pred_chars[i]} b={blank_probs[i]:.2f} nb={nonblank_scores[i]:.2f}"
        ax.text(
            x0,
            y0,
            label,
            fontsize=7,
            color="white",
            bbox={"facecolor": "red", "alpha": 0.55, "pad": 1},
        )

    title = f"GT len={len(gt_text)} pred_nonblank={pred_nonblank_count} GT={gt_text[:60]}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    cli = parse_args()
    if cli.force_pytorch_msda:
        os.environ["DTLR_MSDA_FORCE_PYTORCH"] = "1"
    args = load_cfg_to_args(cli)
    split = "valid" if cli.split == "val" else cli.split
    device = torch.device(cli.device if torch.cuda.is_available() and "cuda" in cli.device else "cpu")
    checkpoint_path = resolve_checkpoint(cli.checkpoint)
    output_dir = make_output_dir(cli, checkpoint_path)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(image_set=split, args=args)
    args.charset = dataset.charset
    model, criterion, _ = build_model_main(args)
    model.to(device)
    model.eval()
    criterion.eval()

    if cli.new_class_embedding:
        adapt_class_head(model, len(dataset.charset), device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    skipped = load_compatible_state(model, checkpoint_model)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cli.num_workers,
        collate_fn=utils.collate_fn,
    )

    records = []

    with torch.no_grad():
        for sample_idx, (samples, targets) in enumerate(loader):
            if sample_idx < cli.start_index:
                continue
            if len(records) >= cli.num_samples:
                break

            samples = samples.to(device)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            target = targets[0]
            direction = direction_name(target)

            outputs = model(samples)
            _, new_pred_logits, _ = criterion.loss_CTC(outputs, targets, None, None, return_preds=True)
            sorted_probs = new_pred_logits[0]

            pred_boxes = outputs["pred_boxes"][0]
            order_idx = sort_indices_by_direction(pred_boxes, direction)
            sorted_boxes = pred_boxes.index_select(0, order_idx)
            token_ids, blank_probs, nonblank_scores, nonblank_ids, pred_chars = decode_top1_strings(
                sorted_probs, dataset.charset, args.mode_chr
            )

            keep_topk = min(cli.viz_topk, sorted_boxes.shape[0])
            keep_idx = torch.topk(nonblank_scores, k=keep_topk, largest=True).indices
            keep_idx = torch.sort(keep_idx).values

            vis_boxes = sorted_boxes.index_select(0, keep_idx)
            vis_blank = blank_probs.index_select(0, keep_idx).cpu().tolist()
            vis_nonblank = nonblank_scores.index_select(0, keep_idx).cpu().tolist()
            vis_chars = [pred_chars[i] for i in keep_idx.tolist()]
            vis_order_ids = keep_idx.cpu() + 1

            size_hw = target["size"].reshape(-1).tolist()
            vis_boxes_xyxy = tensor_boxes_to_xyxy_pixels(vis_boxes, size_hw)
            gt_boxes_xyxy = None
            gt_real = has_real_gt_boxes(target["boxes"])
            if cli.show_gt_boxes and gt_real:
                gt_boxes_xyxy = target_boxes_to_xyxy_pixels(target["boxes"], size_hw)

            gt_labels = [int(x) for x in target["labels"].detach().cpu().tolist()]
            gt_text = "".join(charset_to_str(dataset.charset[x], args.mode_chr) for x in gt_labels)
            pred_nonblank_count = int((token_ids != 0).sum().item())

            image_np = denormalize_image(samples.tensors[0])
            sample_name = f"{len(records):03d}_dataset_{sample_idx:05d}"
            image_path = image_dir / f"{sample_name}.png"
            render_sample(
                image_np=image_np,
                pred_boxes_xyxy=vis_boxes_xyxy,
                pred_order_ids=vis_order_ids,
                pred_chars=vis_chars,
                blank_probs=vis_blank,
                nonblank_scores=vis_nonblank,
                gt_text=gt_text,
                pred_nonblank_count=pred_nonblank_count,
                save_path=image_path,
                gt_boxes_xyxy=gt_boxes_xyxy,
            )

            record = {
                "record_id": sample_name,
                "dataset_index": sample_idx,
                "direction": direction,
                "gt_text": gt_text,
                "gt_len": len(gt_text),
                "pred_nonblank_count": pred_nonblank_count,
                "blank_ratio_full": float((token_ids == 0).float().mean().item()),
                "avg_blank_prob_full": float(blank_probs.mean().item()),
                "avg_nonblank_score_full": float(nonblank_scores.mean().item()),
                "used_real_gt_boxes": gt_real,
                "image_file": str(image_path),
                "topk": [
                    {
                        "read_order": int(vis_order_ids[i].item()),
                        "char": vis_chars[i],
                        "blank_prob": float(vis_blank[i]),
                        "nonblank_score": float(vis_nonblank[i]),
                        "box_xyxy": [float(x) for x in vis_boxes_xyxy[i].cpu().tolist()],
                    }
                    for i in range(len(vis_chars))
                ],
            }
            records.append(record)

    summary = {
        "checkpoint": str(checkpoint_path),
        "config_file": cli.config_file,
        "dataset_file": cli.dataset_file,
        "split": split,
        "device": str(device),
        "num_records": len(records),
        "viz_topk": cli.viz_topk,
        "skipped_checkpoint_keys": len(skipped),
        "avg_blank_ratio_full": float(np.mean([r["blank_ratio_full"] for r in records])) if records else 0.0,
        "avg_pred_nonblank_count": float(np.mean([r["pred_nonblank_count"] for r in records])) if records else 0.0,
        "avg_gt_len": float(np.mean([r["gt_len"] for r in records])) if records else 0.0,
        "real_gt_box_rate": float(np.mean([1.0 if r["used_real_gt_boxes"] else 0.0 for r in records])) if records else 0.0,
        "records_file": str(output_dir / "records.json"),
    }

    (output_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved visualizations to {image_dir}")


if __name__ == "__main__":
    main()
