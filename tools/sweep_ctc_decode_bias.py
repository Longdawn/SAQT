import argparse
import json
from collections import defaultdict
from numbers import Number
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_dataset
from finetuning import build_model_main
from tools.analyze_ctc_errors import (
    _adapt_class_head,
    _load_compatible_state,
    load_cfg_to_args,
)
from util.ctc_decoding import apply_ctc_calibration, decode_greedy
from util.ctc_metrics import CtcTotals as Totals, LENGTH_BIN_KEYS, length_bin, levenshtein_ops
from util.slconfig import DictAction


def parse_args():
    parser = argparse.ArgumentParser("Sweep CTC decode-time blank/nonblank biases")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, default="mth1000")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--new_class_embedding", action="store_true")
    parser.add_argument("--blank_biases", nargs="+", type=float, default=[0.0])
    parser.add_argument(
        "--nonblank_biases",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    )
    parser.add_argument(
        "--ratio_nonblank_biases",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.15, 0.2],
    )
    parser.add_argument(
        "--margin_gate_mins",
        nargs="+",
        type=float,
        default=None,
        help="Only apply blank/nonblank bias when blank-top_nonblank margin is >= this value.",
    )
    parser.add_argument(
        "--margin_gate_maxs",
        nargs="+",
        type=float,
        default=None,
        help="Only apply blank/nonblank bias when blank-top_nonblank margin is <= this value.",
    )
    parser.add_argument("--ratio_min", type=float, default=1.5)
    parser.add_argument("--ratio_max", type=float, default=2.0)
    parser.add_argument(
        "--adaptive_modes",
        nargs="+",
        default=["none"],
        choices=["none", "empty", "pred_short"],
        help=(
            "Adaptive calibration modes. 'none' keeps fixed bias behavior. "
            "'empty' applies adaptive_max_scale only when clean greedy decoding is empty, "
            "otherwise adaptive_min_scale. 'pred_short' applies adaptive_max_scale when "
            "clean collapsed CTC length is <= adaptive_short_pred_max_len."
        ),
    )
    parser.add_argument(
        "--adaptive_min_scales",
        nargs="+",
        type=float,
        default=[1.0],
        help="Bias scale for samples that do not trigger the adaptive condition.",
    )
    parser.add_argument(
        "--adaptive_max_scales",
        nargs="+",
        type=float,
        default=[1.0],
        help="Bias scale for samples that trigger the adaptive condition.",
    )
    parser.add_argument(
        "--adaptive_short_pred_max_lens",
        nargs="+",
        type=int,
        default=[2],
        help="Clean collapsed CTC length thresholds for adaptive_mode=pred_short.",
    )
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="Override config values, same format as finetuning.py --options.",
    )
    return parser.parse_args()


def build_settings(cli):
    if (cli.margin_gate_mins is None) != (cli.margin_gate_maxs is None):
        raise ValueError("margin-gated decode sweep requires both --margin_gate_mins and --margin_gate_maxs")

    margin_gate_mins = cli.margin_gate_mins if cli.margin_gate_mins is not None else [None]
    margin_gate_maxs = cli.margin_gate_maxs if cli.margin_gate_maxs is not None else [None]
    adaptive_modes = getattr(cli, "adaptive_modes", ["none"])
    adaptive_min_scales = getattr(cli, "adaptive_min_scales", [1.0])
    adaptive_max_scales = getattr(cli, "adaptive_max_scales", [1.0])
    adaptive_short_pred_max_lens = getattr(cli, "adaptive_short_pred_max_lens", [2])
    settings = []
    for blank_bias in cli.blank_biases:
        for nonblank_bias in cli.nonblank_biases:
            for ratio_nonblank_bias in cli.ratio_nonblank_biases:
                for margin_gate_min in margin_gate_mins:
                    for margin_gate_max in margin_gate_maxs:
                        if margin_gate_min is not None and margin_gate_max is not None:
                            if float(margin_gate_min) > float(margin_gate_max):
                                raise ValueError("margin_gate_min must be <= margin_gate_max")
                        for adaptive_mode in adaptive_modes:
                            for adaptive_min_scale in adaptive_min_scales:
                                for adaptive_max_scale in adaptive_max_scales:
                                    for adaptive_short_pred_max_len in adaptive_short_pred_max_lens:
                                        settings.append(
                                            {
                                                "blank_bias": float(blank_bias),
                                                "nonblank_bias": float(nonblank_bias),
                                                "ratio_nonblank_bias": float(ratio_nonblank_bias),
                                                "margin_gate_min": (
                                                    None
                                                    if margin_gate_min is None
                                                    else float(margin_gate_min)
                                                ),
                                                "margin_gate_max": (
                                                    None
                                                    if margin_gate_max is None
                                                    else float(margin_gate_max)
                                                ),
                                                "adaptive_mode": str(adaptive_mode),
                                                "adaptive_min_scale": float(adaptive_min_scale),
                                                "adaptive_max_scale": float(adaptive_max_scale),
                                                "adaptive_short_pred_max_len": int(
                                                    adaptive_short_pred_max_len
                                                ),
                                            }
                                        )
    return settings


def require_single_sample_probs(pred_probs, *, context):
    if pred_probs.ndim != 3 or pred_probs.shape[0] != 1:
        raise ValueError(
            f"{context} expects CTC probabilities with shape [1, T, C]; "
            f"got {tuple(pred_probs.shape)}"
        )


def validate_ratio_bias_target(target):
    orig_size = target.get("orig_size") if isinstance(target, dict) else None
    if orig_size is None:
        raise ValueError(
            "ratio-conditioned CTC bias requires target['orig_size'] with at least "
            "height and width values"
        )
    if torch.is_tensor(orig_size):
        values = orig_size.detach().cpu().reshape(-1)
        if values.numel() < 2:
            raise ValueError(
                "ratio-conditioned CTC bias requires target['orig_size'] with at least "
                "height and width values"
            )
        values = values.tolist()
    else:
        if isinstance(orig_size, (str, bytes)):
            raise ValueError(
                "ratio-conditioned CTC bias requires target['orig_size'] to be a "
                "non-string sequence of numeric height and width values"
            )
        try:
            values = list(orig_size)
        except TypeError:
            raise ValueError(
                "ratio-conditioned CTC bias requires target['orig_size'] to be a "
                "non-string sequence of numeric height and width values"
            )
        if len(values) < 2:
            raise ValueError(
                "ratio-conditioned CTC bias requires target['orig_size'] with at least "
                "height and width values"
            )
        if not isinstance(values[0], Number) or not isinstance(values[1], Number):
            raise ValueError(
                "ratio-conditioned CTC bias requires numeric height/width in "
                "target['orig_size']"
            )


def make_summary(total, by_len_bin):
    summary = total.to_summary()
    summary["by_gt_len_bin"] = {
        key: by_len_bin[key].to_summary()
        for key in LENGTH_BIN_KEYS
        if by_len_bin[key].n > 0
    }
    return summary


def main():
    cli = parse_args()
    args = load_cfg_to_args(cli)
    device = torch.device(cli.device if torch.cuda.is_available() and "cuda" in cli.device else "cpu")

    dataset = build_dataset(image_set=cli.split, args=args)
    args.charset = dataset.charset
    model, criterion, _ = build_model_main(args)
    model.to(device)
    model.eval()
    criterion.eval()

    if cli.new_class_embedding:
        _adapt_class_head(model, len(dataset.charset), device)

    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    ckpt_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    skipped = _load_compatible_state(model, ckpt_model)
    print(f"Loaded checkpoint with {len(skipped)} skipped keys")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cli.num_workers,
        collate_fn=utils.collate_fn,
    )

    settings = build_settings(cli)
    totals = [Totals() for _ in settings]
    by_len_bins = [defaultdict(Totals) for _ in settings]

    with torch.no_grad():
        for i, (samples, targets) in enumerate(loader):
            if i >= cli.max_samples:
                break
            samples = samples.to(device)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            _, pred_probs, _ = criterion.loss_CTC(outputs, targets, None, None, return_preds=True)
            require_single_sample_probs(pred_probs, context="sweep_ctc_decode_bias")

            gt_labels = [int(x) for x in targets[0]["labels"].tolist()]
            gt_len = len(gt_labels)
            gt_bin = length_bin(gt_len)

            for idx, setting in enumerate(settings):
                if setting["ratio_nonblank_bias"] != 0.0:
                    validate_ratio_bias_target(targets[0])
                calibrated_scores = apply_ctc_calibration(
                    pred_probs,
                    target=targets[0],
                    blank_bias=setting["blank_bias"],
                    nonblank_bias=setting["nonblank_bias"],
                    ratio_nonblank_bias=setting["ratio_nonblank_bias"],
                    ratio_min=cli.ratio_min,
                    ratio_max=cli.ratio_max,
                    margin_gate_min=setting["margin_gate_min"],
                    margin_gate_max=setting["margin_gate_max"],
                    adaptive_mode=setting["adaptive_mode"],
                    adaptive_min_scale=setting["adaptive_min_scale"],
                    adaptive_max_scale=setting["adaptive_max_scale"],
                    adaptive_short_pred_max_len=setting["adaptive_short_pred_max_len"],
                )
                pred_labels = decode_greedy(calibrated_scores, len(dataset.charset))
                pred_len = len(pred_labels)
                dist, ins, dels, subs = levenshtein_ops(gt_labels, pred_labels)
                totals[idx].add(gt_len, pred_len, dist, ins, dels, subs)
                by_len_bins[idx][gt_bin].add(gt_len, pred_len, dist, ins, dels, subs)

    results = []
    for setting, total, by_len_bin in zip(settings, totals, by_len_bins):
        summary = make_summary(total, by_len_bin)
        results.append({**setting, **summary})

    results.sort(key=lambda row: row["cer_micro"])
    out_path = Path(cli.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Top decode-bias settings by CER:")
    print(
        "rank blank nonblank ratio_bias gate_min gate_max adaptive min_scale max_scale short_len "
        "cer pred/gt empty del sub len1 len2 len11+"
    )
    for rank, row in enumerate(results[:20], start=1):
        bins = row["by_gt_len_bin"]
        print(
            rank,
            f"{row['blank_bias']:.3f}",
            f"{row['nonblank_bias']:.3f}",
            f"{row['ratio_nonblank_bias']:.3f}",
            "none" if row["margin_gate_min"] is None else f"{row['margin_gate_min']:.3f}",
            "none" if row["margin_gate_max"] is None else f"{row['margin_gate_max']:.3f}",
            row["adaptive_mode"],
            f"{row['adaptive_min_scale']:.3f}",
            f"{row['adaptive_max_scale']:.3f}",
            row["adaptive_short_pred_max_len"],
            f"{row['cer_micro']:.6f}",
            f"{row['pred_gt_len_ratio']:.6f}",
            f"{row['empty_pred_rate']:.6f}",
            f"{row['del_rate']:.6f}",
            f"{row['sub_rate']:.6f}",
            f"{bins.get('1', {}).get('cer_micro', 0):.6f}",
            f"{bins.get('2', {}).get('cer_micro', 0):.6f}",
            f"{bins.get('11+', {}).get('cer_micro', 0):.6f}",
        )
    print(f"Saved sweep results -> {out_path}")


if __name__ == "__main__":
    main()
