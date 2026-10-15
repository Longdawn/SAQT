import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

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
from util.ctc_decoding import decode_greedy, ratio_from_target
from util.ctc_metrics import CtcTotals as Totals, LENGTH_BIN_KEYS, length_bin, levenshtein_ops
from util.slconfig import DictAction


def parse_args():
    parser = argparse.ArgumentParser("Evaluate short-sample CTC rescue decoding")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, default="mth1000")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "valid", "test"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--new_class_embedding", action="store_true")
    parser.add_argument("--ratio_thresholds", nargs="+", type=float, default=[1.5, 1.75, 2.0, 2.5, 3.0])
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_cases", type=str, default=None)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="Override config values, same format as finetuning.py --options.",
    )
    return parser.parse_args()


def require_single_sample_probs(pred_probs, *, context):
    if pred_probs.ndim != 3 or pred_probs.shape[0] != 1:
        raise ValueError(
            f"{context} expects CTC probabilities with shape [1, T, C]; "
            f"got {tuple(pred_probs.shape)}"
        )


def decode_baseline(pred_probs, charset_size):
    require_single_sample_probs(pred_probs, context="evaluate_short_rescue.decode_baseline")
    return decode_greedy(pred_probs, charset_size)


def candidate_nonblank_labels(pred_probs, charset_size, topk):
    require_single_sample_probs(pred_probs, context="evaluate_short_rescue.candidate_nonblank_labels")
    probs = pred_probs[0]
    nonblank = probs[:, 1 : charset_size + 1]
    best_scores, best_labels = nonblank.max(dim=-1)
    order = torch.argsort(best_scores, descending=True).tolist()

    labels = []
    used = set()
    for pos in order:
        label = int(best_labels[pos].item())
        if label in used:
            continue
        labels.append(label)
        used.add(label)
        if len(labels) >= topk:
            break
    return labels


def apply_policy(base_labels, pred_probs, target, charset_size, policy, topk):
    if policy["kind"] == "baseline":
        return base_labels

    ratio = ratio_from_target(target)
    candidates = candidate_nonblank_labels(pred_probs, charset_size, topk)
    if not candidates:
        return base_labels

    if policy["kind"] == "empty_top1":
        return candidates[:1] if len(base_labels) == 0 else base_labels

    if policy["kind"] == "ratio_empty_top1":
        if len(base_labels) == 0 and ratio <= policy["ratio_max"]:
            return candidates[:1]
        return base_labels

    if policy["kind"] == "ratio_min1":
        if len(base_labels) < 1 and ratio <= policy["ratio_max"]:
            return candidates[:1]
        return base_labels

    if policy["kind"] == "ratio_min2":
        if len(base_labels) < 2 and ratio <= policy["ratio_max"]:
            return candidates[:2]
        return base_labels

    return base_labels


def make_policies(ratio_thresholds):
    policies = [{"name": "baseline", "kind": "baseline"}, {"name": "empty_top1", "kind": "empty_top1"}]
    for ratio in ratio_thresholds:
        policies.extend(
            [
                {"name": f"ratio{ratio:g}_empty_top1", "kind": "ratio_empty_top1", "ratio_max": ratio},
                {"name": f"ratio{ratio:g}_min1", "kind": "ratio_min1", "ratio_max": ratio},
                {"name": f"ratio{ratio:g}_min2", "kind": "ratio_min2", "ratio_max": ratio},
            ]
        )
    return policies


def summarize(total, by_len_bin):
    summary = total.to_summary()
    summary["by_gt_len_bin"] = {
        key: by_len_bin[key].to_summary()
        for key in LENGTH_BIN_KEYS
        if by_len_bin[key].n > 0
    }
    return summary


def main():
    cli = parse_args()
    if cli.split == "valid":
        cli.split = "val"
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

    policies = make_policies(cli.ratio_thresholds)
    totals = {policy["name"]: Totals() for policy in policies}
    by_bins = {policy["name"]: defaultdict(Totals) for policy in policies}
    changed_cases = []

    with torch.no_grad():
        for i, (samples, targets) in enumerate(loader):
            if i >= cli.max_samples:
                break
            samples = samples.to(device)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            _, pred_probs, _ = criterion.loss_CTC(outputs, targets, None, None, return_preds=True)
            require_single_sample_probs(pred_probs, context="evaluate_short_rescue")

            gt_labels = [int(x) for x in targets[0]["labels"].tolist()]
            gt_len = len(gt_labels)
            base_labels = decode_baseline(pred_probs, len(dataset.charset))
            gt_bin = length_bin(gt_len)
            ratio = ratio_from_target(targets[0])

            base_dist, _, _, _ = levenshtein_ops(gt_labels, base_labels)
            for policy in policies:
                pred_labels = apply_policy(
                    base_labels,
                    pred_probs,
                    targets[0],
                    len(dataset.charset),
                    policy,
                    cli.topk,
                )
                dist, ins, dels, subs = levenshtein_ops(gt_labels, pred_labels)
                totals[policy["name"]].add(gt_len, len(pred_labels), dist, ins, dels, subs)
                by_bins[policy["name"]][gt_bin].add(gt_len, len(pred_labels), dist, ins, dels, subs)

                if (
                    cli.output_cases
                    and policy["name"] != "baseline"
                    and pred_labels != base_labels
                    and len(changed_cases) < 1000
                ):
                    changed_cases.append(
                        {
                            "idx": i,
                            "policy": policy["name"],
                            "ratio": ratio,
                            "gt_len": gt_len,
                            "base_len": len(base_labels),
                            "pred_len": len(pred_labels),
                            "base_dist": base_dist,
                            "rescued_dist": dist,
                            "gt": "".join(dataset.charset[x] for x in gt_labels),
                            "base": "".join(dataset.charset[x] for x in base_labels),
                            "rescued": "".join(dataset.charset[x] for x in pred_labels),
                        }
                    )

    results = []
    for policy in policies:
        row = summarize(totals[policy["name"]], by_bins[policy["name"]])
        row["policy"] = policy["name"]
        results.append(row)
    results.sort(key=lambda row: row["cer_micro"])

    out_json = Path(cli.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if cli.output_cases:
        out_cases = Path(cli.output_cases)
        out_cases.parent.mkdir(parents=True, exist_ok=True)
        with out_cases.open("w", encoding="utf-8") as f:
            for case in changed_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print("Top rescue policies by micro CER:")
    print("rank policy cer pred/gt empty del sub len1 len2 len3-5 len11+")
    for rank, row in enumerate(results[:12], start=1):
        bins = row["by_gt_len_bin"]
        print(
            rank,
            row["policy"],
            f"{row['cer_micro']:.6f}",
            f"{row['pred_gt_len_ratio']:.6f}",
            f"{row['empty_pred_rate']:.6f}",
            f"{row['del_rate']:.6f}",
            f"{row['sub_rate']:.6f}",
            f"{bins.get('1', {}).get('cer_micro', 0):.6f}",
            f"{bins.get('2', {}).get('cer_micro', 0):.6f}",
            f"{bins.get('3-5', {}).get('cer_micro', 0):.6f}",
            f"{bins.get('11+', {}).get('cer_micro', 0):.6f}",
        )
    print(f"Saved results -> {out_json}")
    if cli.output_cases:
        print(f"Saved changed cases -> {cli.output_cases}")


if __name__ == "__main__":
    main()
