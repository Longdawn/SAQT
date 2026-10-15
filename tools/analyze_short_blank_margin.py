import argparse
import json
import sys
from collections import Counter, defaultdict
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
from util.ctc_decoding import decode_greedy
from util.ctc_metrics import levenshtein_ops
from util.slconfig import DictAction


def parse_args():
    parser = argparse.ArgumentParser("Diagnose blank/nonblank margins on short CTC samples")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, default="mth1000")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "valid", "test"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--max_gt_len", type=int, default=2)
    parser.add_argument("--topk_queries", type=int, default=8)
    parser.add_argument("--new_class_embedding", action="store_true")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_cases", type=str, required=True)
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


def label_text(labels, charset):
    return "".join(charset[int(x)] for x in labels)


def short_margin_stats(pred_probs, gt_labels, charset_size, topk_queries):
    require_single_sample_probs(pred_probs, context="analyze_short_blank_margin.short_margin_stats")
    probs = pred_probs[0]
    blank_probs = probs[:, 0]
    nonblank_probs = probs[:, 1 : charset_size + 1]
    best_nonblank_probs, best_nonblank_labels = nonblank_probs.max(dim=-1)
    top_query_idx = torch.argsort(best_nonblank_probs, descending=True)[:topk_queries]

    top_rows = []
    top_labels = []
    for q in top_query_idx.tolist():
        label = int(best_nonblank_labels[q].item())
        row = {
            "query": int(q),
            "blank": float(blank_probs[q].item()),
            "best_nonblank": float(best_nonblank_probs[q].item()),
            "margin_blank_minus_nonblank": float((blank_probs[q] - best_nonblank_probs[q]).item()),
            "best_label": label,
        }
        top_rows.append(row)
        top_labels.append(label)

    gt_in_top_labels = [int(label) in set(top_labels) for label in gt_labels]
    gt_best = []
    for label in gt_labels:
        label_probs = nonblank_probs[:, int(label)]
        best_score, best_query = label_probs.max(dim=0)
        gt_best.append(
            {
                "label": int(label),
                "query": int(best_query.item()),
                "gt_nonblank_score": float(best_score.item()),
                "blank_at_gt_best_query": float(blank_probs[best_query].item()),
                "margin_blank_minus_gt": float((blank_probs[best_query] - best_score).item()),
            }
        )

    top1 = top_rows[0] if top_rows else {}
    return {
        "top1_blank": top1.get("blank", 0.0),
        "top1_nonblank": top1.get("best_nonblank", 0.0),
        "top1_margin_blank_minus_nonblank": top1.get("margin_blank_minus_nonblank", 0.0),
        "top1_label": top1.get("best_label", -1),
        "gt_any_in_topk": any(gt_in_top_labels),
        "gt_all_in_topk": all(gt_in_top_labels) if gt_in_top_labels else False,
        "gt_best_mean_margin_blank_minus_gt": (
            sum(row["margin_blank_minus_gt"] for row in gt_best) / max(len(gt_best), 1)
        ),
        "gt_best": gt_best,
        "top_queries": top_rows,
    }


def summarize_cases(cases):
    summary = {
        "samples": len(cases),
        "empty_pred_rate": sum(1 for c in cases if c["pred_len"] == 0) / max(len(cases), 1),
        "error_rate": sum(1 for c in cases if c["dist"] > 0) / max(len(cases), 1),
        "avg_cer": sum(c["cer"] for c in cases) / max(len(cases), 1),
        "avg_top1_margin_blank_minus_nonblank": (
            sum(c["top1_margin_blank_minus_nonblank"] for c in cases) / max(len(cases), 1)
        ),
        "blank_beats_top_nonblank_rate": (
            sum(1 for c in cases if c["top1_margin_blank_minus_nonblank"] > 0) / max(len(cases), 1)
        ),
        "gt_any_in_topk_rate": sum(1 for c in cases if c["gt_any_in_topk"]) / max(len(cases), 1),
        "gt_all_in_topk_rate": sum(1 for c in cases if c["gt_all_in_topk"]) / max(len(cases), 1),
        "avg_gt_best_margin_blank_minus_gt": (
            sum(c["gt_best_mean_margin_blank_minus_gt"] for c in cases) / max(len(cases), 1)
        ),
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

    cases = []
    with torch.no_grad():
        for i, (samples, targets) in enumerate(loader):
            if i >= cli.max_samples:
                break
            gt_len = int(len(targets[0]["labels"]))
            if gt_len > cli.max_gt_len:
                continue

            samples = samples.to(device)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            _, pred_probs, _ = criterion.loss_CTC(outputs, targets, None, None, return_preds=True)
            require_single_sample_probs(pred_probs, context="analyze_short_blank_margin")

            gt_labels = [int(x) for x in targets[0]["labels"].tolist()]
            pred_labels = decode_greedy(pred_probs, len(dataset.charset))
            dist, ins, dels, subs = levenshtein_ops(gt_labels, pred_labels)
            margin = short_margin_stats(pred_probs, gt_labels, len(dataset.charset), cli.topk_queries)

            cases.append(
                {
                    "idx": i,
                    "gt_len": gt_len,
                    "pred_len": len(pred_labels),
                    "dist": int(dist),
                    "cer": float(dist / max(gt_len, 1)),
                    "ins": int(ins),
                    "del": int(dels),
                    "sub": int(subs),
                    "gt": label_text(gt_labels, dataset.charset),
                    "pred": label_text(pred_labels, dataset.charset),
                    **margin,
                }
            )

    by_len = defaultdict(list)
    for case in cases:
        by_len[str(case["gt_len"])].append(case)

    error_cases = [case for case in cases if case["dist"] > 0]
    empty_cases = [case for case in cases if case["pred_len"] == 0]
    nonempty_error_cases = [case for case in cases if case["dist"] > 0 and case["pred_len"] > 0]

    summary = {
        "checkpoint": cli.checkpoint,
        "split": cli.split,
        "max_gt_len": cli.max_gt_len,
        "topk_queries": cli.topk_queries,
        "skipped_keys": len(skipped),
        "all_short": summarize_cases(cases),
        "error_cases": summarize_cases(error_cases),
        "empty_cases": summarize_cases(empty_cases),
        "nonempty_error_cases": summarize_cases(nonempty_error_cases),
        "by_gt_len": {key: summarize_cases(value) for key, value in sorted(by_len.items(), key=lambda x: int(x[0]))},
        "error_type_counts": Counter(
            "empty" if c["pred_len"] == 0 else "nonempty_error" if c["dist"] > 0 else "correct"
            for c in cases
        ),
    }

    out_json = Path(cli.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cases_sorted = sorted(
        cases,
        key=lambda c: (
            c["dist"] == 0,
            c["pred_len"] != 0,
            -c["top1_margin_blank_minus_nonblank"],
            c["idx"],
        ),
    )
    out_cases = Path(cli.output_cases)
    out_cases.parent.mkdir(parents=True, exist_ok=True)
    with out_cases.open("w", encoding="utf-8") as f:
        for case in cases_sorted:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary -> {out_json}")
    print(f"Saved cases -> {out_cases}")


if __name__ == "__main__":
    main()
