import argparse
import json
from collections import defaultdict
from numbers import Number
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import util.misc as utils
from datasets import build_dataset
from finetuning import build_model_main
from util.ctc_decoding import apply_ctc_calibration, decode_greedy
from util.ctc_metrics import CtcTotals, LENGTH_BIN_KEYS, length_bin, levenshtein_ops
from util.slconfig import DictAction, SLConfig


def parse_args():
    parser = argparse.ArgumentParser("Fast batched CTC micro CER/AR/CR evaluator")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, default="mth_combo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "valid", "test"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--new_class_embedding", action="store_true")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--blank_bias", type=float, default=0.0)
    parser.add_argument("--nonblank_bias", type=float, default=0.0)
    parser.add_argument("--ratio_nonblank_bias", type=float, default=0.0)
    parser.add_argument("--ratio_min", type=float, default=1.5)
    parser.add_argument("--ratio_max", type=float, default=2.0)
    parser.add_argument(
        "--sample_manifest",
        type=str,
        default=None,
        help="Optional JSONL manifest mapping sample id to source_id.",
    )
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
    }
    for key, value in defaults.items():
        if key not in base:
            base[key] = value
    return argparse.Namespace(**base)


def _adapt_class_head(model, charset_size, device):
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


def _load_compatible_state(model, checkpoint_model):
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


def _has_usable_orig_size(target):
    orig_size = target.get("orig_size") if isinstance(target, dict) else None
    if orig_size is None or isinstance(orig_size, (str, bytes)):
        return False
    if torch.is_tensor(orig_size):
        values = orig_size.detach().cpu().reshape(-1)
        return values.numel() >= 2
    try:
        values = list(orig_size)
    except TypeError:
        return False
    return (
        len(values) >= 2
        and isinstance(values[0], Number)
        and isinstance(values[1], Number)
    )


def calibrate_decode_scores(pred_probs, target, cli):
    ratio_nonblank_bias = cli.ratio_nonblank_bias if _has_usable_orig_size(target) else 0.0
    return apply_ctc_calibration(
        pred_probs,
        target=target,
        blank_bias=cli.blank_bias,
        nonblank_bias=cli.nonblank_bias,
        ratio_nonblank_bias=ratio_nonblank_bias,
        ratio_min=cli.ratio_min,
        ratio_max=cli.ratio_max,
    )


def add_rates(summary):
    summary["micro_cer"] = summary["cer_micro"]
    summary["micro_ar"] = 1.0 - summary["cer_micro"]
    summary["micro_cr"] = 1.0 - summary["del_rate"] - summary["sub_rate"]
    return summary


def main():
    cli = parse_args()
    args = load_cfg_to_args(cli)
    if "cuda" in cli.device and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DTLR CTC evaluator")
    device = torch.device(cli.device)

    dataset = build_dataset(image_set=cli.split, args=args)
    dataset_size = len(dataset)
    start_index = max(0, int(cli.start_index))
    end_index = dataset_size if cli.end_index is None else min(dataset_size, int(cli.end_index))
    if start_index > end_index:
        raise ValueError(f"start_index must be <= end_index, got {start_index} > {end_index}")
    eval_dataset = Subset(dataset, range(start_index, end_index))
    args.charset = dataset.charset
    model, criterion, _ = build_model_main(args)

    if cli.new_class_embedding:
        _adapt_class_head(model, len(dataset.charset), device)

    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    ckpt_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    skipped = _load_compatible_state(model, ckpt_model)
    model.to(device)
    model.eval()
    criterion.to(device)
    criterion.eval()
    print(f"Loaded checkpoint with {len(skipped)} skipped keys")

    loader = DataLoader(
        eval_dataset,
        batch_size=cli.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        collate_fn=utils.collate_fn,
    )

    totals = CtcTotals()
    by_len_bin = defaultdict(CtcTotals)
    by_len_exact = defaultdict(CtcTotals)
    by_source = defaultdict(CtcTotals)
    source_by_id = {}
    if cli.sample_manifest is not None:
        with Path(cli.sample_manifest).open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                source_by_id[str(row["id"])] = str(row["source_id"])
    processed = 0

    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(loader):
            if cli.max_samples is not None and processed >= cli.max_samples:
                break
            samples = samples.to(device)
            targets = [
                {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in target.items()}
                for target in targets
            ]
            outputs = model(samples)
            _, pred_probs, _ = criterion.loss_CTC(outputs, targets, None, None, return_preds=True)

            batch_size = pred_probs.shape[0]
            for sample_idx in range(batch_size):
                if cli.max_samples is not None and processed >= cli.max_samples:
                    break
                target = targets[sample_idx]
                decode_scores = calibrate_decode_scores(pred_probs[sample_idx], target, cli)
                pred_labels = decode_greedy(decode_scores, len(dataset.charset))
                gt_labels = [int(x) for x in target["labels"].detach().cpu().tolist()]

                dist, ins, dels, subs = levenshtein_ops(gt_labels, pred_labels)
                gt_len = len(gt_labels)
                pred_len = len(pred_labels)
                totals.add(gt_len, pred_len, dist, ins, dels, subs)
                by_len_bin[length_bin(gt_len)].add(gt_len, pred_len, dist, ins, dels, subs)
                by_len_exact[str(gt_len)].add(gt_len, pred_len, dist, ins, dels, subs)
                if source_by_id:
                    dataset_index = start_index + processed
                    sample_id = str(dataset.samples[dataset_index]["id"])
                    if sample_id not in source_by_id:
                        raise KeyError(f"Sample {sample_id!r} is missing from {cli.sample_manifest}")
                    by_source[source_by_id[sample_id]].add(
                        gt_len, pred_len, dist, ins, dels, subs
                    )
                processed += 1

            if batch_idx % 50 == 0:
                summary = totals.to_summary()
                print(
                    f"processed {processed}/{len(dataset)} "
                    f"CER={summary['cer_micro']:.6f} "
                    f"AR={1.0 - summary['cer_micro']:.6f} "
                    f"CR={1.0 - summary['del_rate'] - summary['sub_rate']:.6f}",
                    flush=True,
                )

    summary = add_rates(totals.to_summary())
    summary.update(
        {
            "dataset": cli.dataset_file,
            "split": cli.split,
            "checkpoint": cli.checkpoint,
            "config": cli.config_file,
            "dataset_size": dataset_size,
            "start_index": start_index,
            "end_index": end_index,
            "evaluated_samples": processed,
            "total_chars": totals.gt_chars,
            "insertions": totals.ins,
            "deletions": totals.dels,
            "substitutions": totals.subs,
            "blank_bias": cli.blank_bias,
            "nonblank_bias": cli.nonblank_bias,
            "ratio_nonblank_bias": cli.ratio_nonblank_bias,
            "skipped_keys": len(skipped),
            "by_gt_len_bin": {
                key: add_rates(by_len_bin[key].to_summary())
                for key in LENGTH_BIN_KEYS
                if by_len_bin[key].n > 0
            },
            "by_gt_len_exact": {
                key: add_rates(value.to_summary())
                for key, value in sorted(by_len_exact.items(), key=lambda item: int(item[0]))
            },
            "by_source": {
                key: add_rates(value.to_summary())
                for key, value in sorted(by_source.items())
            },
        }
    )

    out_json = Path(cli.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary -> {out_json}")


if __name__ == "__main__":
    main()
