import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataset
from finetuning import build_model_main
from util.slconfig import SLConfig

_saved_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
import evaluation as eval_mod
sys.argv = _saved_argv


def parse_args():
    parser = argparse.ArgumentParser("Compute micro AR/CR for chinese evaluation runs")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--new_class_embedding", action="store_true")
    parser.add_argument("--new_label_enc", action="store_true")
    parser.add_argument("--fix_enc_out_class", action="store_true")
    parser.add_argument("--unicode", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def build_args(cli):
    args_dataset = SLConfig.fromfile(cli.config)
    args_dataset.dataset_file = cli.dataset
    args_dataset.device = cli.device
    args_dataset.coco_path = "/comp_robot/cv_public_dataset/COCO2017/"
    args_dataset.fix_size = False
    return args_dataset


def load_model(model, dataset, args_dataset, cli):
    device = args_dataset.device

    def _load_compatible_weights(model_obj, ckpt_model):
        model_state = model_obj.state_dict()
        filtered = {}
        skipped = []
        for k, v in ckpt_model.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        model_obj.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"Skipped {len(skipped)} incompatible params")

    if cli.new_class_embedding:
        features_dim = model.class_embed[0].weight.data.shape[1]
        new_charset_size = len(args_dataset.charset)
        new_class_embed = nn.Linear(features_dim, new_charset_size)
        new_decoder_class_embed = nn.Linear(features_dim, new_charset_size)
        new_enc_out_class_embed = nn.Linear(features_dim, new_charset_size)
        if model.dec_pred_class_embed_share:
            class_embed_layerlist = [
                new_class_embed for _ in range(model.transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                nn.Linear(features_dim, new_charset_size)
                for _ in range(model.transformer.num_decoder_layers)
            ]
        model.class_embed = nn.ModuleList(class_embed_layerlist).to(device)
        model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
        if not cli.fix_enc_out_class:
            model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
        if cli.new_label_enc:
            model.label_enc = nn.Embedding(len(dataset.charset) + 1, features_dim).to(device)

    checkpoint = torch.load(cli.weights, map_location="cpu", weights_only=False)
    ckpt_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    _load_compatible_weights(model, ckpt_model)
    model.eval()
    model.to(device)


def main():
    cli = parse_args()
    args_dataset = build_args(cli)

    # Make evaluation helpers use the same runtime options.
    eval_mod.args = argparse.Namespace(
        dataset=cli.dataset,
        mode=cli.mode,
        new_class_embedding=cli.new_class_embedding,
        new_label_enc=cli.new_label_enc,
        NMS_inference=False,
        metrics="chinese",
        unicode=cli.unicode,
        weights=cli.weights,
        config=cli.config,
        fix_enc_out_class=cli.fix_enc_out_class,
        TH=None,
        NMS=None,
        text_direction="auto",
        decode_by_pred_direction=False,
    )
    eval_mod.args_dataset = args_dataset

    dataset_val = build_dataset(image_set=cli.mode, args=args_dataset)
    args_dataset.charset = dataset_val.charset
    eval_mod.dataset_val = dataset_val

    model, _, postprocessors = build_model_main(args_dataset)
    eval_mod.postprocessors = postprocessors
    load_model(model, dataset_val, args_dataset, cli)

    macro_ar = []
    macro_cr = []
    macro_cer = []
    total_chars = 0
    total_ins = 0
    total_del = 0
    total_sub = 0

    with torch.no_grad():
        for i in range(len(dataset_val)):
            image, targets = dataset_val[i]
            outputs = model.cuda()(image[None].cuda())
            cer_it, _, _, predicted_labels = eval_mod.compute_cer_impact(
                outputs, [targets], dataset_val.charset, {}, TH=None, NM=None
            )
            gt_labels = [int(item) for item in targets["labels"]]
            ins, dels, subs = eval_mod.compute_edit_operations(gt_labels, predicted_labels)
            gt_len = len(gt_labels)

            macro_cer.append(cer_it)
            macro_ar.append(1 - cer_it)
            macro_cr.append(eval_mod.compute_CR(gt_labels, predicted_labels))

            total_chars += gt_len
            total_ins += ins
            total_del += dels
            total_sub += subs

            if i % 500 == 0 or i == len(dataset_val) - 1:
                print(
                    f"\rprocessed {i+1}/{len(dataset_val)} "
                    f"macro_AR={np.mean(macro_ar):.6f} macro_CR={np.mean(macro_cr):.6f}",
                    end="",
                )
    print()

    total_err = total_ins + total_del + total_sub
    micro_cer = total_err / max(total_chars, 1)
    micro_ar = 1.0 - micro_cer
    micro_cr = 1.0 - (total_del + total_sub) / max(total_chars, 1)

    summary = {
        "dataset": cli.dataset,
        "mode": cli.mode,
        "weights": cli.weights,
        "config": cli.config,
        "samples": len(dataset_val),
        "macro_ar": float(np.mean(macro_ar)),
        "macro_cr": float(np.mean(macro_cr)),
        "macro_cer": float(np.mean(macro_cer)),
        "micro_ar": float(micro_ar),
        "micro_cr": float(micro_cr),
        "micro_cer": float(micro_cer),
        "total_chars": int(total_chars),
        "insertions": int(total_ins),
        "deletions": int(total_del),
        "substitutions": int(total_sub),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if cli.output_json:
        out = Path(cli.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
