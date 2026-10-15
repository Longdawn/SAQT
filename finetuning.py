# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
TENSORBOARD = False

import argparse
import ast
import datetime
import json
import math
import random
import time
from pathlib import Path
import os
import sys
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils
import torch.nn as nn
import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate_CTC, train_one_epoch, train_one_epoch_CTC
import pickle
from collections import Counter

if TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter

    run = SummaryWriter()
else:
    import wandb
os.environ["WANDB_SILENT"] = "true"


def _maybe_parse_literal(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def _normalize_old_charset_for_mapping(old_charset, new_charset):
    if not old_charset or not new_charset:
        return old_charset

    new_example = new_charset[0]
    old_example = old_charset[0]

    if isinstance(new_example, str) and isinstance(old_example, int):
        normalized = []
        for item in old_charset:
            try:
                normalized.append(chr(int(item)))
            except Exception:
                normalized.append(str(item))
        return normalized

    if isinstance(new_example, int) and isinstance(old_example, str):
        normalized = []
        for item in old_charset:
            if isinstance(item, str) and len(item) == 1:
                normalized.append(ord(item))
            else:
                normalized.append(item)
        return normalized

    return old_charset


def _init_linear_from_random_rows(dst_linear, src_linear):
    """Initialize each output row from a random pretrained output row."""
    with torch.no_grad():
        src_rows = src_linear.weight.data.shape[0]
        indices = torch.randint(0, src_rows, (dst_linear.weight.data.shape[0],))
        dst_linear.weight.data.copy_(src_linear.weight.data[indices].clone())
        if dst_linear.bias is not None and src_linear.bias is not None:
            dst_linear.bias.data.copy_(src_linear.bias.data[indices].clone())


def _init_modulelist_from_random_rows(dst_modulelist, src_modulelist):
    seen = set()
    src_ref = src_modulelist[0]
    for module in dst_modulelist:
        module_id = id(module)
        if module_id in seen:
            continue
        _init_linear_from_random_rows(module, src_ref)
        seen.add(module_id)


def _freeze_all_but(model, trainable_parameters):
    trainable_ids = {id(p) for p in trainable_parameters}
    previous_requires_grad = {}
    for name, parameter in model.named_parameters():
        previous_requires_grad[name] = parameter.requires_grad
        parameter.requires_grad_(id(parameter) in trainable_ids)
    return previous_requires_grad


def _restore_requires_grad(model, previous_requires_grad):
    if previous_requires_grad is None:
        return
    for name, parameter in model.named_parameters():
        if name in previous_requires_grad:
            parameter.requires_grad_(previous_requires_grad[name])


def _best_cer_from_log(log_path):
    best_cer = float("inf")
    log_path = Path(log_path)
    if not log_path.is_file():
        return best_cer

    with log_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            for key in ("test_cer_oracle_direction", "test_cer"):
                value = record.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    best_cer = min(best_cer, float(value))
    return best_cer


def _resolve_resume_progress(
    checkpoint,
    resume_from_output_checkpoint,
    new_class_embedding,
    log_path,
):
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    global_step = int(checkpoint.get("global_step", 0))

    if resume_from_output_checkpoint and "epoch" in checkpoint:
        start_epoch = int(checkpoint["epoch"]) + 1
    elif new_class_embedding:
        start_epoch = 0
    elif "epoch" in checkpoint:
        start_epoch = int(checkpoint["epoch"]) + 1
    else:
        start_epoch = 0

    best_cer = float("inf")
    if resume_from_output_checkpoint:
        stored_best = checkpoint.get("best_cer")
        if isinstance(stored_best, (int, float)) and math.isfinite(float(stored_best)):
            best_cer = float(stored_best)
        else:
            best_cer = _best_cer_from_log(log_path)

    return start_epoch, global_step, best_cer


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument(
        "--coco_path", type=str, default="/comp_robot/cv_public_dataset/COCO2017/"
    )
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--pretrain_model_path", help="load from other checkpoint")
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")
    parser.add_argument("--new_class_embedding", action="store_true")
    parser.add_argument("--smart_mapping", action="store_true")
    parser.add_argument("--resume_finetuning", action="store_true")
    parser.add_argument("--path_old_charset", type=str, default=None)
    parser.add_argument("--random_erasing", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")

    return parser


class WithReplacementRandomSampler(Sampler):
    """Samples elements randomly, with replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # generate samples of `len(data_source)` that are of value from `0` to `len(data_source)-1`
        samples = torch.LongTensor(len(self.data_source))
        samples.random_(0, len(self.data_source))
        return iter(samples)

    def __len__(self):
        return len(self.data_source)


LENGTH_BIN_KEYS = ["1", "2", "3-5", "6-10", "11+"]


def _length_bin_from_gt_len(gt_len):
    gt_len = int(gt_len)
    if gt_len <= 1:
        return "1"
    if gt_len == 2:
        return "2"
    if gt_len <= 5:
        return "3-5"
    if gt_len <= 10:
        return "6-10"
    return "11+"


def _normalize_length_balance_weights(raw_weights):
    if raw_weights is None:
        return {"1": 4.0, "2": 3.0, "3-5": 2.0, "6-10": 1.0, "11+": 1.0}

    if isinstance(raw_weights, dict):
        normalized = {}
        for key in LENGTH_BIN_KEYS:
            normalized[key] = float(raw_weights.get(key, 1.0))
        return normalized

    if isinstance(raw_weights, (list, tuple)) and len(raw_weights) == len(LENGTH_BIN_KEYS):
        return {key: float(value) for key, value in zip(LENGTH_BIN_KEYS, raw_weights)}

    raise ValueError(
        "mth1000_length_sample_weights must be a dict keyed by "
        f"{LENGTH_BIN_KEYS} or a list/tuple of length {len(LENGTH_BIN_KEYS)}"
    )


def _build_length_balanced_sampler(dataset, args, logger):
    if not getattr(args, "mth1000_length_balance", False):
        return None

    if not hasattr(dataset, "samples"):
        logger.info(
            "Length-balanced sampling requested, but dataset has no `samples` attribute. "
            "Falling back to RandomSampler."
        )
        return None

    weights_cfg = _normalize_length_balance_weights(
        _maybe_parse_literal(getattr(args, "mth1000_length_sample_weights", None))
    )

    per_sample_weights = []
    raw_counts = Counter()
    weighted_counts = Counter()

    for sample in dataset.samples:
        text = sample.get("text", "") if isinstance(sample, dict) else ""
        bucket = _length_bin_from_gt_len(len(text))
        sample_weight = float(weights_cfg[bucket])
        per_sample_weights.append(sample_weight)
        raw_counts[bucket] += 1
        weighted_counts[bucket] += sample_weight

    if not per_sample_weights:
        logger.info(
            "Length-balanced sampling requested, but dataset is empty. "
            "Falling back to RandomSampler."
        )
        return None

    total_weight = sum(weighted_counts.values())
    effective_dist = {
        key: (weighted_counts[key] / total_weight if total_weight > 0 else 0.0)
        for key in LENGTH_BIN_KEYS
    }
    logger.info(
        "Using length-balanced WeightedRandomSampler with weights="
        f"{weights_cfg}, raw_counts={dict(raw_counts)}, effective_dist={effective_dist}"
    )

    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(per_sample_weights, dtype=torch.double),
        num_samples=len(per_sample_weights),
        replacement=True,
    )


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)

    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, "use_ema", None):
        args.use_ema = False
    if not getattr(args, "debug", None):
        args.debug = False
    if not hasattr(args, "mode_chr"):
        args.mode_chr = True
    if not getattr(args, "eval_epoch", None):
        args.eval_epoch = 1
    if not hasattr(args, "use_direction_head"):
        args.use_direction_head = False
    if not hasattr(args, "direction_loss_coef"):
        args.direction_loss_coef = 1.0
    if not hasattr(args, "decode_by_pred_direction"):
        args.decode_by_pred_direction = False
    if not hasattr(args, "deform_attn_debug"):
        args.deform_attn_debug = False
    if not hasattr(args, "deform_attn_force_pytorch"):
        args.deform_attn_force_pytorch = False
    if not hasattr(args, "deform_attn_fallback_on_error"):
        args.deform_attn_fallback_on_error = False
    if not hasattr(args, "deform_attn_dump_dir"):
        args.deform_attn_dump_dir = None
    if not hasattr(args, "direction_source"):
        args.direction_source = "label"
    if not hasattr(args, "ctc_blank_max"):
        args.ctc_blank_max = 1.0
    if not hasattr(args, "max_skipped_batches"):
        args.max_skipped_batches = 200
    if not hasattr(args, "mth1000_length_balance"):
        args.mth1000_length_balance = False
    if not hasattr(args, "mth1000_length_sample_weights"):
        args.mth1000_length_sample_weights = None
    if not hasattr(args, "step_lr_schedule"):
        args.step_lr_schedule = []
    if not hasattr(args, "exact_checkpoint_steps"):
        args.exact_checkpoint_steps = []
    if not hasattr(args, "max_optimizer_steps"):
        args.max_optimizer_steps = None
    if not hasattr(args, "new_class_warmup_steps"):
        args.new_class_warmup_steps = 0

    args.lr_drop_list = _maybe_parse_literal(getattr(args, "lr_drop_list", []))
    args.step_lr_schedule = _maybe_parse_literal(getattr(args, "step_lr_schedule", []))
    args.exact_checkpoint_steps = _maybe_parse_literal(
        getattr(args, "exact_checkpoint_steps", [])
    )
    args.max_optimizer_steps = _maybe_parse_literal(
        getattr(args, "max_optimizer_steps", None)
    )
    args.mth1000_length_balance = bool(
        _maybe_parse_literal(getattr(args, "mth1000_length_balance", False))
    )
    args.mth1000_length_sample_weights = _maybe_parse_literal(
        getattr(args, "mth1000_length_sample_weights", None)
    )
    args.new_class_warmup_steps = int(
        _maybe_parse_literal(getattr(args, "new_class_warmup_steps", 0)) or 0
    )

    if args.debug or args.deform_attn_debug:
        os.environ["DTLR_MSDA_DIAG"] = "1"
    if args.deform_attn_force_pytorch:
        os.environ["DTLR_MSDA_FORCE_PYTORCH"] = "1"
    if args.deform_attn_fallback_on_error:
        os.environ["DTLR_MSDA_FALLBACK_ON_ERROR"] = "1"
    if getattr(args, "deform_attn_dump_dir", None):
        os.environ["DTLR_MSDA_DUMP_DIR"] = str(args.deform_attn_dump_dir)

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(
        output=os.path.join(args.output_dir, "info.txt"),
        distributed_rank=args.rank,
        color=False,
        name="detr",
    )
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + " ".join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info("world size: {}".format(args.world_size))
    logger.info("rank: {}".format(args.rank))
    logger.info("local_rank: {}".format(args.local_rank))
    logger.info("args: " + str(args) + "\n")

    if not TENSORBOARD:
        run = wandb.init(project="OCRDETR-general-CTC", config=args, mode="disabled")
        run.define_metric("*", step_metric="global_step")

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()

    # build model

    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("number of params:" + str(n_parameters))
    logger.info(
        "params:\n"
        + json.dumps(
            {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
            indent=2,
        )
    )

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)
    train_fraction_manifest = getattr(dataset_train, "train_fraction_manifest", None)
    if train_fraction_manifest is not None:
        logger.info(
            "train_fraction_manifest:\n"
            + json.dumps(train_fraction_manifest, ensure_ascii=False, indent=2)
        )

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = _build_length_balanced_sampler(dataset_train, args, logger)
        if sampler_train is None:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    args.charset = dataset_train.charset
    new_class_warmup_parameters = None

    if args.new_class_embedding and args.resume_finetuning:
        device = args.device

        features_dim = model.class_embed[0].weight.data.shape[1]
        new_charset_size = len(
            args.charset
        )  ## Size of the charset corresponding to the new dataset

        new_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )
        new_decoder_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )
        new_enc_out_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )

        if model.dec_pred_class_embed_share:
            class_embed_layerlist = [
                new_class_embed for i in range(model.transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(new_class_embed)
                for i in range(model.transformer.num_decoder_layers)
            ]
        new_class_embed = nn.ModuleList(class_embed_layerlist)

        if args.smart_mapping:
            if args.path_old_charset is not None:
                old_charset = pickle.load(open(args.path_old_charset, "rb"))
            else:
                with open(
                    os.path.join(
                        os.path.dirname(datasets.__file__), "default_charset.json"
                    ),
                    "r",
                ) as f:
                    old_charset = json.load(f)
            old_charset = _normalize_old_charset_for_mapping(
                old_charset, args.charset
            )
            not_mapped = []
            possible_mapping = list(range(len(old_charset)))
            mapping = {}
            for i, char in enumerate(args.charset):
                if char in old_charset:
                    mapping[i] = old_charset.index(char)
                    possible_mapping.remove(mapping[i])
                else:
                    not_mapped.append(char)

            while len(possible_mapping) < len(not_mapped):
                possible_mapping.append(np.random.randint(0, len(old_charset)))
            possible_mapping = list(np.random.permutation(possible_mapping))

            for i, char in enumerate(args.charset):
                if char not in old_charset:
                    mapping[i] = possible_mapping[0]
                    possible_mapping.pop(0)

            assert len(mapping) == len(args.charset)

            for j in range(model.transformer.num_decoder_layers):
                for i in range(new_charset_size):
                    new_class_embed[j].weight.data[i, :] = model.class_embed[
                        j
                    ].weight.data[mapping[i], :]
                    new_class_embed[j].bias.data[i] = model.class_embed[j].bias.data[
                        mapping[i]
                    ]

                    new_decoder_class_embed.weight.data[i, :] = (
                        model.transformer.decoder.class_embed[
                            j
                        ].weight.data[mapping[i], :]
                    )
                    new_decoder_class_embed.bias.data[i] = (
                        model.transformer.decoder.class_embed[j].bias.data[mapping[i]]
                    )

                    new_enc_out_class_embed.weight.data[i, :] = (
                        model.transformer.enc_out_class_embed.weight.data[mapping[i], :]
                    )
                    new_enc_out_class_embed.bias.data[i] = (
                        model.transformer.enc_out_class_embed.bias.data[mapping[i]]
                    )
        if args.smart_mapping:
            model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
            model.class_embed = new_class_embed.to(device)
            model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
            model.label_enc = nn.Embedding(
                len(dataset_val.charset) + 1, features_dim
            ).to(device)
            parameters_to_optimize = (
                list(model.class_embed.parameters())
                + list(model.transformer.decoder.class_embed.parameters())
                + list(model.transformer.enc_out_class_embed.parameters())
            )
        else:
            _init_modulelist_from_random_rows(new_class_embed, model.class_embed)
            _init_linear_from_random_rows(
                new_decoder_class_embed, model.transformer.decoder.class_embed[0]
            )
            _init_linear_from_random_rows(
                new_enc_out_class_embed, model.transformer.enc_out_class_embed
            )
            model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
            model.class_embed = new_class_embed.to(device)
            model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
            model.label_enc = nn.Embedding(
                len(dataset_val.charset) + 1, features_dim
            ).to(
                device
            )  ### This is used for the dn process but will not be used during the finetuning, we define it to avoid errors

            parameters_to_optimize = (
                list(model.class_embed.parameters())
                + list(model.transformer.decoder.class_embed.parameters())
                + list(model.transformer.enc_out_class_embed.parameters())
            )

        if getattr(args, "use_direction_head", False) and hasattr(model, "direction_embed"):
            parameters_to_optimize += list(model.direction_embed.parameters())

        # optimizer = torch.optim.AdamW(parameters_to_optimize, lr=args.lr,weight_decay=args.weight_decay)

    def _build_scheduler(current_optimizer):
        if getattr(args, "auto_lr_plateau", False):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                current_optimizer,
                mode=getattr(args, "auto_lr_mode", "min"),
                factor=float(getattr(args, "auto_lr_factor", 0.5)),
                patience=int(getattr(args, "auto_lr_patience", 1)),
                threshold=float(getattr(args, "auto_lr_threshold", 1e-4)),
                threshold_mode=getattr(args, "auto_lr_threshold_mode", "rel"),
                cooldown=int(getattr(args, "auto_lr_cooldown", 0)),
                min_lr=float(getattr(args, "auto_lr_min_lr", 1e-6)),
            )

        if args.onecyclelr:
            return torch.optim.lr_scheduler.OneCycleLR(
                current_optimizer,
                max_lr=args.lr,
                steps_per_epoch=len(data_loader_train),
                epochs=args.epochs,
                pct_start=float(getattr(args, "onecycle_pct_start", 0.2)),
                div_factor=float(getattr(args, "onecycle_div_factor", 25.0)),
                final_div_factor=float(
                    getattr(args, "onecycle_final_div_factor", 1e4)
                ),
            )

        if args.multi_step_lr:
            return torch.optim.lr_scheduler.MultiStepLR(
                current_optimizer,
                milestones=list(args.lr_drop_list),
                gamma=float(getattr(args, "multi_step_gamma", 0.1)),
            )

        return torch.optim.lr_scheduler.StepLR(
            current_optimizer,
            args.lr_drop,
            gamma=float(getattr(args, "step_lr_gamma", 0.1)),
        )
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu", weights_only=False)

        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(args.output_dir)

    def _load_compatible_state(model_obj, state_dict_raw):
        clean_state = utils.clean_state_dict(state_dict_raw)
        model_state = model_obj.state_dict()
        filtered = {}
        skipped = []
        for k, v in clean_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        load_out = model_obj.load_state_dict(filtered, strict=False)
        logger.info(str(load_out))
        if len(skipped) > 0:
            logger.info(f"Skipped {len(skipped)} incompatible keys during load.")

    output_checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    resume_from_output_checkpoint = bool(
        args.output_dir and os.path.isfile(output_checkpoint_path)
    )
    if resume_from_output_checkpoint:
        args.resume = output_checkpoint_path
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)

        _load_compatible_state(model_without_ddp, checkpoint["model"])
        if args.use_ema:
            if "ema_model" in checkpoint:
                ema_m.module.load_state_dict(
                    utils.clean_state_dict(checkpoint["ema_model"])
                )
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)

    if args.new_class_embedding and not args.resume_finetuning:
        device = args.device

        features_dim = model.class_embed[0].weight.data.shape[1]
        new_charset_size = len(
            args.charset
        )  ## Size of the charset corresponding to the new dataset

        new_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )
        new_decoder_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )
        new_enc_out_class_embed = nn.Linear(
            features_dim,
            new_charset_size,
        )

        if model.dec_pred_class_embed_share:
            class_embed_layerlist = [
                new_class_embed for i in range(model.transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(new_class_embed)
                for i in range(model.transformer.num_decoder_layers)
            ]
        new_class_embed = nn.ModuleList(class_embed_layerlist)

        if args.smart_mapping:
            if args.path_old_charset is not None:
                old_charset = pickle.load(open(args.path_old_charset, "rb"))
            else:
                with open(
                    os.path.join(
                        os.path.dirname(datasets.__file__), "default_charset.json"
                    ),
                    "r",
                ) as f:
                    old_charset = json.load(f)
            old_charset = _normalize_old_charset_for_mapping(
                old_charset, args.charset
            )

            not_mapped = []
            possible_mapping = list(range(len(old_charset)))
            mapping = {}
            for i, char in enumerate(args.charset):
                if char in old_charset:
                    mapping[i] = old_charset.index(char)
                    possible_mapping.remove(mapping[i])
                else:
                    not_mapped.append(char)

            while len(possible_mapping) < len(not_mapped):
                possible_mapping.append(np.random.randint(0, len(old_charset)))
            possible_mapping = list(np.random.permutation(possible_mapping))

            for i, char in enumerate(args.charset):
                if char not in old_charset:
                    mapping[i] = possible_mapping[0]
                    possible_mapping.pop(0)

            assert len(mapping) == len(args.charset)
            for j in range(model.transformer.num_decoder_layers):
                for i in range(new_charset_size):
                    new_class_embed[j].weight.data[i, :] = model.class_embed[
                        j
                    ].weight.data[mapping[i], :]

                    new_class_embed[j].bias.data[i] = model.class_embed[j].bias.data[
                        mapping[i]
                    ]

                    new_decoder_class_embed.weight.data[i, :] = (
                        model.transformer.decoder.class_embed[
                            j
                        ].weight.data[mapping[i], :]
                    )
                    new_decoder_class_embed.bias.data[i] = (
                        model.transformer.decoder.class_embed[j].bias.data[mapping[i]]
                    )

                    new_enc_out_class_embed.weight.data[i, :] = (
                        model.transformer.enc_out_class_embed.weight.data[mapping[i], :]
                    )
                    new_enc_out_class_embed.bias.data[i] = (
                        model.transformer.enc_out_class_embed.bias.data[mapping[i]]
                    )
        if args.smart_mapping:
            model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
            model.class_embed = new_class_embed.to(device)
            model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
            model.label_enc = nn.Embedding(
                len(dataset_val.charset) + 1, features_dim
            ).to(device)
            parameters_to_optimize = (
                list(model.class_embed.parameters())
                + list(model.transformer.decoder.class_embed.parameters())
                + list(model.transformer.enc_out_class_embed.parameters())
            )

        else:
            _init_modulelist_from_random_rows(new_class_embed, model.class_embed)
            _init_linear_from_random_rows(
                new_decoder_class_embed, model.transformer.decoder.class_embed[0]
            )
            _init_linear_from_random_rows(
                new_enc_out_class_embed, model.transformer.enc_out_class_embed
            )
            model.class_embed = new_class_embed.to(device)
            model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
            model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
            model.label_enc = nn.Embedding(
                len(dataset_val.charset) + 1, features_dim
            ).to(device)

            parameters_to_optimize = (
                list(model.class_embed.parameters())
                + list(model.transformer.decoder.class_embed.parameters())
                + list(model.transformer.enc_out_class_embed.parameters())
            )

        if getattr(args, "use_direction_head", False) and hasattr(model, "direction_embed"):
            parameters_to_optimize += list(model.direction_embed.parameters())

        new_class_warmup_parameters = parameters_to_optimize
        optimizer = torch.optim.AdamW(
            parameters_to_optimize, lr=args.lr, weight_decay=args.weight_decay
        )

        # When both --resume and --new_class_embedding are enabled, reload once
        # after rebuilding heads so classifier weights can be restored correctly.
        if args.resume:
            if args.resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            _load_compatible_state(model_without_ddp, checkpoint["model"])

    if (not args.resume) and args.pretrain_model_path:
        checkpoint_full = torch.load(args.pretrain_model_path, map_location="cpu", weights_only=False)
        checkpoint = checkpoint_full["model"]
        # `pretrain_model_path` is used for weight initialization only.
        # Do not inherit the source checkpoint epoch into a new finetuning run.
        from collections import OrderedDict

        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict(
            {
                k: v
                for k, v in utils.clean_state_dict(checkpoint).items()
                if check_keep(k, _ignorekeywordlist)
            }
        )
        for k, v in utils.clean_state_dict(checkpoint).items():
            if check_keep(k, _ignorekeywordlist) is False:
                print("Ignored: {}".format(k))

        _load_compatible_state(model_without_ddp, _tmp_st)

        if args.use_ema:
            if "ema_model" in checkpoint:
                ema_m.module.load_state_dict(
                    utils.clean_state_dict(checkpoint["ema_model"])
                )
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)

    # Optimizer can be rebuilt in new_class_embedding branches above.
    # Always build scheduler after optimizer is finalized.
    lr_scheduler = _build_scheduler(optimizer)
    if resume_from_output_checkpoint:
        missing_training_state = [
            key
            for key in ("optimizer", "lr_scheduler")
            if key not in checkpoint
        ]
        if missing_training_state:
            raise RuntimeError(
                "Exact output-directory resume requires checkpoint training state: "
                + ", ".join(missing_training_state)
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        logger.info(
            "Restored optimizer and lr_scheduler from "
            f"{output_checkpoint_path}"
        )

    if args.eval:
        run = wandb.init(project="OCRDETR-general-CTC", config=args, mode="disabled")

        run.define_metric("*", step_metric="global_step")
        os.environ["EVAL_FLAG"] = "TRUE"
        test_stats, coco_evaluator = evaluate_CTC(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
            wo_class_error=wo_class_error,
            args=args,
            run=run,
            mode_chr=args.mode_chr,
        )
        if args.output_dir and coco_evaluator is not None:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )

        log_stats = {**{f"test_{k}": v for k, v in test_stats.items()}}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    structure_teacher = None
    structure_retention_coef = float(
        getattr(args, "query_geometry_retention_coef", 0.0)
    )
    if structure_retention_coef > 0.0:
        teacher_path = getattr(args, "query_geometry_teacher_checkpoint", None)
        if not teacher_path:
            raise ValueError(
                "query_geometry_teacher_checkpoint is required when "
                "query_geometry_retention_coef > 0"
            )
        teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
        structure_teacher = copy.deepcopy(model_without_ddp)
        _load_compatible_state(structure_teacher, teacher_checkpoint["model"])
        structure_teacher.to(device)
        structure_teacher.eval()
        for parameter in structure_teacher.parameters():
            parameter.requires_grad_(False)
        logger.info(
            "Enabled query geometry retention: "
            f"teacher={teacher_path}, coef={structure_retention_coef}, "
            "axes=(vertical_center,height)"
        )

    # model.class_embed = new_class_embed

    print("Start training")
    checkpoint_for_progress = checkpoint if "checkpoint" in locals() else None
    args.start_epoch, global_step, best_cer = _resolve_resume_progress(
        checkpoint_for_progress,
        resume_from_output_checkpoint,
        args.new_class_embedding,
        output_dir / "log.txt",
    )
    if resume_from_output_checkpoint:
        logger.info(
            "Exact resume progress: "
            f"start_epoch={args.start_epoch}, global_step={global_step}, "
            f"best_cer={best_cer}"
        )

    exact_checkpoint_steps = sorted(
        {int(step) for step in getattr(args, "exact_checkpoint_steps", [])}
    )
    saved_exact_checkpoint_steps = set()
    warmup_state = {
        "active": False,
        "switch_step": int(getattr(args, "new_class_warmup_steps", 0)),
        "saved_requires_grad": None,
        "switched": False,
    }

    if (
        args.new_class_embedding
        and not args.resume_finetuning
        and warmup_state["switch_step"] > 0
        and new_class_warmup_parameters is not None
    ):
        if global_step < warmup_state["switch_step"]:
            warmup_state["saved_requires_grad"] = _freeze_all_but(
                model_without_ddp, new_class_warmup_parameters
            )
            warmup_state["active"] = True
            if logger is not None:
                logger.info(
                    "Stage-1 finetuning active: optimizing only re-initialized "
                    f"classification parameters until global_step={warmup_state['switch_step']}"
                )
        else:
            if logger is not None:
                logger.info(
                    "Skipping stage-1 class-only warmup because global_step "
                    f"{global_step} >= {warmup_state['switch_step']}"
                )

    def _build_weights(epoch, global_step_value):
        weights = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "global_step": int(global_step_value),
            "best_cer": float(best_cer),
            "args": args,
        }
        if args.use_ema:
            weights.update({"ema_model": ema_m.module.state_dict()})
        return weights

    def _save_exact_step_checkpoint(global_step_value):
        if global_step_value not in exact_checkpoint_steps:
            return
        if global_step_value in saved_exact_checkpoint_steps:
            return

        checkpoint_path = output_dir / f"checkpoint_step{int(global_step_value)}.pth"
        utils.save_on_master(_build_weights(epoch, global_step_value), checkpoint_path)
        if exact_checkpoint_steps and global_step_value == exact_checkpoint_steps[0]:
            utils.save_on_master(
                _build_weights(epoch, global_step_value),
                output_dir / "checkpoint_stage1_end.pth",
            )
        saved_exact_checkpoint_steps.add(global_step_value)
        if logger is not None:
            logger.info(
                f"Saved exact-step checkpoint at global_step={int(global_step_value)}"
            )

    def _after_step(global_step_value):
        nonlocal optimizer, lr_scheduler
        _save_exact_step_checkpoint(global_step_value)

        if (
            warmup_state["active"]
            and (not warmup_state["switched"])
            and global_step_value >= warmup_state["switch_step"]
        ):
            _restore_requires_grad(
                model_without_ddp, warmup_state["saved_requires_grad"]
            )
            optimizer = torch.optim.AdamW(
                get_param_dict(args, model_without_ddp),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            lr_scheduler = _build_scheduler(optimizer)
            if getattr(args, "step_lr_schedule", []):
                schedule = getattr(args, "step_lr_schedule", [])
                if len(schedule) > 0:
                    lr_value = schedule[0][1]
                    for start_step, candidate_lr in schedule:
                        if global_step_value >= int(start_step):
                            lr_value = candidate_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_value
            warmup_state["active"] = False
            warmup_state["switched"] = True
            if logger is not None:
                logger.info(
                    "Switching to stage-2 end-to-end finetuning at "
                    f"global_step={global_step_value}"
                )
            return optimizer, lr_scheduler

        return None

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        try:
            dataset_train.generates_synthetic_data()
        except:
            pass

        train_stats, global_step = train_one_epoch_CTC(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            wo_class_error=wo_class_error,
            lr_scheduler=lr_scheduler,
            args=args,
            logger=(logger if args.save_log else None),
            ema_m=ema_m,
            run=run,
            global_step_start=global_step,
            step_callback=_after_step,
            structure_teacher=structure_teacher,
        )

        if epoch % args.eval_epoch == 0:
            args.num_classes = len(dataset_train.charset)
            __, criterion, postprocessors = build_model_main(args)
            test_stats, coco_evaluator = evaluate_CTC(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
                wo_class_error=wo_class_error,
                args=args,
                logger=(logger if args.save_log else None),
                run=run,
                epoch=epoch,
                mode_chr=args.mode_chr,
            )
        
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }

            current_cer = test_stats.get("cer_oracle_direction", None)
            if current_cer is None:
                current_cer = test_stats.get("cer", None)
            is_best = current_cer is not None and current_cer < best_cer
            if is_best:
                best_cer = current_cer
                checkpoint_path = output_dir / "checkpoint_best_regular.pth"
                utils.save_on_master(
                    _build_weights(epoch, global_step),
                    checkpoint_path,
                )
                if logger is not None:
                    logger.info(
                        f"New best checkpoint at epoch {epoch}, cer_oracle_direction={best_cer:.6f}"
                    )
            elif logger is not None and current_cer is not None:
                logger.info(
                    f"Epoch {epoch} not best. cer_oracle_direction={current_cer:.6f}, best={best_cer:.6f}"
                )

            ep_paras = {"epoch": epoch, "n_parameters": n_parameters}
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats.update(ep_paras)
            log_stats["epoch_time"] = epoch_time_str

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                output_dir / "eval" / name,
                            )

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]

        if getattr(args, "step_lr_schedule", []):
            pass
        elif getattr(args, "auto_lr_plateau", False):
            metric_for_plateau = None
            if epoch % args.eval_epoch == 0:
                metric_for_plateau = test_stats.get("cer_oracle_direction", None)
                if metric_for_plateau is None:
                    metric_for_plateau = test_stats.get("cer", None)
                if metric_for_plateau is None:
                    metric_for_plateau = test_stats.get("loss_CTC_unscaled", None)
                if metric_for_plateau is None:
                    metric_for_plateau = test_stats.get("loss", None)
            if metric_for_plateau is None:
                metric_for_plateau = train_stats.get("loss_CTC_unscaled", None)
            if metric_for_plateau is None:
                metric_for_plateau = train_stats.get("loss", 0.0)
            lr_scheduler.step(metric_for_plateau)
        elif not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(_build_weights(epoch, global_step), checkpoint_path)

        if getattr(args, "max_optimizer_steps", None) is not None and global_step >= int(
            args.max_optimizer_steps
        ):
            if logger is not None:
                logger.info(
                    f"Reached max_optimizer_steps={int(args.max_optimizer_steps)} at epoch {epoch}"
                )
            break

        # eval ema
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate_CTC(
                ema_m.module,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
                wo_class_error=wo_class_error,
                args=args,
                logger=(logger if args.save_log else None),
            )
            log_stats.update({f"ema_test_{k}": v for k, v in ema_test_stats.items()})
            if True:
                checkpoint_path = output_dir / "checkpoint_best_ema.pth"
                utils.save_on_master(
                    {
                        "model": ema_m.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        ep_paras = {"epoch": epoch, "n_parameters": n_parameters}

        try:
            log_stats.update({"now_time": str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get("copyfilelist")
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove

        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finetuning", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
