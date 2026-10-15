# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
from util.utils import to_device
import numpy as np
import torch
import torch.nn.functional as F
import util.misc as utils
try:
    from datasets.panoptic_eval import PanopticEvaluator
except ModuleNotFoundError:
    PanopticEvaluator = None
import re
import editdistance
from util.visualizer import COCOVisualizer

from util import box_ops
vslzr = COCOVisualizer()


def _get_scheduler_lrs(lr_scheduler, optimizer):
    if lr_scheduler is not None and hasattr(lr_scheduler, "get_last_lr"):
        try:
            return lr_scheduler.get_last_lr()
        except Exception:
            pass
    return [group["lr"] for group in optimizer.param_groups]


def _apply_manual_step_lr(optimizer, step_schedule, global_step):
    if not step_schedule:
        return

    target_lr = None
    for step_boundary, step_lr in step_schedule:
        if global_step >= int(step_boundary):
            target_lr = float(step_lr)
        else:
            break

    if target_lr is None:
        return

    for group in optimizer.param_groups:
        group["lr"] = target_lr


def _direction_to_label(direction_value):
    if torch.is_tensor(direction_value):
        direction_value = int(direction_value.item())
    return "vertical" if int(direction_value) == 1 else "horizontal"


def _get_target_direction(target_item):
    if isinstance(target_item, dict) and "direction" in target_item:
        return _direction_to_label(target_item["direction"])
    return "horizontal"


def _get_decode_directions(outputs, targets, decode_by_pred_direction=False):
    if decode_by_pred_direction and "pred_direction" in outputs:
        return [
            _direction_to_label(item)
            for item in outputs["pred_direction"].argmax(-1)
        ]
    return [_get_target_direction(target) for target in targets]


def _sort_pred_logits_by_direction(pred_logits, pred_boxes, directions):
    idx_list = []
    for i, direction in enumerate(directions):
        sort_axis = 1 if direction == "vertical" else 0
        _, idx_i = torch.sort(pred_boxes[i, :, sort_axis], descending=False)
        idx_list.append(idx_i)

    idx = torch.stack(idx_list, dim=0)
    return torch.gather(
        pred_logits,
        1,
        idx.unsqueeze(-1).expand(-1, -1, pred_logits.shape[-1]),
    )


def compute_query_geometry_retention_loss(
    student_outputs,
    teacher_outputs,
    targets,
    confidence_power=1.0,
):
    """Keep high-confidence vertical query geometry near a frozen teacher."""
    student_boxes = student_outputs["pred_boxes"]
    teacher_boxes = teacher_outputs["pred_boxes"].detach()
    teacher_scores = teacher_outputs["pred_logits"].detach().sigmoid().amax(-1)

    sample_losses = []
    selected_queries = 0
    for batch_idx, target in enumerate(targets):
        target_length = int(target["labels"].numel())
        query_count = int(teacher_scores.shape[1])
        selected_count = min(target_length, query_count)
        if selected_count <= 0:
            continue

        scores, indices = teacher_scores[batch_idx].topk(selected_count)
        # Vertical center and height are the geometry used to establish and
        # stabilize the reading sequence. Horizontal extent is intentionally
        # left free to adapt across domains.
        student_geometry = student_boxes[batch_idx, indices][:, [1, 3]]
        teacher_geometry = teacher_boxes[batch_idx, indices][:, [1, 3]]
        per_query = F.smooth_l1_loss(
            student_geometry,
            teacher_geometry,
            reduction="none",
        ).mean(-1)
        weights = scores.clamp(min=1e-4).pow(float(confidence_power))
        sample_losses.append((per_query * weights).sum() / weights.sum())
        selected_queries += selected_count

    if not sample_losses:
        return student_boxes.sum() * 0.0, {"selected_queries": 0.0}
    return torch.stack(sample_losses).mean(), {
        "selected_queries": float(selected_queries) / len(sample_losses)
    }


def _compute_direction_accuracy(outputs, targets):
    if "pred_direction" not in outputs:
        return None

    target_directions = []
    valid_indices = []
    for idx, target in enumerate(targets):
        if isinstance(target, dict) and "direction" in target:
            valid_indices.append(idx)
            target_directions.append(1 if _get_target_direction(target) == "vertical" else 0)

    if len(valid_indices) == 0:
        return None

    pred = outputs["pred_direction"].argmax(-1).detach().cpu()
    correct = 0
    for idx, target_direction in zip(valid_indices, target_directions):
        correct += int(pred[idx].item() == target_direction)
    return correct / len(valid_indices)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,run = None, postprocessors = None, output_dir = None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    iterations = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        try:
            samples = samples.to(device)


            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(enabled=args.amp):
                if need_tgt_for_training:
                    outputs = model(samples, targets)
                else:
                    outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict

                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            run.log(loss_dict)
            iterations += len(targets)
            run.log({'global_step': iterations+len(data_loader) * epoch})
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()


            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            # amp backward function
            if args.amp:
                optimizer.zero_grad()
                scaler.scale(losses).backward()
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # original backward function
                optimizer.zero_grad()
                losses.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()


            if args.use_ema:
                if epoch >= args.ema_epoch:
                    ema_m.update(model)

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            if 'class_error' in loss_dict_reduced:

                metric_logger.update(class_error=loss_dict_reduced['class_error'])

            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            _cnt += 1
            if args.debug:
                if _cnt % 15 == 0:
                    print("BREAK!"*5)
                    break
        except Exception as e:
            print(e)
            continue
        

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()


    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})

    return resstat

def save_snapchot(image,output,targets,charset, postprocessors,output_dir, epoch = 0 ):
    outputs = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda().repeat(len(targets), 1))
    images,__ = image.decompose()
    output_dir = os.path.join(output_dir, f'epoch_{epoch}')
    for it, (image, output, target) in enumerate(zip(images, outputs, targets)):
   
        image = image.cpu()

        thershold = 0.1#
        
        scores = output['scores'].cpu()
        labels = output['labels'].cpu()
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'].cpu())
        select_mask = scores > thershold

        box_label = [charset[int(item)] for item in labels[select_mask]]
        pred_dict = {
            'boxes': boxes[select_mask],
            'size': target['size'].cpu(),
            'box_label': box_label,
            'image_id': 0,
        }
        # create folder epoch
        
        os.makedirs(output_dir, exist_ok=True)
        vslzr.visualize(image, pred_dict, savedir=output_dir, it = it, fontsize=8, offset=90)

def add_prefix_to_keys(dictionary, prefix):
    new_dict = {}
    for key, value in dictionary.items():
        new_key = prefix + str(key)
        new_dict[new_key] = value
    return new_dict





def train_one_epoch_CTC(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,run = None,
                    global_step_start=0, step_callback=None, structure_teacher=None):

    model.train()
   # criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0

    old_wer = 0
    old_cer = 0
    iterations = 0
    global_step = int(global_step_start)
    it_loader = 0
    it_CER = 0 
    skipped_batches = 0
    max_skipped_batches = getattr(args, "max_skipped_batches", 200)
    step_lr_schedule = getattr(args, "step_lr_schedule", [])
    max_optimizer_steps = getattr(args, "max_optimizer_steps", None)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger): ## to modify
        try:
            _apply_manual_step_lr(optimizer, step_lr_schedule, global_step)
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
         
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(samples, targets)
            
            loss_dict, preds, new_preds= criterion.loss_CTC(outputs, targets,None,None, return_preds = True)
            weight_dict = dict(criterion.weight_dict)
            structure_retention_coef = float(
                getattr(args, "query_geometry_retention_coef", 0.0)
            )
            if structure_teacher is not None and structure_retention_coef > 0.0:
                with torch.no_grad():
                    teacher_outputs = structure_teacher(samples, targets)
                retention_loss, retention_stats = compute_query_geometry_retention_loss(
                    outputs,
                    teacher_outputs,
                    targets,
                    confidence_power=float(
                        getattr(args, "query_geometry_retention_confidence_power", 1.0)
                    ),
                )
                loss_dict["loss_query_geometry_retention"] = retention_loss
                loss_dict["query_geometry_retention_selected"] = retention_loss.new_tensor(
                    retention_stats["selected_queries"]
                )
                weight_dict["loss_query_geometry_retention"] = structure_retention_coef
            loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

            loss_value = loss.item()
            loss_dict_wandb = {}
            ## copy the loss dict to wandb but add "train_" prefix to the keys
            for key, value in loss_dict.items():
                loss_dict_wandb["train_" + key] = value
            
            if run is not None:
                run.log(loss_dict_wandb)

            loss_dict_reduced_scaled = {
                k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict
            }
            loss_dict_unscaled = {
                f"{k}_unscaled": v for k, v in loss_dict.items() if k in weight_dict
            }

            
            if run is not None:
                run.log({'global_step': global_step})
            if it_loader % 100 == 99:
        
                batch_size = outputs["pred_logits"].shape[0]
                it_CER += 1
                wer_it, cer_it = compute_wer(
                    outputs,
                    targets,
                    args.charset,
                    preds,
                    mode_chr=args.mode_chr,
                    decode_by_pred_direction=getattr(args, "decode_by_pred_direction", False),
                    preds_already_sorted=True,
                )
                old_wer += wer_it / batch_size
                old_cer += cer_it / batch_size
                if run is not None:
                    run.log({"train_wer":old_wer/it_CER})
                    run.log({"train_cer":old_cer/it_CER})
                direction_acc = _compute_direction_accuracy(outputs, targets)
                if direction_acc is not None:
                    if run is not None:
                        run.log({"train_direction_acc": direction_acc})
            it_loader +=1
            


            if not math.isfinite(loss_value):
                
                print("Loss is {}, stopping training".format(loss_value))
                print(loss)
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            if args.onecyclelr:
                lr_scheduler.step()

            global_step += 1
            if step_callback is not None:
                callback_result = step_callback(global_step)
                if callback_result is not None:
                    if (
                        isinstance(callback_result, tuple)
                        and len(callback_result) == 2
                    ):
                        optimizer, lr_scheduler = callback_result
                    else:
                        optimizer = callback_result


            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_unscaled)
            if "direction_acc" in loss_dict:
                metric_logger.update(direction_acc=loss_dict["direction_acc"].item())

            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            _cnt += 1
            if args.debug:
                if _cnt % 15 == 0:
                    print("BREAK!"*5)
                    break 
        except Exception as e:
            skipped_batches += 1
            batch_shapes = None
            batch_target_info = []
            try:
                if hasattr(samples, "tensors"):
                    batch_shapes = tuple(samples.tensors.shape)
                elif torch.is_tensor(samples):
                    batch_shapes = tuple(samples.shape)
            except Exception:
                batch_shapes = "unavailable"

            try:
                for t in targets:
                    info = {}
                    for key in ["image_id", "size", "orig_size", "direction", "labels"]:
                        if key not in t:
                            continue
                        value = t[key]
                        if torch.is_tensor(value):
                            if value.numel() == 1:
                                info[key] = value.item()
                            elif key == "labels":
                                info[key] = {
                                    "shape": tuple(value.shape),
                                    "min": value.min().item() if value.numel() > 0 else None,
                                    "max": value.max().item() if value.numel() > 0 else None,
                                }
                            else:
                                info[key] = tuple(value.detach().cpu().tolist())
                        else:
                            info[key] = value
                    info["keys"] = sorted(list(t.keys()))
                    batch_target_info.append(info)
            except Exception:
                batch_target_info = ["failed_to_collect_target_info"]

            if logger is not None:
                logger.warning(f"[train_one_epoch_CTC] skipped one batch due to: {e}")
                logger.warning(
                    f"[train_one_epoch_CTC] batch_shapes={batch_shapes} batch_target_info={batch_target_info}"
                )
                logger.warning(
                    f"[train_one_epoch_CTC] skipped_batches={skipped_batches}/{max_skipped_batches}"
                )
            else:
                print(f"[train_one_epoch_CTC] skipped one batch due to: {e}")
                print(
                    f"[train_one_epoch_CTC] batch_shapes={batch_shapes} batch_target_info={batch_target_info}"
                )
            if skipped_batches >= max_skipped_batches:
                raise RuntimeError(
                    f"Too many skipped batches: {skipped_batches}. "
                    "Stop training to avoid silent failure."
                )
            continue
    
        iterations += len(targets)
        if max_optimizer_steps is not None and global_step >= int(max_optimizer_steps):
            break
        if  iterations >= args.max_iterations:
            break


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()


    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})

    if run is not None:
        run.log({'lr_scheduler': _get_scheduler_lrs(lr_scheduler, optimizer)})
    return resstat, global_step


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        if PanopticEvaluator is None:
            raise RuntimeError(
                "Panoptic evaluation is not available in this SAQT repository cleanup. "
                "Use the line-level recognition/detection evaluation paths instead."
            )
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)


            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    return stats, None# coco_evaluator

@torch.no_grad()
def evaluate_CTC(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None,run = None,epoch = 0, mode_chr =False):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    old_cer = 0
    old_wer = 0
    pred_old_cer = 0
    pred_old_wer = 0
    has_pred_direction_metrics = False
    direction_acc_sum = 0
    direction_acc_batches = 0
    iterations = 0
    predicted_str_total = []
    idx_list = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
       
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

    
        loss_dict, preds,__= criterion.loss_CTC(outputs, targets,None,None, return_preds = True)
        
        loss_dict_wandb = {}
        ## copy the loss dict to wandb but add "train_" prefix to the keys
        for key, value in loss_dict.items():
            loss_dict_wandb["test_" + key] = value
        
        if run is not None:
            run.log(loss_dict_wandb)

        weight_dict = criterion.weight_dict


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        if "direction_acc" in loss_dict:
            metric_logger.update(direction_acc=loss_dict["direction_acc"].item())

        if iterations == 0:
            try:
                save_snapchot(samples,outputs,targets,args.charset, postprocessors, output_dir = output_dir, epoch = epoch)
            except:
                pass
        
        wer_it_oracle, cer_it_oracle, predicted_str_total_it = compute_wer(
            outputs,
            targets,
            args.charset,
            preds,
            return_preds=True,
            mode_chr=mode_chr,
            decode_by_pred_direction=False,
            preds_already_sorted=True,
        )
        old_wer += wer_it_oracle
        old_cer += cer_it_oracle

        if "pred_direction" in outputs:
            has_pred_direction_metrics = True
            pred_wer, pred_cer = compute_wer(
                outputs,
                targets,
                args.charset,
                preds,
                mode_chr=mode_chr,
                decode_by_pred_direction=True,
                preds_already_sorted=True,
            )
            pred_old_wer += pred_wer
            pred_old_cer += pred_cer
            direction_acc = _compute_direction_accuracy(outputs, targets)
            if direction_acc is not None:
                direction_acc_sum += direction_acc
                direction_acc_batches += 1

        iterations += len(targets)
        for i in range(len(predicted_str_total_it)):
            predicted_str_total.append(predicted_str_total_it[i])
            idx_list.append(targets[i]['img_idx'].item())

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if run is not None:
        run.log({"test_wer_oracle_direction":old_wer/iterations})
        run.log({"test_cer_oracle_direction":old_cer/iterations})
    if has_pred_direction_metrics:
        if run is not None:
            run.log({"test_wer_pred_direction":pred_old_wer/iterations})
            run.log({"test_cer_pred_direction":pred_old_cer/iterations})
    if direction_acc_batches > 0:
        if run is not None:
            run.log({"test_direction_acc": direction_acc_sum / direction_acc_batches})
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    stats["wer_oracle_direction"] = old_wer / iterations
    stats["cer_oracle_direction"] = old_cer / iterations
    if has_pred_direction_metrics:
        stats["wer_pred_direction"] = pred_old_wer / iterations
        stats["cer_pred_direction"] = pred_old_cer / iterations
    if direction_acc_batches > 0:
        stats["direction_acc"] = direction_acc_sum / direction_acc_batches

    return stats, None

@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())


    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        if PanopticEvaluator is None:
            raise RuntimeError(
                "Panoptic evaluation is not available in this SAQT repository cleanup. "
                "Use the line-level recognition/detection evaluation paths instead."
            )
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res



@torch.no_grad()
def format_string_for_wer(str):
    """
    Format string for WER computation: remove layout tokens, treat punctuation as word, replace line break by space
    """
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)  # punctuation processed as word
    str = re.sub('([ \n])+', " ", str).strip()  # keep only one space character
    return str.split(" ")

@torch.no_grad()
def keep_all_but_tokens(str, tokens):
    """
    Remove all layout tokens from string
    """
    return re.sub('([' + tokens + '])', '', str)

@torch.no_grad()
def edit_wer_from_formatted_split_text(gt, pred):
    """
    Compute edit distance at word level from formatted string as list
    """
    return editdistance.eval(gt, pred)

@torch.no_grad()
def convert_output_to_pred(outputs,charset, new_pred_logits, direction="horizontal"):

    # pred =  new_pred_logits.max(-1)[1]
    # preds = []
    # preds_labels = []
    # for i in range(pred.shape[0]):
    #     mask = pred[i] !=0
    #     pred_seq = pred[i][mask]
     
    #     preds.append([charset[i-1] for i in pred_seq])
    #     preds_labels.append([ i-1 for i in pred_seq])
    pred = new_pred_logits.argmax(-1)
    preds = []
    preds_labels = []
    max_token = len(charset)
    for i in range(pred.shape[0]):
        ctc_seq = remove_duplicates(pred[i].tolist())
        valid_tokens = [token for token in ctc_seq if 1 <= token <= max_token]
        preds.append([charset[token - 1] for token in valid_tokens])
        preds_labels.append([token - 1 for token in valid_tokens])
    return preds,preds_labels
@torch.no_grad()
def remove_duplicates(sequence):
    processed_output = []
    prev_char = None

    for char in sequence:
        if char != prev_char and char !=0:
            processed_output.append(char)

        prev_char = char

    return processed_output
@torch.no_grad()
def compute_wer(
    outputs,
    targets,
    charset,
    preds,
    return_preds=False,
    duplicate=False,
    mode_chr=False,
    decode_by_pred_direction=False,
    preds_already_sorted=True,
):
    N_batch = outputs["pred_logits"].shape[0]
    wer = 0
    cer = 0
    max_token = len(charset)

    if not duplicate:
        predicted_str_total,predicted_labels_total = [], []
        pred_logits = preds if torch.is_tensor(preds) else torch.stack(preds, dim=0)
        if preds_already_sorted:
            sorted_by_pred_logits = pred_logits
        else:
            directions = _get_decode_directions(
                outputs, targets, decode_by_pred_direction=decode_by_pred_direction
            )
            sorted_by_pred_logits = _sort_pred_logits_by_direction(
                pred_logits, outputs["pred_boxes"], directions
            )
        pred = sorted_by_pred_logits.argmax(-1)
        for i in range(N_batch):
            ctc_seq = remove_duplicates(pred[i].tolist())
            valid_tokens = [token for token in ctc_seq if 1 <= token <= max_token]
            predicted_labels_total.append([token - 1 for token in valid_tokens])
            predicted_str_total.append([charset[token - 1] for token in valid_tokens])
    else:
        preds = preds.argmax(-1)
        predicted_str_total = []
    for i in range(N_batch):
        target_str_list = [charset[int(item)] for item in targets[i]['labels']]
        if duplicate:
            pred_i = preds[i]
            shifted_pred_i  = torch.roll(pred_i, shifts=1)
            mask = pred_i[1:] != shifted_pred_i[1:]
            pred_i = torch.cat([pred_i[0].unsqueeze(0),pred_i[1:][mask]],dim = 0)
            mask = pred_i !=0
            labels_predicted_i = (pred_i[mask]-1).cpu().numpy()
            cer += character_error_rate(labels_predicted_i, [int(item) for item in targets[i]['labels']])
            predicted_str = [charset[int(item)] for item in labels_predicted_i]
            predicted_str_total.append(predicted_str)

        else:
            cer += character_error_rate(predicted_labels_total[i], [int(item) for item in targets[i]['labels']])
            predicted_str = predicted_str_total[i]
        if  mode_chr:
            target_str = "".join(target_str_list)
        else:
            target_str = [chr(int(item)) for item in target_str_list]
            target_str = "".join(target_str)

        target_str = target_str.replace('¬','')
        if  mode_chr:
            predicted_str = "".join(predicted_str)
        else:
            predicted_str = [chr(int(item)) for item in predicted_str]
            predicted_str = "".join(predicted_str)
        predicted_str = predicted_str.replace('¬','')

        split_gt = [format_string_for_wer(target_str)]
        split_pred = [format_string_for_wer(predicted_str)]
        edit_words = [edit_wer_from_formatted_split_text(gt, pred) for (gt, pred) in zip(split_gt, split_pred)]
        nb_words = [len(gt) for gt in split_gt]
        wer += np.sum(np.array(edit_words) / np.array(nb_words))
    
    wer, cer = wer, cer 
    if return_preds:
        return wer, cer, predicted_str_total
    else:
        return wer, cer
@torch.no_grad()
def character_error_rate(predicted_str, gt_str):
    """
    Compute the Character Error Rate (CER) between predicted and ground truth strings.
    
    Args:
    predicted_str (str): The predicted string.
    gt_str (str): The ground truth string.
    
    Returns:
    cer (float): The Character Error Rate.
    """
    # Define function to calculate Levenshtein distance
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    # Calculate Levenshtein distance between strings
    distance = levenshtein_distance(predicted_str, gt_str)

    # CER is the Levenshtein distance divided by the length of the ground truth string
    cer = distance / max(len(gt_str), 1)
    if len(gt_str) == 0 or len(predicted_str) ==0:
        cer = 1
    return cer
