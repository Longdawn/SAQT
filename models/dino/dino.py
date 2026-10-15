# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple

from torchvision.ops.boxes import nms
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util import box_ops
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm, dice_loss
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn, dn_post_process


def apply_activation_gating(
    logits,
    activation_logits,
    use_activation_gating=False,
    activation_gate_blank_coef=0.0,
    activation_gate_nonblank_coef=0.0,
):
    if not use_activation_gating or activation_logits is None:
        return logits
    activation = torch.sigmoid(activation_logits)
    gated = logits.clone()
    gated[:, :, :1] = gated[:, :, :1] + (1.0 - activation) * activation_gate_blank_coef
    gated[:, :, 1:] = gated[:, :, 1:] + activation * activation_gate_nonblank_coef
    return gated


def apply_length_adaptive_ctc_bias(
    probabilities,
    predicted_log_length,
    bias_scale=0.0,
    max_bias=1.0,
):
    """Adjust blank/nonblank log odds from a sample-level predicted length."""
    if predicted_log_length is None or float(bias_scale) == 0.0:
        zero = probabilities.new_zeros(probabilities.shape[0])
        return probabilities, zero
    if probabilities.dim() != 3 or probabilities.shape[-1] < 2:
        raise ValueError("probabilities must have shape [B, Q, C] with C >= 2")
    if float(bias_scale) < 0:
        raise ValueError("bias_scale must be non-negative")
    if float(max_bias) < 0:
        raise ValueError("max_bias must be non-negative")

    predicted_log_length = predicted_log_length.reshape(-1).to(probabilities)
    if predicted_log_length.shape[0] != probabilities.shape[0]:
        raise ValueError("predicted_log_length batch size does not match probabilities")

    expected_nonblank = probabilities[:, :, 1:].sum(dim=-1).sum(dim=-1)
    expected_log_length = torch.log1p(expected_nonblank.detach().clamp(min=0.0))
    adaptive_bias = float(bias_scale) * (predicted_log_length - expected_log_length)
    adaptive_bias = adaptive_bias.clamp(min=-float(max_bias), max=float(max_bias))

    log_probs = torch.log(probabilities.clamp(min=1e-8))
    adjusted = log_probs.clone()
    adjusted[:, :, :1] = adjusted[:, :, :1] - adaptive_bias[:, None, None]
    adjusted[:, :, 1:] = adjusted[:, :, 1:] + adaptive_bias[:, None, None]
    return adjusted.softmax(dim=-1), adaptive_bias


def compute_query_count_loss(activation_logits, target_lengths, short_weight=1.0):
    activation = torch.sigmoid(activation_logits)
    if activation.dim() == 3 and activation.shape[-1] == 1:
        activation = activation.squeeze(-1)
    expected_count = activation.sum(-1)
    target_lengths = target_lengths.to(expected_count.device).float()
    loss = F.smooth_l1_loss(expected_count, target_lengths, reduction="none")
    if short_weight > 1.0:
        weights = torch.ones_like(loss)
        weights[target_lengths <= 2.0] = short_weight
        loss = loss * weights
    return loss.mean(), expected_count


def compute_ctc_collapsed_count_loss(
    probabilities,
    target_lengths,
    short_weight=1.0,
):
    """Match the expected greedy-CTC collapsed length to the transcript length."""
    if probabilities.dim() != 3 or probabilities.shape[-1] < 2:
        raise ValueError("probabilities must have shape [B, Q, C] with C >= 2")
    if probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one query")

    nonblank = probabilities[:, :, 1:]
    expected_count = nonblank[:, 0].sum(dim=-1)
    if nonblank.shape[1] > 1:
        # A nonblank token is emitted when its current class differs from the
        # preceding raw CTC token. This models the unpadded greedy decoder used
        # at inference rather than the blank-interleaved training sequence.
        expected_count = expected_count + (
            nonblank[:, 1:] * (1.0 - nonblank[:, :-1])
        ).sum(dim=(-1, -2))

    target_lengths = target_lengths.to(expected_count).float().clamp(min=1.0)
    if target_lengths.numel() != expected_count.numel():
        raise ValueError("target_lengths batch size does not match probabilities")
    count_ratio = expected_count / target_lengths
    loss = F.smooth_l1_loss(
        count_ratio,
        torch.ones_like(count_ratio),
        reduction="none",
    )
    if short_weight > 1.0:
        weights = torch.ones_like(loss)
        weights[target_lengths <= 2.0] = float(short_weight)
        loss = loss * weights
    return loss.mean(), expected_count


def soft_sort_permutation(scores, temperature):
    """Return the SoftSort row-stochastic relaxation of ascending argsort."""
    if scores.dim() != 2:
        raise ValueError("scores must have shape [B, Q]")
    if scores.shape[1] == 0:
        raise ValueError("scores must contain at least one query")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    sorted_scores = torch.sort(scores, dim=-1).values
    distances = (sorted_scores.unsqueeze(-1) - scores.unsqueeze(-2)).abs()
    return torch.softmax(-distances / float(temperature), dim=-1)


def compute_transcript_guided_soft_order_loss(
    unsorted_ctc_probs,
    pred_boxes,
    targets,
    temperature=0.01,
):
    """Use transcript CTC as a soft-sort surrogate for query box ordering."""
    if unsorted_ctc_probs.dim() != 3 or unsorted_ctc_probs.shape[-1] < 2:
        raise ValueError("unsorted_ctc_probs must have shape [B, Q, C] with C >= 2")
    if pred_boxes.shape[:2] != unsorted_ctc_probs.shape[:2]:
        raise ValueError("pred_boxes and unsorted_ctc_probs must share [B, Q]")
    if len(targets) != unsorted_ctc_probs.shape[0]:
        raise ValueError("targets batch size does not match probabilities")

    losses = []
    entropies = []
    for batch_idx, target in enumerate(targets):
        target_labels = target["labels"].long()
        if target_labels.numel() == 0:
            continue
        direction = target.get("direction")
        if torch.is_tensor(direction):
            direction = int(direction.item())
        sort_axis = 1 if direction is not None and int(direction) == 1 else 0

        permutation = soft_sort_permutation(
            pred_boxes[batch_idx : batch_idx + 1, :, sort_axis],
            temperature,
        )[0]
        unique_labels = torch.unique(target_labels, sorted=True)
        selected_indices = torch.cat(
            (
                unique_labels.new_zeros(1),
                unique_labels + 1,
            )
        )
        selected_probs = unsorted_ctc_probs[batch_idx].detach().index_select(
            -1, selected_indices
        )
        other_prob = (1.0 - selected_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        local_probs = torch.cat((selected_probs, other_prob), dim=-1)
        soft_sorted_probs = permutation.matmul(local_probs).clamp(min=1e-8)
        soft_sorted_probs = soft_sorted_probs / soft_sorted_probs.sum(
            dim=-1, keepdim=True
        )

        padded_probs = soft_sorted_probs.new_zeros(
            soft_sorted_probs.shape[0] * 2,
            soft_sorted_probs.shape[1],
        )
        padded_probs[::2] = soft_sorted_probs
        padded_probs[1::2, 0] = 1.0

        local_targets = (
            (target_labels[:, None] == unique_labels[None, :])
            .to(torch.int64)
            .argmax(dim=-1)
            + 1
        )
        input_lengths = torch.tensor(
            [padded_probs.shape[0]], device=padded_probs.device, dtype=torch.long
        )
        target_lengths = torch.tensor(
            [local_targets.numel()], device=padded_probs.device, dtype=torch.long
        )
        losses.append(
            F.ctc_loss(
                padded_probs.log().unsqueeze(1),
                local_targets.unsqueeze(0),
                input_lengths,
                target_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )
        )
        entropies.append(
            -(permutation * permutation.clamp(min=1e-8).log()).sum(-1).mean()
        )

    if not losses:
        zero = pred_boxes.sum() * 0.0
        return zero, {"permutation_entropy": zero.detach()}
    return torch.stack(losses).mean(), {
        "permutation_entropy": torch.stack(entropies).mean().detach()
    }


def compute_query_budget_loss(
    pred_logits,
    targets,
    budget_scale=2.0,
    budget_margin=8.0,
):
    nonblank_probs = pred_logits.sigmoid().amax(dim=-1)
    expected_count = nonblank_probs.sum(dim=-1)
    target_lengths = torch.tensor(
        [float(target["labels"].numel()) for target in targets],
        dtype=pred_logits.dtype,
        device=pred_logits.device,
    )
    target_count = float(budget_scale) * target_lengths + float(budget_margin)
    normalizer = target_lengths.clamp(min=1.0)
    over_budget = (expected_count - target_count).clamp(min=0.0)
    loss = ((over_budget**2) / normalizer).mean()
    return loss, {
        "expected_count_mean": float(expected_count.detach().mean().item()),
        "target_count_mean": float(target_count.detach().mean().item()),
    }


def fuse_glyph_prototype_logits(pred_logits, pred_proto_logits, glyph_proto_fuse_coef=0.0):
    if pred_proto_logits is None or float(glyph_proto_fuse_coef) == 0.0:
        return pred_logits
    return pred_logits + float(glyph_proto_fuse_coef) * pred_proto_logits


class QuerySequenceRefiner(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, kernel_size=3, dropout=0.0):
        super().__init__()
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        kernel_size = int(kernel_size)
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_dim),
                        "dwconv": nn.Conv1d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            groups=hidden_dim,
                        ),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.GELU(),
                            nn.Dropout(float(dropout)),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                            nn.Dropout(float(dropout)),
                        ),
                    }
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, query_feats):
        refined = query_feats
        for layer in self.layers:
            residual = refined
            x = layer["norm"](refined)
            x = layer["dwconv"](x.transpose(1, 2)).transpose(1, 2)
            x = layer["ffn"](x)
            refined = residual + x
        return refined


class VerticalGeometryQueryAdapter(nn.Module):
    """Inject sorted-query geometry through a zero-start residual branch."""

    def __init__(self, hidden_dim, geometry_hidden_dim=64, init_scale=0.0):
        super().__init__()
        hidden_dim = int(hidden_dim)
        geometry_hidden_dim = int(geometry_hidden_dim)
        if geometry_hidden_dim < 1:
            raise ValueError("geometry_hidden_dim must be >= 1")
        self.geometry_mlp = nn.Sequential(
            nn.Linear(4, geometry_hidden_dim),
            nn.GELU(),
            nn.Linear(geometry_hidden_dim, hidden_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    @staticmethod
    def geometry_features(sorted_boxes, axis=1):
        axis = int(axis)
        if axis not in (0, 1):
            raise ValueError("axis must be 0 (horizontal) or 1 (vertical)")
        center = sorted_boxes[:, :, axis]
        extent = sorted_boxes[:, :, axis + 2].clamp(min=1e-4)

        prev_gap = torch.zeros_like(center)
        next_gap = torch.zeros_like(center)
        if center.shape[1] > 1:
            pair_gap = center[:, 1:] - center[:, :-1]
            pair_extent = 0.5 * (extent[:, 1:] + extent[:, :-1])
            normalized_gap = (pair_gap / pair_extent.clamp(min=1e-4)).clamp(0.0, 4.0)
            prev_gap[:, 1:] = normalized_gap
            next_gap[:, :-1] = normalized_gap

        return torch.stack((center, extent, prev_gap, next_gap), dim=-1)

    def forward(self, sorted_query_feats, sorted_boxes, axis=1):
        geometry = self.geometry_features(sorted_boxes, axis=axis)
        geometry_delta = self.geometry_mlp(geometry)
        scale = torch.tanh(self.residual_scale)
        return sorted_query_feats + scale * geometry_delta


def compute_short_query_ce_loss(char_probs, targets, max_len=0, gamma=0.0):
    ce_losses = []
    eligible_samples = 0
    selected_chars = 0
    device = char_probs.device

    for batch_idx, target in enumerate(targets):
        labels = target["labels"].long().to(device)
        if labels.numel() == 0:
            continue
        if int(max_len) > 0 and labels.numel() > int(max_len):
            continue

        eligible_samples += 1
        used_queries = torch.zeros(
            char_probs.shape[1],
            dtype=torch.bool,
            device=device,
        )
        for label in labels:
            if label.item() < 0 or label.item() >= char_probs.shape[-1]:
                continue
            scores = char_probs[batch_idx, :, label].detach().clone()
            if used_queries.all():
                used_queries.zero_()
            scores[used_queries] = -1.0
            query_idx = scores.argmax()
            used_queries[query_idx] = True

            prob = char_probs[batch_idx, query_idx, label].clamp(min=1e-8)
            nll = -torch.log(prob)
            if float(gamma) > 0:
                nll = ((1.0 - prob).detach() ** float(gamma)) * nll
            ce_losses.append(nll)
            selected_chars += 1

    if ce_losses:
        loss = torch.stack(ce_losses).mean()
    else:
        loss = char_probs.sum() * 0.0

    return loss, {
        "eligible_samples": eligible_samples,
        "selected_chars": selected_chars,
    }


def compute_glyph_proto_aux_loss(
    pred_proto_logits,
    char_probs,
    targets,
    max_len=0,
    gamma=0.0,
):
    ce_losses = []
    eligible_samples = 0
    selected_chars = 0
    device = pred_proto_logits.device

    for batch_idx, target in enumerate(targets):
        labels = target["labels"].long().to(device)
        if labels.numel() == 0:
            continue
        if int(max_len) > 0 and labels.numel() > int(max_len):
            continue

        eligible_samples += 1
        used_queries = torch.zeros(
            pred_proto_logits.shape[1],
            dtype=torch.bool,
            device=device,
        )
        for label in labels:
            if label.item() < 0 or label.item() >= pred_proto_logits.shape[-1]:
                continue
            scores = char_probs[batch_idx, :, label].detach().clone()
            if used_queries.all():
                used_queries.zero_()
            scores[used_queries] = -1.0
            query_idx = scores.argmax()
            used_queries[query_idx] = True

            proto_logits = pred_proto_logits[batch_idx, query_idx].unsqueeze(0)
            label_tensor = label.view(1)
            nll = F.cross_entropy(proto_logits, label_tensor, reduction="none").squeeze(0)
            if float(gamma) > 0:
                prob = proto_logits.softmax(-1)[0, label].detach()
                nll = ((1.0 - prob) ** float(gamma)) * nll
            ce_losses.append(nll)
            selected_chars += 1

    if ce_losses:
        loss = torch.stack(ce_losses).mean()
    else:
        loss = pred_proto_logits.sum() * 0.0

    return loss, {
        "eligible_samples": eligible_samples,
        "selected_chars": selected_chars,
    }


def _ctc_viterbi_path(log_probs, labels, blank=0):
    labels = [int(label) for label in labels]
    if len(labels) == 0:
        return torch.full(
            (log_probs.shape[0],),
            int(blank),
            dtype=torch.long,
            device=log_probs.device,
        )

    extended = [int(blank)]
    for label in labels:
        extended.extend([label, int(blank)])
    ext = torch.tensor(extended, dtype=torch.long, device=log_probs.device)

    with torch.no_grad():
        scores = log_probs.detach()
        time_steps = scores.shape[0]
        state_count = ext.numel()
        neg_inf = torch.finfo(scores.dtype).min
        dp = scores.new_full((time_steps, state_count), neg_inf)
        backptr = torch.zeros(
            (time_steps, state_count),
            dtype=torch.long,
            device=scores.device,
        )

        dp[0, 0] = scores[0, ext[0]]
        if state_count > 1:
            dp[0, 1] = scores[0, ext[1]]

        for t in range(1, time_steps):
            for s in range(state_count):
                prev_states = [s]
                if s - 1 >= 0:
                    prev_states.append(s - 1)
                if (
                    s - 2 >= 0
                    and int(ext[s].item()) != int(blank)
                    and int(ext[s].item()) != int(ext[s - 2].item())
                ):
                    prev_states.append(s - 2)

                prev = torch.stack([dp[t - 1, prev_s] for prev_s in prev_states])
                best_offset = int(prev.argmax().item())
                best_state = prev_states[best_offset]
                dp[t, s] = prev[best_offset] + scores[t, ext[s]]
                backptr[t, s] = best_state

        if state_count == 1:
            state = 0
        else:
            final_candidates = torch.stack([dp[-1, state_count - 1], dp[-1, state_count - 2]])
            state = state_count - 1 if int(final_candidates.argmax().item()) == 0 else state_count - 2

        path_states = []
        for t in range(time_steps - 1, -1, -1):
            path_states.append(state)
            state = int(backptr[t, state].item())
        path_states.reverse()
        path_states = torch.tensor(path_states, dtype=torch.long, device=log_probs.device)
        return ext[path_states]


def compute_ctc_viterbi_alignment_loss(
    log_probs,
    targets_tensor,
    target_lengths,
    blank=0,
    blank_weight=0.0,
    nonblank_weight=1.0,
    max_target_len=0,
):
    if log_probs.dim() != 3:
        raise ValueError("log_probs must have shape [T, B, C]")

    losses = []
    selected_samples = 0
    time_steps = log_probs.shape[0]
    frame_index = torch.arange(time_steps, device=log_probs.device)

    for batch_idx in range(log_probs.shape[1]):
        target_len = int(target_lengths[batch_idx].item())
        if target_len <= 0:
            continue
        if int(max_target_len) > 0 and target_len > int(max_target_len):
            continue

        labels = targets_tensor[batch_idx, :target_len].long().tolist()
        path = _ctc_viterbi_path(log_probs[:, batch_idx, :], labels, blank=blank)
        nll = -log_probs[:, batch_idx, :][frame_index, path]
        weights = torch.full_like(nll, float(blank_weight))
        weights[path != int(blank)] = float(nonblank_weight)
        normalizer = weights.sum().clamp(min=1.0)
        losses.append((nll * weights).sum() / normalizer)
        selected_samples += 1

    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = log_probs.sum() * 0.0

    return loss, {
        "selected_samples": selected_samples,
    }


class DINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        random_refpoints_xy=False,
        fix_refpoints_hw=-1,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        two_stage_add_query_num=0,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        decoder_sa_type="sa",
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        use_direction_head=False,
        use_query_activation_head=False,
        query_activation_init_bias=0.0,
        use_glyph_prototype_head=False,
        glyph_proto_dim=256,
        glyph_proto_temperature=1.0,
        glyph_proto_trainable=True,
        use_query_sequence_refiner=False,
        query_sequence_refiner_layers=1,
        query_sequence_refiner_kernel=3,
        query_sequence_refiner_dropout=0.0,
        use_sorted_ctc_refiner=False,
        sorted_ctc_refiner_layers=1,
        sorted_ctc_refiner_kernel=3,
        sorted_ctc_refiner_dropout=0.0,
        sorted_ctc_refiner_axis=1,
        use_vertical_geometry_adapter=False,
        vertical_geometry_adapter_hidden_dim=64,
        vertical_geometry_adapter_axis=1,
        vertical_geometry_adapter_init_scale=0.0,
        use_ctc_length_calibration=False,
        ctc_length_init=8.0,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.use_direction_head = use_direction_head
        self.use_query_activation_head = use_query_activation_head
        self.query_activation_init_bias = float(query_activation_init_bias)
        self.use_glyph_prototype_head = bool(use_glyph_prototype_head)
        self.glyph_proto_dim = int(glyph_proto_dim)
        self.glyph_proto_temperature = float(glyph_proto_temperature)
        self.glyph_proto_trainable = bool(glyph_proto_trainable)
        self.use_query_sequence_refiner = bool(use_query_sequence_refiner)
        self.query_sequence_refiner_layers = int(query_sequence_refiner_layers)
        self.query_sequence_refiner_kernel = int(query_sequence_refiner_kernel)
        self.query_sequence_refiner_dropout = float(query_sequence_refiner_dropout)
        self.use_sorted_ctc_refiner = bool(use_sorted_ctc_refiner)
        self.sorted_ctc_refiner_layers = int(sorted_ctc_refiner_layers)
        self.sorted_ctc_refiner_kernel = int(sorted_ctc_refiner_kernel)
        self.sorted_ctc_refiner_dropout = float(sorted_ctc_refiner_dropout)
        self.sorted_ctc_refiner_axis = int(sorted_ctc_refiner_axis)
        self.use_vertical_geometry_adapter = bool(use_vertical_geometry_adapter)
        self.vertical_geometry_adapter_hidden_dim = int(
            vertical_geometry_adapter_hidden_dim
        )
        self.vertical_geometry_adapter_axis = int(vertical_geometry_adapter_axis)
        self.vertical_geometry_adapter_init_scale = float(
            vertical_geometry_adapter_init_scale
        )
        self.use_ctc_length_calibration = bool(use_ctc_length_calibration)
        self.ctc_length_init = float(ctc_length_init)
        if self.ctc_length_init < 0:
            raise ValueError("ctc_length_init must be non-negative")

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert (
                two_stage_type == "no"
            ), "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [
                _class_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]
        if decoder_sa_type == "ca_label":
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()
        if self.use_direction_head:
            self.direction_embed = nn.Linear(hidden_dim, 2)
        if self.use_query_activation_head:
            self.query_activation_embed = nn.Linear(hidden_dim, 1)
            nn.init.constant_(
                self.query_activation_embed.bias.data,
                self.query_activation_init_bias,
            )
        if self.use_glyph_prototype_head:
            # Isolate prototype-branch random initialization so enabling this
            # branch does not perturb later baseline-sensitive reinitialization
            # paths such as `--new_class_embedding`.
            torch_rng_state = torch.get_rng_state()
            self.glyph_proto_proj = nn.Linear(hidden_dim, self.glyph_proto_dim)
            self.glyph_prototype_table = nn.Embedding(num_classes, self.glyph_proto_dim)
            nn.init.normal_(self.glyph_prototype_table.weight, std=0.02)
            torch.set_rng_state(torch_rng_state)
            if not self.glyph_proto_trainable:
                self.glyph_prototype_table.weight.requires_grad_(False)
        if self.use_query_sequence_refiner:
            torch_rng_state = torch.get_rng_state()
            self.query_sequence_refiner = QuerySequenceRefiner(
                hidden_dim,
                num_layers=self.query_sequence_refiner_layers,
                kernel_size=self.query_sequence_refiner_kernel,
                dropout=self.query_sequence_refiner_dropout,
            )
            torch.set_rng_state(torch_rng_state)
        if self.use_sorted_ctc_refiner:
            torch_rng_state = torch.get_rng_state()
            self.sorted_ctc_refiner = QuerySequenceRefiner(
                hidden_dim,
                num_layers=self.sorted_ctc_refiner_layers,
                kernel_size=self.sorted_ctc_refiner_kernel,
                dropout=self.sorted_ctc_refiner_dropout,
            )
            torch.set_rng_state(torch_rng_state)
        if self.use_vertical_geometry_adapter:
            torch_rng_state = torch.get_rng_state()
            self.vertical_geometry_adapter = VerticalGeometryQueryAdapter(
                hidden_dim,
                geometry_hidden_dim=self.vertical_geometry_adapter_hidden_dim,
                init_scale=self.vertical_geometry_adapter_init_scale,
            )
            torch.set_rng_state(torch_rng_state)
        if self.use_ctc_length_calibration:
            torch_rng_state = torch.get_rng_state()
            self.ctc_length_embed = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.ctc_length_embed[-1].weight)
            nn.init.constant_(
                self.ctc_length_embed[-1].bias,
                math.log1p(self.ctc_length_init),
            )
            torch.set_rng_state(torch_rng_state)

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, 2:]
            )
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print("learn a shared h and w")
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError(
                "Unknown fix_refpoints_hw {}".format(self.fix_refpoints_hw)
            )

    def forward(self, samples: NestedTensor, targets: List = None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.dn_number > 0 and targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(
                    targets,
                    self.dn_number,
                    self.dn_label_noise_ratio,
                    self.dn_box_noise_scale,
                ),
                training=self.training,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                label_enc=self.label_enc,
            )
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask
        )
        # In case num object=0
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        final_query_hs = hs[-1]
        if self.use_query_sequence_refiner:
            final_query_hs = self.query_sequence_refiner(final_query_hs)

        outputs_class_list = []
        for dec_lid, (layer_cls_embed, layer_hs) in enumerate(zip(self.class_embed, hs)):
            if dec_lid == len(hs) - 1:
                layer_hs = final_query_hs
            outputs_class_list.append(layer_cls_embed(layer_hs))
        outputs_class = torch.stack(outputs_class_list)
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = dn_post_process(
                outputs_class,
                outputs_coord_list,
                dn_meta,
                self.aux_loss,
                self._set_aux_loss,
            )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
        if (
            (self.use_sorted_ctc_refiner or self.use_vertical_geometry_adapter)
            and dn_meta is None
            and hs[-1].shape[:2] == outputs_coord_list[-1].shape[:2]
        ):
            if self.use_vertical_geometry_adapter:
                sort_axis = 1 if self.vertical_geometry_adapter_axis == 1 else 0
            else:
                sort_axis = 1 if self.sorted_ctc_refiner_axis == 1 else 0
            _, sorted_idx = torch.sort(
                outputs_coord_list[-1][:, :, sort_axis],
                dim=1,
                descending=False,
            )
            sorted_query_hs = torch.gather(
                hs[-1],
                1,
                sorted_idx.unsqueeze(-1).expand(-1, -1, hs[-1].shape[-1]),
            )
            sorted_boxes = torch.gather(
                outputs_coord_list[-1],
                1,
                sorted_idx.unsqueeze(-1).expand(-1, -1, 4),
            )
            if self.use_vertical_geometry_adapter:
                sorted_query_hs = self.vertical_geometry_adapter(
                    sorted_query_hs,
                    sorted_boxes,
                    axis=sort_axis,
                )
            if self.use_sorted_ctc_refiner:
                refined_sorted_query_hs = self.sorted_ctc_refiner(sorted_query_hs)
            else:
                refined_sorted_query_hs = sorted_query_hs
            out["pred_sorted_ctc_logits"] = self.class_embed[-1](refined_sorted_query_hs)
        if self.use_query_activation_head:
            out["pred_activation_logits"] = self.query_activation_embed(final_query_hs)
        if self.use_glyph_prototype_head:
            proto_query = F.normalize(self.glyph_proto_proj(final_query_hs), dim=-1)
            proto_table = self.glyph_prototype_table.weight
            current_num_classes = outputs_class[-1].shape[-1]
            if proto_table.shape[0] != current_num_classes:
                if proto_table.shape[0] < current_num_classes:
                    raise RuntimeError(
                        f"glyph prototype table has {proto_table.shape[0]} rows but "
                        f"class head outputs {current_num_classes} classes"
                    )
                proto_table = proto_table[:current_num_classes]
            proto_table = F.normalize(proto_table, dim=-1)
            proto_logits = torch.matmul(proto_query, proto_table.transpose(0, 1))
            proto_logits = proto_logits / max(self.glyph_proto_temperature, 1e-6)
            out["pred_proto_logits"] = proto_logits
        if self.use_direction_head:
            out["pred_direction"] = self.direction_embed(final_query_hs.mean(dim=1))
        if self.use_ctc_length_calibration:
            pooled_queries = torch.cat(
                (final_query_hs.mean(dim=1), final_query_hs.amax(dim=1)),
                dim=-1,
            )
            out["pred_log_length"] = self.ctc_length_embed(pooled_queries).squeeze(-1)
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out["interm_outputs"] = {
                "pred_logits": interm_class,
                "pred_boxes": interm_coord,
            }
            out["interm_outputs_for_matching_pre"] = {
                "pred_logits": interm_class,
                "pred_boxes": init_box_proposal,
            }

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (
                    layer_box_embed,
                    layer_class_embed,
                    layer_hs_enc,
                    layer_ref_enc,
                ) in enumerate(
                    zip(
                        self.enc_bbox_embed,
                        self.enc_class_embed,
                        hs_enc[:-1],
                        ref_enc[:-1],
                    )
                ):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = (
                        layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    )
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out["enc_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    """This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        focal_alpha,
        losses,
        CTC=False,
        ctc_blank_max=1.0,
        ctc_count_loss_short_weight=1.0,
        ctc_collapsed_count_loss_short_weight=1.0,
        ctc_soft_order_temperature=0.01,
        short_gt_presence_max_len=0,
        short_gt_presence_margin=0.0,
        short_gt_ce_max_len=0,
        short_gt_ce_gamma=0.0,
        use_query_activation_head=False,
        use_query_count_loss=False,
        use_activation_gating=False,
        use_glyph_prototype_head=False,
        glyph_proto_fuse_coef=0.0,
        glyph_proto_aux_max_len=0,
        glyph_proto_aux_gamma=0.0,
        activation_gate_blank_coef=0.0,
        activation_gate_nonblank_coef=0.0,
        query_count_short_weight=1.0,
        query_budget_scale=2.0,
        query_budget_margin=8.0,
        use_sorted_ctc_refiner=False,
        ctc_viterbi_loss_blank_weight=0.0,
        ctc_viterbi_loss_nonblank_weight=1.0,
        ctc_viterbi_loss_max_len=0,
        use_ctc_length_calibration=False,
        ctc_length_bias_scale=0.0,
        ctc_length_bias_max=1.0,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.CTC = CTC
        self.ctc_blank_max = float(ctc_blank_max)
        self.ctc_count_loss_short_weight = float(ctc_count_loss_short_weight)
        self.ctc_collapsed_count_loss_short_weight = float(
            ctc_collapsed_count_loss_short_weight
        )
        self.ctc_soft_order_temperature = float(ctc_soft_order_temperature)
        if self.ctc_soft_order_temperature <= 0.0:
            raise ValueError("ctc_soft_order_temperature must be positive")
        self.short_gt_presence_max_len = int(short_gt_presence_max_len)
        self.short_gt_presence_margin = float(short_gt_presence_margin)
        self.short_gt_ce_max_len = int(short_gt_ce_max_len)
        self.short_gt_ce_gamma = float(short_gt_ce_gamma)
        self.use_query_activation_head = bool(use_query_activation_head)
        self.use_query_count_loss = bool(use_query_count_loss)
        self.use_activation_gating = bool(use_activation_gating)
        self.use_glyph_prototype_head = bool(use_glyph_prototype_head)
        self.glyph_proto_fuse_coef = float(glyph_proto_fuse_coef)
        self.glyph_proto_aux_max_len = int(glyph_proto_aux_max_len)
        self.glyph_proto_aux_gamma = float(glyph_proto_aux_gamma)
        self.activation_gate_blank_coef = float(activation_gate_blank_coef)
        self.activation_gate_nonblank_coef = float(activation_gate_nonblank_coef)
        self.query_count_short_weight = float(query_count_short_weight)
        self.query_budget_scale = float(query_budget_scale)
        self.query_budget_margin = float(query_budget_margin)
        self.use_sorted_ctc_refiner = bool(use_sorted_ctc_refiner)
        self.ctc_viterbi_loss_blank_weight = float(ctc_viterbi_loss_blank_weight)
        self.ctc_viterbi_loss_nonblank_weight = float(ctc_viterbi_loss_nonblank_weight)
        self.ctc_viterbi_loss_max_len = int(ctc_viterbi_loss_max_len)
        self.use_ctc_length_calibration = bool(use_ctc_length_calibration)
        self.ctc_length_bias_scale = float(ctc_length_bias_scale)
        self.ctc_length_bias_max = float(ctc_length_bias_max)
        if self.ctc_length_bias_max < 0:
            raise ValueError("ctc_length_bias_max must be non-negative")
        self.direction_loss_enabled = "loss_direction" in self.weight_dict
        self.ctc_count_loss_enabled = "loss_ctc_count" in self.weight_dict
        self.ctc_collapsed_count_loss_enabled = (
            "loss_ctc_collapsed_count" in self.weight_dict
        )
        self.ctc_soft_order_loss_enabled = "loss_ctc_soft_order" in self.weight_dict
        self.ctc_viterbi_loss_enabled = "loss_ctc_viterbi" in self.weight_dict
        self.ctc_length_loss_enabled = "loss_ctc_length" in self.weight_dict
        self.glyph_proto_aux_enabled = "loss_glyph_proto_aux" in self.weight_dict
        self.short_gt_presence_enabled = "loss_short_gt_presence" in self.weight_dict
        self.short_gt_ce_enabled = "loss_short_gt_ce" in self.weight_dict
        self.query_budget_loss_enabled = "loss_query_budget" in self.weight_dict
        self.query_count_loss_enabled = (
            self.use_query_count_loss and "loss_query_count" in self.weight_dict
        )
        if self.CTC:
            self.losses_all = copy.deepcopy(self.losses)
            self.losses_CTC = ["loss_CTC"]

    def _extract_direction_target(self, target_item):
        if not isinstance(target_item, dict) or "direction" not in target_item:
            return None
        direction = target_item["direction"]
        if torch.is_tensor(direction):
            direction = int(direction.item())
        return 1 if int(direction) == 1 else 0

    def loss_direction(self, outputs, targets, indices, num_boxes, log=True):
        if "pred_direction" not in outputs:
            zero = outputs["pred_logits"].sum() * 0.0
            return {"loss_direction": zero, "direction_acc": zero}

        valid_indices = []
        direction_targets = []
        for idx, target in enumerate(targets):
            direction = self._extract_direction_target(target)
            if direction is None:
                continue
            valid_indices.append(idx)
            direction_targets.append(direction)

        if len(valid_indices) == 0:
            zero = outputs["pred_direction"].sum() * 0.0
            return {"loss_direction": zero, "direction_acc": zero}

        valid_indices = torch.as_tensor(
            valid_indices, device=outputs["pred_direction"].device, dtype=torch.long
        )
        direction_targets = torch.as_tensor(
            direction_targets, device=outputs["pred_direction"].device, dtype=torch.long
        )
        pred_direction = outputs["pred_direction"].index_select(0, valid_indices)
        loss_direction = F.cross_entropy(pred_direction, direction_targets)
        direction_acc = (pred_direction.argmax(-1) == direction_targets).float().mean()
        return {"loss_direction": loss_direction, "direction_acc": direction_acc}

    def loss_CTC(
        self,
        outputs,
        targets,
        indices,
        num_boxes,
        log=True,
        return_preds=False
    ):
        def _get_direction(target_item):
            if isinstance(target_item, dict) and "direction" in target_item:
                direction = target_item["direction"]
                if torch.is_tensor(direction):
                    direction = int(direction.item())
                else:
                    direction = int(direction)
                return "vertical" if direction == 1 else "horizontal"
            return "horizontal"

        pred_logits = outputs["pred_logits"]
        pred_proto_logits = outputs.get("pred_proto_logits")
        pred_logits = fuse_glyph_prototype_logits(
            pred_logits,
            pred_proto_logits,
            glyph_proto_fuse_coef=self.glyph_proto_fuse_coef,
        )
        device = pred_logits.device
        
        pred_logits_topk = pred_logits
        pred_boxes_topk = outputs["pred_boxes"]

        # Sort each sample with its own reading direction.
        idx_list = []
        for i in range(pred_boxes_topk.shape[0]):
            direction = _get_direction(targets[i])
            sort_axis = 1 if direction == "vertical" else 0
            __, idx_i = torch.sort(pred_boxes_topk[i, :, sort_axis], descending=False)
            idx_list.append(idx_i)
        idx = torch.stack(idx_list, dim=0)

        if "pred_sorted_ctc_logits" in outputs:
            sorted_by_x_pred_logits = outputs["pred_sorted_ctc_logits"]
        else:
            sorted_by_x_pred_logits = torch.gather(
                pred_logits_topk,
                1,
                idx.unsqueeze(-1).expand(-1, -1, pred_logits_topk.shape[-1]),
            )
        sorted_by_x_pred_logits = sorted_by_x_pred_logits.sigmoid()

        new_pred_logits = torch.zeros(
            (
                sorted_by_x_pred_logits.shape[0],
                sorted_by_x_pred_logits.shape[1],
                sorted_by_x_pred_logits.shape[2] + 1,
            )
        ).to(device)
        new_pred_logits[:, :, 1:] = sorted_by_x_pred_logits

        ## computes the proba of the blank token
        eps = 0.003  # / pred_logits.shape[-1] # 0.01/num_classes
        mask = sorted_by_x_pred_logits.sum(-1) < 1 - eps
        new_pred_logits[:, :, 0][mask] = 1 - sorted_by_x_pred_logits[mask].sum(-1)

        mask = ~mask

        new_pred_logits[:, :, 0][mask] = eps
        new_pred_logits[:, :, 1:][mask] = (
            (1 - eps)
            * sorted_by_x_pred_logits[mask]
            / sorted_by_x_pred_logits[mask].sum(-1).unsqueeze(-1)
        )

        if self.ctc_soft_order_loss_enabled:
            unsorted_ctc_probs = torch.zeros_like(new_pred_logits).scatter(
                1,
                idx.unsqueeze(-1).expand_as(new_pred_logits),
                new_pred_logits,
            )
            soft_order_loss, soft_order_stats = (
                compute_transcript_guided_soft_order_loss(
                    unsorted_ctc_probs,
                    pred_boxes_topk,
                    targets,
                    temperature=self.ctc_soft_order_temperature,
                )
            )
        else:
            soft_order_loss = None
            soft_order_stats = None

        # Optional anti-collapse control: cap blank probability so decode is less
        # likely to become empty on vertical long lines.
        if 0.0 < self.ctc_blank_max < 1.0:
            blank_probs = new_pred_logits[:, :, 0]
            over_mask = blank_probs > self.ctc_blank_max
            if over_mask.any():
                nonblank_sum = new_pred_logits[:, :, 1:].sum(-1).clamp(min=1e-6)
                scale = (1.0 - self.ctc_blank_max) / nonblank_sum
                new_pred_logits[:, :, 1:][over_mask] = (
                    new_pred_logits[:, :, 1:][over_mask]
                    * scale[over_mask].unsqueeze(-1)
                )
                new_pred_logits[:, :, 0][over_mask] = self.ctc_blank_max

        activation_logits = outputs.get("pred_activation_logits")
        if self.use_activation_gating and activation_logits is not None:
            sorted_activation_logits = torch.gather(
                activation_logits,
                1,
                idx.unsqueeze(-1),
            )
            gated_log_probs = apply_activation_gating(
                torch.log(new_pred_logits.clamp(min=1e-8)),
                sorted_activation_logits,
                use_activation_gating=True,
                activation_gate_blank_coef=self.activation_gate_blank_coef,
                activation_gate_nonblank_coef=self.activation_gate_nonblank_coef,
            )
            new_pred_logits = gated_log_probs.softmax(-1)

        adaptive_length_bias = new_pred_logits.new_zeros(new_pred_logits.shape[0])
        predicted_log_length = outputs.get("pred_log_length")
        if self.use_ctc_length_calibration and predicted_log_length is not None:
            new_pred_logits, adaptive_length_bias = apply_length_adaptive_ctc_bias(
                new_pred_logits,
                predicted_log_length,
                bias_scale=self.ctc_length_bias_scale,
                max_bias=self.ctc_length_bias_max,
            )
        
        
        blank_tensor = torch.zeros_like(new_pred_logits) + 1e-5
        blank_tensor[:, :, 0] = 1

        # ## add blank tokens
        pred_logits_padded = torch.zeros(
            (
                new_pred_logits.shape[0],
                new_pred_logits.shape[1] * 2,
                new_pred_logits.shape[2],
            )
        ).to(device)  # + 1e-5
        pred_logits_padded[:, ::2, :] = new_pred_logits
        pred_logits_padded[:, 1::2, :] = blank_tensor

        # pred_logits_padded = new_pred_logits
        # pred_logits_padded = pred_logits
        length_pred = torch.full(
            size=(pred_logits_padded.shape[0],),
            fill_value=pred_logits_padded.shape[1],
            dtype=torch.int64,
        )

        with torch.no_grad():
            length_input = torch.zeros_like(length_pred)
            max_length = 0
            for i, target in enumerate(targets):
                len_target = len(target["labels"])
                length_input[i] = len_target
                if len_target > max_length:
                    max_length = len_target
            targets_tensor = torch.zeros(pred_logits.shape[0], max_length)
            for i, target in enumerate(targets):
                targets_tensor[i, : len(target["labels"])] = target["labels"] + 1
        ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
        pred_log_probs = torch.log(pred_logits_padded.permute(1, 0, 2))
        loss = ctc_loss(
            pred_log_probs,
            targets_tensor,
            length_pred,
            length_input,
        )  
    

        blank_pred_ratio = (new_pred_logits.argmax(-1) == 0).float().mean()
        losses = {"loss_CTC": loss, "blank_pred_ratio": blank_pred_ratio.detach()}

        if soft_order_loss is not None:
            losses["loss_ctc_soft_order"] = soft_order_loss
            losses["ctc_soft_order_entropy"] = soft_order_stats[
                "permutation_entropy"
            ]

        if self.ctc_length_loss_enabled and predicted_log_length is not None:
            target_log_length = torch.log1p(
                length_input.to(device=device, dtype=pred_logits.dtype)
            )
            losses["loss_ctc_length"] = F.smooth_l1_loss(
                predicted_log_length.reshape(-1),
                target_log_length,
            )
            losses["ctc_predicted_length"] = torch.expm1(
                predicted_log_length.detach().clamp(min=0.0)
            ).mean()
            losses["ctc_adaptive_bias"] = adaptive_length_bias.detach().mean()

        if self.query_count_loss_enabled and activation_logits is not None:
            query_count_loss, expected_activation_count = compute_query_count_loss(
                activation_logits,
                length_input.float(),
                short_weight=self.query_count_short_weight,
            )
            losses["loss_query_count"] = query_count_loss
            losses["query_expected_count"] = expected_activation_count.detach().mean()
            losses["query_target_count"] = length_input.float().detach().mean()

        if self.ctc_count_loss_enabled:
            target_count = length_input.float().to(device).clamp(min=1.0)
            expected_nonblank_count = new_pred_logits[:, :, 1:].sum(-1).sum(-1)
            count_ratio = expected_nonblank_count / target_count
            count_target = torch.ones_like(count_ratio)
            count_loss = F.smooth_l1_loss(count_ratio, count_target, reduction="none")

            if self.ctc_count_loss_short_weight > 1.0:
                short_mask = target_count <= 2.0
                weights = torch.ones_like(count_loss)
                weights[short_mask] = self.ctc_count_loss_short_weight
                count_loss = count_loss * weights

            losses["loss_ctc_count"] = count_loss.mean()
            losses["ctc_expected_count"] = expected_nonblank_count.detach().mean()
            losses["ctc_target_count"] = target_count.detach().mean()

        if self.ctc_collapsed_count_loss_enabled:
            collapsed_count_loss, expected_collapsed_count = (
                compute_ctc_collapsed_count_loss(
                    new_pred_logits,
                    length_input,
                    short_weight=self.ctc_collapsed_count_loss_short_weight,
                )
            )
            losses["loss_ctc_collapsed_count"] = collapsed_count_loss
            losses["ctc_expected_collapsed_count"] = (
                expected_collapsed_count.detach().mean()
            )
            losses["ctc_collapsed_target_count"] = length_input.float().detach().mean()

        if self.ctc_viterbi_loss_enabled:
            unpadded_log_probs = torch.log(new_pred_logits.permute(1, 0, 2))
            viterbi_loss, viterbi_stats = compute_ctc_viterbi_alignment_loss(
                unpadded_log_probs,
                targets_tensor.to(device),
                length_input.to(device),
                blank=0,
                blank_weight=self.ctc_viterbi_loss_blank_weight,
                nonblank_weight=self.ctc_viterbi_loss_nonblank_weight,
                max_target_len=self.ctc_viterbi_loss_max_len,
            )
            losses["loss_ctc_viterbi"] = viterbi_loss
            losses["ctc_viterbi_selected_samples"] = torch.tensor(
                float(viterbi_stats["selected_samples"]),
                device=device,
            )

        if self.short_gt_presence_enabled:
            presence_losses = []
            log_blank = torch.log(new_pred_logits[:, :, 0].clamp(min=1e-8))
            log_nonblank = torch.log(new_pred_logits[:, :, 1:].clamp(min=1e-8))
            for batch_idx, target in enumerate(targets):
                labels = target["labels"].long().to(device)
                if labels.numel() == 0:
                    continue
                if self.short_gt_presence_max_len > 0 and labels.numel() > self.short_gt_presence_max_len:
                    continue
                for label in labels:
                    # For each required character, at least one query should make
                    # that character beat blank. This targets short-sample
                    # empty/deletion failures more directly than expected-count.
                    margin_scores = log_nonblank[batch_idx, :, label] - log_blank[batch_idx]
                    best_margin = margin_scores.max()
                    presence_losses.append(F.softplus(self.short_gt_presence_margin - best_margin))

            if presence_losses:
                losses["loss_short_gt_presence"] = torch.stack(presence_losses).mean()
            else:
                losses["loss_short_gt_presence"] = loss * 0.0

        if self.short_gt_ce_enabled:
            char_probs = new_pred_logits[:, :, 1:].clamp(min=1e-8)
            short_gt_ce_loss, short_gt_ce_stats = compute_short_query_ce_loss(
                char_probs,
                targets,
                max_len=self.short_gt_ce_max_len,
                gamma=self.short_gt_ce_gamma,
            )
            losses["loss_short_gt_ce"] = short_gt_ce_loss
            losses["short_gt_ce_eligible_samples"] = torch.tensor(
                float(short_gt_ce_stats["eligible_samples"]),
                device=device,
            )
            losses["short_gt_ce_selected_chars"] = torch.tensor(
                float(short_gt_ce_stats["selected_chars"]),
                device=device,
            )

        if self.glyph_proto_aux_enabled:
            pred_proto_logits = outputs.get("pred_proto_logits")
            if pred_proto_logits is None:
                losses["loss_glyph_proto_aux"] = loss * 0.0
                losses["glyph_proto_aux_selected_chars"] = torch.tensor(0.0, device=device)
                losses["glyph_proto_aux_eligible_samples"] = torch.tensor(0.0, device=device)
            else:
                sorted_proto_logits = torch.gather(
                    pred_proto_logits,
                    1,
                    idx.unsqueeze(-1).expand(-1, -1, pred_proto_logits.shape[-1]),
                )
                char_probs = new_pred_logits[:, :, 1:].clamp(min=1e-8)
                proto_aux_loss, proto_aux_stats = compute_glyph_proto_aux_loss(
                    sorted_proto_logits,
                    char_probs,
                    targets,
                    max_len=self.glyph_proto_aux_max_len,
                    gamma=self.glyph_proto_aux_gamma,
                )
                losses["loss_glyph_proto_aux"] = proto_aux_loss
                losses["glyph_proto_aux_eligible_samples"] = torch.tensor(
                    float(proto_aux_stats["eligible_samples"]),
                    device=device,
                )
                losses["glyph_proto_aux_selected_chars"] = torch.tensor(
                    float(proto_aux_stats["selected_chars"]),
                    device=device,
                )

        if self.direction_loss_enabled:
            losses.update(self.loss_direction(outputs, targets, indices, num_boxes))
        if return_preds:
            return losses, new_pred_logits, None

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs

        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )

        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if self.query_budget_loss_enabled:
            loss_query_budget, query_budget_stats = compute_query_budget_loss(
                src_logits,
                targets,
                budget_scale=self.query_budget_scale,
                budget_margin=self.query_budget_margin,
            )
            losses["loss_query_budget"] = loss_query_budget
            losses["query_budget_expected_count"] = torch.tensor(
                query_budget_stats["expected_count_mean"],
                dtype=src_logits.dtype,
                device=src_logits.device,
            )
            losses["query_budget_target_count"] = torch.tensor(
                query_budget_stats["target_count_mean"],
                dtype=src_logits.dtype,
                device=src_logits.device,
            )

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]

        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
 

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses["loss_xy"] = loss_bbox[..., :2].sum() / num_boxes
            losses["loss_hw"] = loss_bbox[..., 2:].sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "loss_CTC": self.loss_CTC,
            "direction": self.loss_direction,
        }

        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward_CTC(self, outputs, targets, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        device = next(iter(outputs.values())).device

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, None, num_boxes))

        kwargs = {}
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, None, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            l_dict = self.get_loss(
                loss, interm_outputs, targets, None, num_boxes, **kwargs
            )
            l_dict = {k + f"_interm": v for k, v in l_dict.items()}
            losses.update(l_dict)

        # enc output loss
        if "enc_outputs" in outputs:
            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, enc_outputs, targets, None, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups

    def forward_standard(self, outputs, targets, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """


        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        device = next(iter(outputs.values())).device
        

        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs["dn_meta"]

        if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.range(0, len(targets[i]["labels"]) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(scalar)) * single_pad
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
            l_dict = {}
            for loss in self.losses:
    
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(
                    self.get_loss(
                        loss,
                        output_known_lbs_bboxes,
                        targets,
                        dn_pos_idx,
                        num_boxes * scalar,
                        **kwargs,
                    )
                )

            l_dict = {k + f"_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_xy_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_hw_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["cardinality_error_dn"] = torch.as_tensor(0.0).to("cuda")
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes["aux_outputs"][idx]
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if "labels" in loss:
                            kwargs = {"log": False}

                        l_dict.update(
                            self.get_loss(
                                loss,
                                aux_outputs_known,
                                targets,
                                dn_pos_idx,
                                num_boxes * scalar,
                                **kwargs,
                            )
                        )

                    l_dict = {k + f"_dn_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_xy_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_hw_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["cardinality_error_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs = {"log": False}
                l_dict = self.get_loss(
                    loss, interm_outputs, targets, indices, num_boxes, **kwargs
                )
                l_dict = {k + f"_interm": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if "enc_outputs" in outputs:
            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, enc_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def forward(self, outputs, targets, return_indices=False, eval=False):
        if self.CTC and not eval:
            self.losses = self.losses_CTC
            return self.forward_CTC(outputs, targets, return_indices)
        elif self.CTC and eval:
            self.losses = self.losses_all
            return self.forward_standard(outputs, targets, return_indices)
        else:
            return self.forward_standard(outputs, targets, return_indices)

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [
                nms(b, s, iou_threshold=self.nms_iou_threshold)
                for b, s in zip(boxes, scores)
            ]

            results = [
                {"scores": s[i], "labels": l[i], "boxes": b[i]}
                for s, l, b, i in zip(scores, labels, boxes, item_indices)
            ]
        else:
            results = [
                {"scores": s, "labels": l, "boxes": b}
                for s, l, b in zip(scores, labels, boxes)
            ]

        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name="dino")
def build_dino(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = DINO(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        use_direction_head=getattr(args, "use_direction_head", False),
        use_query_activation_head=getattr(args, "use_query_activation_head", False),
        query_activation_init_bias=getattr(args, "query_activation_init_bias", 0.0),
        use_glyph_prototype_head=getattr(args, "use_glyph_prototype_head", False),
        glyph_proto_dim=getattr(args, "glyph_proto_dim", 256),
        glyph_proto_temperature=getattr(args, "glyph_proto_temperature", 1.0),
        glyph_proto_trainable=getattr(args, "glyph_proto_trainable", True),
        use_query_sequence_refiner=getattr(args, "use_query_sequence_refiner", False),
        query_sequence_refiner_layers=getattr(args, "query_sequence_refiner_layers", 1),
        query_sequence_refiner_kernel=getattr(args, "query_sequence_refiner_kernel", 3),
        query_sequence_refiner_dropout=getattr(args, "query_sequence_refiner_dropout", 0.0),
        use_sorted_ctc_refiner=getattr(args, "use_sorted_ctc_refiner", False),
        sorted_ctc_refiner_layers=getattr(args, "sorted_ctc_refiner_layers", 1),
        sorted_ctc_refiner_kernel=getattr(args, "sorted_ctc_refiner_kernel", 3),
        sorted_ctc_refiner_dropout=getattr(args, "sorted_ctc_refiner_dropout", 0.0),
        sorted_ctc_refiner_axis=getattr(args, "sorted_ctc_refiner_axis", 1),
        use_vertical_geometry_adapter=getattr(
            args, "use_vertical_geometry_adapter", False
        ),
        vertical_geometry_adapter_hidden_dim=getattr(
            args, "vertical_geometry_adapter_hidden_dim", 64
        ),
        vertical_geometry_adapter_axis=getattr(
            args, "vertical_geometry_adapter_axis", 1
        ),
        vertical_geometry_adapter_init_scale=getattr(
            args, "vertical_geometry_adapter_init_scale", 0.0
        ),
        use_ctc_length_calibration=getattr(
            args, "use_ctc_length_calibration", False
        ),
        ctc_length_init=getattr(args, "ctc_length_init", 8.0),
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    # for DN training
    if args.use_dn:
        weight_dict["loss_ce_dn"] = args.cls_loss_coef
        weight_dict["loss_bbox_dn"] = args.bbox_loss_coef
        weight_dict["loss_giou_dn"] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in clean_weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != "no":
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            "loss_ce": 1.0,
            "loss_bbox": 1.0 if not no_interm_box_loss else 0.0,
            "loss_giou": 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update(
            {
                k + f"_interm": v * interm_loss_coef * _coeff_weight_dict[k]
                for k, v in clean_weight_dict_wo_dn.items()
            }
        )
        weight_dict.update(interm_weight_dict)

    # CTC/direction losses belong to the line-recognition path only, so keep them
    # out of the DETR aux/intermediate weight construction above.
    weight_dict["loss_CTC"] = args.CTC_loss_coef
    ctc_count_loss_coef = getattr(args, "ctc_count_loss_coef", 0.0)
    if ctc_count_loss_coef > 0:
        weight_dict["loss_ctc_count"] = ctc_count_loss_coef
    ctc_collapsed_count_loss_coef = getattr(
        args, "ctc_collapsed_count_loss_coef", 0.0
    )
    if ctc_collapsed_count_loss_coef > 0:
        weight_dict["loss_ctc_collapsed_count"] = ctc_collapsed_count_loss_coef
    ctc_soft_order_loss_coef = getattr(args, "ctc_soft_order_loss_coef", 0.0)
    if ctc_soft_order_loss_coef > 0:
        weight_dict["loss_ctc_soft_order"] = ctc_soft_order_loss_coef
    ctc_viterbi_loss_coef = getattr(args, "ctc_viterbi_loss_coef", 0.0)
    if ctc_viterbi_loss_coef > 0:
        weight_dict["loss_ctc_viterbi"] = ctc_viterbi_loss_coef
    ctc_length_loss_coef = getattr(args, "ctc_length_loss_coef", 0.0)
    if getattr(args, "use_ctc_length_calibration", False) and ctc_length_loss_coef > 0:
        weight_dict["loss_ctc_length"] = ctc_length_loss_coef
    short_gt_presence_loss_coef = getattr(args, "short_gt_presence_loss_coef", 0.0)
    if short_gt_presence_loss_coef > 0:
        weight_dict["loss_short_gt_presence"] = short_gt_presence_loss_coef
    short_gt_ce_loss_coef = getattr(args, "short_gt_ce_loss_coef", 0.0)
    if short_gt_ce_loss_coef > 0:
        weight_dict["loss_short_gt_ce"] = short_gt_ce_loss_coef
    glyph_proto_aux_loss_coef = getattr(args, "glyph_proto_aux_loss_coef", 0.0)
    if glyph_proto_aux_loss_coef > 0:
        weight_dict["loss_glyph_proto_aux"] = glyph_proto_aux_loss_coef
    query_count_loss_coef = getattr(args, "query_count_loss_coef", 0.0)
    if getattr(args, "use_query_count_loss", False) and query_count_loss_coef > 0:
        weight_dict["loss_query_count"] = query_count_loss_coef
    query_budget_loss_coef = getattr(args, "query_budget_loss_coef", 0.0)
    if getattr(args, "use_query_budget_loss", False) and query_budget_loss_coef > 0:
        weight_dict["loss_query_budget"] = query_budget_loss_coef
    if getattr(args, "use_direction_head", False):
        weight_dict["loss_direction"] = getattr(args, "direction_loss_coef", 1.0)

    losses = ["labels", "boxes", "cardinality"]
    if getattr(args, "use_direction_head", False):
        losses.append("direction")
    if args.masks:
        losses += ["masks"]

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        losses=losses,
        CTC=getattr(args, "mode_chr", False),
        ctc_blank_max=getattr(args, "ctc_blank_max", 1.0),
        ctc_count_loss_short_weight=getattr(args, "ctc_count_loss_short_weight", 1.0),
        ctc_collapsed_count_loss_short_weight=getattr(
            args, "ctc_collapsed_count_loss_short_weight", 1.0
        ),
        ctc_soft_order_temperature=getattr(args, "ctc_soft_order_temperature", 0.01),
        short_gt_presence_max_len=getattr(args, "short_gt_presence_max_len", 0),
        short_gt_presence_margin=getattr(args, "short_gt_presence_margin", 0.0),
        short_gt_ce_max_len=getattr(args, "short_gt_ce_max_len", 0),
        short_gt_ce_gamma=getattr(args, "short_gt_ce_gamma", 0.0),
        use_query_activation_head=getattr(args, "use_query_activation_head", False),
        use_query_count_loss=getattr(args, "use_query_count_loss", False),
        use_activation_gating=getattr(args, "use_activation_gating", False),
        use_glyph_prototype_head=getattr(args, "use_glyph_prototype_head", False),
        glyph_proto_fuse_coef=getattr(args, "glyph_proto_fuse_coef", 0.0),
        glyph_proto_aux_max_len=getattr(args, "glyph_proto_aux_max_len", 0),
        glyph_proto_aux_gamma=getattr(args, "glyph_proto_aux_gamma", 0.0),
        activation_gate_blank_coef=getattr(args, "activation_gate_blank_coef", 0.0),
        activation_gate_nonblank_coef=getattr(args, "activation_gate_nonblank_coef", 0.0),
        query_count_short_weight=getattr(args, "query_count_short_weight", 1.0),
        query_budget_scale=getattr(args, "query_budget_scale", 2.0),
        query_budget_margin=getattr(args, "query_budget_margin", 8.0),
        use_sorted_ctc_refiner=getattr(args, "use_sorted_ctc_refiner", False),
        ctc_viterbi_loss_blank_weight=getattr(args, "ctc_viterbi_loss_blank_weight", 0.0),
        ctc_viterbi_loss_nonblank_weight=getattr(args, "ctc_viterbi_loss_nonblank_weight", 1.0),
        ctc_viterbi_loss_max_len=getattr(args, "ctc_viterbi_loss_max_len", 0),
        use_ctc_length_calibration=getattr(
            args, "use_ctc_length_calibration", False
        ),
        ctc_length_bias_scale=getattr(args, "ctc_length_bias_scale", 0.0),
        ctc_length_bias_max=getattr(args, "ctc_length_bias_max", 1.0),
    )
    criterion.to(device)
    postprocessors = {
        "bbox": PostProcess(
            num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold
        )
    }
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85
            )

    return model, criterion, postprocessors
