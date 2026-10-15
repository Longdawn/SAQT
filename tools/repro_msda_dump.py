import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dino.ops.functions.ms_deform_attn_func import (
    MSDeformAttnFunction,
    ms_deform_attn_core_pytorch,
)


def summarize(name, tensor):
    if not torch.is_tensor(tensor):
        return f"{name}=<{type(tensor).__name__}>"
    parts = [f"{name}.shape={tuple(tensor.shape)}", f"{name}.dtype={tensor.dtype}", f"{name}.device={tensor.device}"]
    if tensor.numel() > 0 and torch.is_floating_point(tensor):
        parts.append(f"{name}.min={tensor.min().item():.6g}")
        parts.append(f"{name}.max={tensor.max().item():.6g}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--im2col_step", default=64, type=int)
    args = parser.parse_args()

    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    value = payload["value"].to(args.device)
    spatial_shapes = payload["spatial_shapes"].to(args.device)
    level_start_index = payload["level_start_index"].to(args.device)
    sampling_locations = payload["sampling_locations"].to(args.device)
    attention_weights = payload["attention_weights"].to(args.device)

    print(summarize("value", value))
    print(summarize("spatial_shapes", spatial_shapes))
    print(summarize("level_start_index", level_start_index))
    print(summarize("sampling_locations", sampling_locations))
    print(summarize("attention_weights", attention_weights))

    print("\n[PyTorch core]")
    torch.cuda.synchronize(device=args.device)
    out_pt = ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
    torch.cuda.synchronize(device=args.device)
    print(summarize("out_pt", out_pt))

    print("\n[CUDA op]")
    torch.cuda.synchronize(device=args.device)
    out_cuda = MSDeformAttnFunction.apply(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        args.im2col_step,
    )
    torch.cuda.synchronize(device=args.device)
    print(summarize("out_cuda", out_cuda))

    max_abs = (out_cuda - out_pt).abs().max().item()
    print(f"\nmax_abs_diff={max_abs:.6g}")


if __name__ == "__main__":
    main()
