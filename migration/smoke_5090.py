"""Check the actual CUDA extension's forward and backward against the reference."""
import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    for flag in ["DTLR_MSDA_FORCE_PYTORCH", "DTLR_MSDA_FALLBACK_ON_ERROR"]:
        if os.environ.get(flag, "0") == "1":
            raise RuntimeError(f"Unset {flag}: this check must exercise the compiled extension")
    import torch
    import MultiScaleDeformableAttention as extension
    from models.dino.ops.functions.ms_deform_attn_func import (
        MSDeformAttnFunction, ms_deform_attn_core_pytorch,
    )

    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (12, 0)
    torch.manual_seed(42)
    shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long, device="cuda")
    starts = torch.tensor([0, 16], dtype=torch.long, device="cuda")
    errors = []
    for queries in [7, 64]:
        value = torch.randn(2, 20, 8, 32, device="cuda")
        locations = torch.rand(2, queries, 8, 2, 4, 2, device="cuda")
        weights = torch.rand(2, queries, 8, 2, 4, device="cuda")
        weights = weights / weights.sum((-1, -2), keepdim=True)
        actual_inputs = [x.detach().clone().requires_grad_(True) for x in [value, locations, weights]]
        reference_inputs = [x.detach().clone().requires_grad_(True) for x in [value, locations, weights]]
        actual = MSDeformAttnFunction.apply(actual_inputs[0], shapes, starts,
                                            actual_inputs[1], actual_inputs[2], 64)
        reference = ms_deform_attn_core_pytorch(reference_inputs[0], shapes,
                                               reference_inputs[1], reference_inputs[2])
        torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-3)
        gradient = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(actual, actual_inputs, gradient)
        reference_gradients = torch.autograd.grad(reference, reference_inputs, gradient)
        entry = {"queries": queries, "forward_max_abs": float((actual-reference).abs().max())}
        for name, actual_grad, reference_grad in zip(
            ["value", "sampling_locations", "attention_weights"], actual_gradients, reference_gradients
        ):
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, reference_grad, atol=2e-4, rtol=2e-3)
            entry[name + "_grad_max_abs"] = float((actual_grad-reference_grad).abs().max())
        errors.append(entry)
    torch.cuda.synchronize()
    result = {"status": "PASS", "scope": "FP32 CUDA extension forward/backward only; model checks still required",
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
              "extension": extension.__file__, "cases": errors}
    output = pathlib.Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
