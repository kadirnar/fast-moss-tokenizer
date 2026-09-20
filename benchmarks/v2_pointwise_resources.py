"""Inspect representative compiled pointwise kernels; not a latency benchmark."""

import hashlib
import json
from pathlib import Path
import re

import torch
import triton

from fast_moss.v2_pointwise import _rotate, _scale_add


@torch.inference_mode()
def main():
    report = {
        "scope": "representative Triton BF16/FP32 rotary and residual binaries; separate FP32 multiply/add and no FTZ",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": hashlib.sha256(
            Path("fast_moss/v2_pointwise.py").read_bytes()
        ).hexdigest(),
        "kernels": [],
    }

    def record(name, kernel):
        ptx = kernel.asm["ptx"]
        row = {
            "name": name,
            "registers": kernel.n_regs,
            "spills": kernel.n_spills,
            "shared_bytes": kernel.metadata.shared,
            "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            "fma_instructions": len(re.findall(r"\bfma\.", ptx)),
            "ftz_instructions": len(re.findall(r"\.ftz\.", ptx)),
            "fp32_mul_instructions": len(re.findall(r"\bmul(?:\.rn)?\.f32\b", ptx)),
            "fp32_add_sub_instructions": len(
                re.findall(r"\b(?:add|sub)(?:\.rn)?\.f32\b", ptx)
            ),
        }
        assert row["fma_instructions"] == 0 and row["ftz_instructions"] == 0, row
        assert row["fp32_mul_instructions"] and row["fp32_add_sub_instructions"], row
        report["kernels"].append(row)

    for dtype in [torch.float32, torch.bfloat16]:
        q = torch.randn(1, 12, 24, 64, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        cos = torch.randn(1, 24, 32, device="cuda")
        sin = torch.randn_like(cos)
        oq = torch.empty_like(q)
        ok = torch.empty_like(k)
        n = q.numel() // 2
        kernel = _rotate[(triton.cdiv(n, 256),)](
            q,
            k,
            cos,
            sin,
            oq,
            ok,
            n,
            12,
            24,
            64,
            *q.stride(),
            *k.stride(),
            256,
            enable_fp_fusion=False,
        )
        record("rotate_" + str(dtype), kernel)
    for xdtype in [torch.float32, torch.bfloat16]:
        for udtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(1, 24, 768, device="cuda", dtype=xdtype)
            u = torch.randn(x.shape, device="cuda", dtype=udtype)
            scale = torch.randn(768, device="cuda")
            out = torch.empty(x.shape, device="cuda")
            kernel = _scale_add[(triton.cdiv(x.numel(), 256),)](
                x, u, scale, out, x.numel(), 768, 256, enable_fp_fusion=False
            )
            record("scale_add_" + str(xdtype) + "_" + str(udtype), kernel)
    torch.cuda.synchronize()
    report["all_passed"] = True
    Path("results/v2_pointwise_resources.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("PASS", len(report["kernels"]), "representative kernels")


if __name__ == "__main__":
    main()
