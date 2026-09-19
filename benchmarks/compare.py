"""Matched FP32, all-codebook baseline/candidate timings and exact-output gates."""
import argparse
import json
import time
from pathlib import Path
from contextlib import nullcontext

import torch

from benchmarks.baseline import measure
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from fast_moss.streaming import StreamingSession


def difference(reference, candidate):
    if reference.shape != candidate.shape:
        return {"exact": False, "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
    d = reference.double() - candidate.double()
    return {"exact": torch.equal(reference, candidate), "max_abs": d.abs().max().item(),
            "rmse": d.square().mean().sqrt().item(),
            "different_elements": (reference != candidate).sum().item(), "elements": reference.numel()}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=.08)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--audio", help="Optional mono WAV/FLAC, resampled to 24 kHz")
    p.add_argument("--backend", choices=["triton", "cute", "none"], default="triton")
    p.add_argument("--kv-backend", choices=["none", "triton"], default="none")
    p.add_argument("--rope-backend", choices=["none", "triton"], default="none")
    p.add_argument("--share-rope-tables", action="store_true")
    p.add_argument("--attention-mask-backend", choices=["none", "triton"], default="none")
    p.add_argument("--stream-chunks", type=int, default=0)
    p.add_argument("--output", default="results/compare.json")
    p.add_argument("--structural", action="store_true", help="Use a reduced random model, NOT the checkpoint")
    a = p.parse_args()
    if a.structural:
        from benchmarks.fixtures import structural_model
        model = structural_model()
    else:
        model = load_model()
    torch.manual_seed(2026)
    if a.audio:
        import soundfile as sf
        import torchaudio.functional as AF
        samples, sr = sf.read(a.audio, dtype="float32", always_2d=True)
        mono = torch.from_numpy(samples).mean(1)
        if sr != 24000:
            mono = AF.resample(mono, sr, 24000)
        mono = mono[:round(a.seconds * 24000)]
        x = mono.cuda()[None, None].repeat(a.batch, 1, 1)
    else:
        x = torch.randn(a.batch, 1, round(a.seconds * 24000), device="cuda") * .05
    if x.shape[-1] % model.downsample_rate:
        raise ValueError("Use whole 1920-sample frames for matched upstream/offline comparisons")
    report = {"scope": "reduced random structural fixture" if a.structural else "full checkpoint",
              "revision": REVISION, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "dtype": "float32", "tf32": False, "batch": a.batch, "samples": x.shape[-1],
              "input": a.audio or "seeded Gaussian, amplitude 0.05", "quantizers": 32,
              "residual_backend":a.backend,"rope_backend":a.rope_backend,"kv_backend":a.kv_backend,
              "share_rope_tables":a.share_rope_tables,"attention_mask_backend":a.attention_mask_backend,
            "attention_mask_format":"aligned_fp32_additive" if a.attention_mask_backend=="triton" else "upstream_boolean","results": {}}
    encode = lambda inp: (model._encode_frame(inp).audio_codes,)
    codes = encode(x)[0]
    decode = lambda inp: (model._decode_frame(inp).audio,)
    reference = {"encode": codes, "decode": decode(codes)[0]}
    for mode in ["reference", "cached", "fused"]:
        ctx = nullcontext() if mode == "reference" else optimized(
            model, residual_backend="none" if mode == "cached" else a.backend,
            rope_backend=a.rope_backend if mode=="fused" else "none",
            share_rope_tables=a.share_rope_tables if mode=="fused" else False,
            attention_mask_backend=a.attention_mask_backend if mode=="fused" else "none")
        with ctx:
            for direction, fn, inp in [("encode", encode, x), ("decode", decode, codes)]:
                key = f"{mode}_{direction}"
                report["results"][key] = {"fidelity": difference(reference[direction], fn(inp)[0]),
                                           "timing": measure(lambda: fn(inp), repeats=a.repeats)}
                print(key, report["results"][key], flush=True)
                t = time.perf_counter()
                graph = GraphedCallable(fn, inp)
                torch.cuda.synchronize()
                key += "_graph"
                report["results"][key] = {"setup_seconds": time.perf_counter() - t,
                                           "fidelity": difference(reference[direction], graph(inp)[0]),
                                           "timing": measure(lambda: graph(inp), repeats=a.repeats)}
                print(key, report["results"][key], flush=True)
                del graph
    if a.stream_chunks:
        report["streaming_reference"] = "Eager StreamingSession with complete chunk history; upstream ring capacity corrected"
        report["streaming"] = {}
        for direction, inp in [("encode", x), ("decode", codes)]:
            chunks = list(inp.chunk(a.stream_chunks, dim=-1))
            frames = chunks[0].shape[-1] // 1920 if direction == "encode" else chunks[0].shape[-1]
            if frames < 1 or any(t.shape != chunks[0].shape for t in chunks):
                raise ValueError("Streaming benchmark requires equal complete chunks")
            with StreamingSession(model, direction, a.batch, frames, use_graph=False) as session:
                ref = [session.push(chunk)[0].clone() for chunk in chunks]
            with optimized(model, residual_backend=a.backend, kv_backend=a.kv_backend,
                           rope_backend=a.rope_backend,share_rope_tables=a.share_rope_tables,attention_mask_backend=a.attention_mask_backend):
                with StreamingSession(model, direction, a.batch, frames) as session:
                    actual = [session.push(chunk)[0] for chunk in chunks]
                    fidelity = [difference(r, c) for r, c in zip(ref, actual)]
                    def stream_pass():
                        session.reset()
                        for chunk in chunks:
                            session.push(chunk)
                    timing = measure(stream_pass, repeats=a.repeats)
            with StreamingSession(model, direction, a.batch, frames, use_graph=False) as session:
                baseline = measure(stream_pass, repeats=a.repeats)
            report["streaming"][direction] = {"chunks": len(chunks), "fidelity": fidelity,
                                                "kv_backend": a.kv_backend,
                                                "reference": baseline, "optimized_graph": timing}
            print("streaming", direction, report["streaming"][direction], flush=True)
    report["all_exact"] = all(r["fidelity"]["exact"] for r in report["results"].values()) and all(
        f["exact"] for r in report.get("streaming", {}).values() for f in r["fidelity"])
    report["peak_memory_bytes"]=torch.cuda.max_memory_allocated()
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_exact"]:
        raise SystemExit("Fidelity gate failed; inspect the saved report")


if __name__ == "__main__":
    main()
