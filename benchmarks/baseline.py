"""FP32 full-checkpoint latency and operator evidence, without reducing RVQ depth."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from fast_moss.loading import load_model, REVISION


def measure(fn, warmup=3, repeats=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wall, gpu = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t) * 1000)
        gpu.append(start.elapsed_time(end))
    return {"wall_ms_median": statistics.median(wall), "gpu_ms_median": statistics.median(gpu),
            "wall_ms_samples": wall, "gpu_ms_samples": gpu}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=0.08)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--output", default="results/baseline.json")
    p.add_argument("--structural", action="store_true", help="Use a reduced random model, NOT the checkpoint")
    a = p.parse_args()
    torch.manual_seed(2026)
    if a.structural:
        from benchmarks.fixtures import structural_model
        model = structural_model()
    else:
        model = load_model()
    x = torch.randn(a.batch, 1, round(a.seconds * 24000), device="cuda") * 0.05
    codes = model.encode(x, return_dict=True).audio_codes
    report = {"scope": "reduced random structural fixture" if a.structural else "full checkpoint",
              "revision": REVISION, "torch": torch.__version__, "python": platform.python_version(),
              "gpu": torch.cuda.get_device_name(), "dtype": "float32", "tf32": False,
              "seconds": a.seconds, "batch": a.batch, "quantizers": codes.shape[0],
              "parameters": sum(p.numel() for p in model.parameters()), "input": "seeded Gaussian, amplitude 0.05"}
    for name, fn in {"encode": lambda: model.encode(x, return_dict=True),
                     "decode": lambda: model.decode(codes, return_dict=True)}.items():
        report[name] = measure(fn, repeats=a.repeats)
        print(name, report[name], flush=True)
        if a.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
                fn()
                torch.cuda.synchronize()
            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=35)
            Path(a.output).with_suffix(f".{name}.txt").write_text(
                "\n".join(line.rstrip() for line in table.splitlines()) + "\n")
            prof.export_chrome_trace(str(Path(a.output).with_suffix(f".{name}.json.gz")))
    report["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
