"""Kernel-only timings; never interpret these as whole-model speedups."""
import json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from benchmarks.baseline import measure
from fast_moss.kernels import scale_add
from fast_moss.cute_kernels import scale_add as cute_scale_add
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    records = []
    for shape in [(1, 1, 1280), (1, 8, 768), (1, 100, 768), (8, 100, 768)]:
        x = torch.randn(shape, device="cuda")
        u = torch.randn_like(x)
        s = torch.randn(shape[-1], device="cuda")
        for name, kernel in {"torch": lambda x, u, s: x + s * u,
                             "triton": scale_add, "cute": cute_scale_add}.items():
            assert torch.equal(kernel(x, u, s), x + s * u)
            fn = lambda: kernel(x, u, s)
            eager = measure(fn, repeats=30)
            captured = GraphedCallable(lambda a, b, c: (kernel(a, b, c),), x, u, s)
            # Time graph replay alone for device kernel cost, without input copies/output clones.
            graph = measure(captured.graph.replay, repeats=30)
            amortized = do_bench_cudagraph(fn, rep=20, return_mode="median")
            record = {"shape": shape, "backend": name, "eager": eager, "graph_replay": graph,
                      "amortized_graph_kernel_ms": amortized}
            records.append(record)
            print(shape, name, "eager us", round(eager["wall_ms_median"] * 1000, 2),
                  "amortized graph kernel us", round(amortized * 1000, 2), flush=True)
    Path("results/kernels.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(),
                                                      "scope": "residual only", "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
