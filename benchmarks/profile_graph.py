"""Profile the current optimized full-model CUDA graphs to find remaining costs."""
from contextlib import ExitStack
import math
import argparse
import json
from pathlib import Path
import torch
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from fast_moss.streaming import StreamingSession
from benchmarks.baseline import measure


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seconds",type=float,default=.24)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--output",default="results/full_graph_profile.json")
    p.add_argument("--share-rope-tables",action="store_true")
    p.add_argument("--attention-mask-backend", choices=["none", "triton"], default="none")
    a=p.parse_args()
    torch.manual_seed(2026)
    model=load_model()
    x=torch.randn(a.batch,1,round(a.seconds*24000),device="cuda")*.05
    report={"scope":"full checkpoint optimized graph", "revision":REVISION,"torch":torch.__version__,
            "gpu":torch.cuda.get_device_name(),"dtype":"float32","tf32":False,"quantizers":32,
            "seconds":a.seconds,"batch":a.batch,"streaming":a.streaming,"share_rope_tables":a.share_rope_tables,"attention_mask_backend":a.attention_mask_backend,
            "attention_mask_format":"aligned_fp32_additive" if a.attention_mask_backend=="triton" else "upstream_boolean","results":{}}
    with optimized(model,residual_backend="triton",rope_backend="triton",kv_backend="triton",
                   share_rope_tables=a.share_rope_tables,attention_mask_backend=a.attention_mask_backend):
        codes=model._encode_frame(x).audio_codes
        for name,fn,inp in [("encode",lambda v:(model._encode_frame(v).audio_codes,),x),
                            ("decode",lambda v:(model._decode_frame(v).audio,),codes)]:
            with ExitStack() as stack:
                if a.streaming:
                    frames = x.shape[-1] // 1920
                    if frames < 1 or frames * 1920 != x.shape[-1]:
                        raise ValueError("Streaming requires complete codec frames")
                    session = stack.enter_context(StreamingSession(model, name, a.batch, frames))
                    run = lambda: session.push(inp)
                    # Fill the ten-second ring before profiling steady state.
                    context_frames = model.causal_transformer_context_duration * model.sampling_rate / model.downsample_rate
                    for _ in range(math.ceil(context_frames / frames) + 1):
                        run()
                else:
                    graph = GraphedCallable(fn, inp)
                    run = lambda: graph(inp)
                report["results"][name]=measure(run,repeats=10)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):
                        run()
                    torch.cuda.synchronize()
                table=prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=30)
                Path(a.output).with_suffix(f".{name}.txt").write_text(
                    "Five graph replays; input copies/output clones included. Streaming profiles use a filled ring.\n"+
                    "\n".join(line.rstrip() for line in table.splitlines())+"\n")
                prof.export_chrome_trace(str(Path(a.output).with_suffix(f".{name}.json.gz")))
                print(name,report["results"][name],flush=True)
                if not a.streaming:
                    del graph
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
