"""Profile the current optimized full-model CUDA graphs to find remaining costs."""
import argparse
import json
from pathlib import Path
import torch
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from benchmarks.baseline import measure


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seconds",type=float,default=.24)
    p.add_argument("--output",default="results/full_graph_profile.json")
    a=p.parse_args()
    torch.manual_seed(2026)
    model=load_model()
    x=torch.randn(1,1,round(a.seconds*24000),device="cuda")*.05
    report={"scope":"full checkpoint optimized graph", "revision":REVISION,"torch":torch.__version__,
            "gpu":torch.cuda.get_device_name(),"dtype":"float32","tf32":False,"quantizers":32,
            "seconds":a.seconds,"batch":1,"results":{}}
    with optimized(model,residual_backend="triton",rope_backend="triton",kv_backend="triton"):
        codes=model._encode_frame(x).audio_codes
        for name,fn,inp in [("encode",lambda v:(model._encode_frame(v).audio_codes,),x),
                            ("decode",lambda v:(model._decode_frame(v).audio,),codes)]:
            graph=GraphedCallable(fn,inp)
            report["results"][name]=measure(lambda:graph(inp),repeats=10)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    graph(inp)
                torch.cuda.synchronize()
            table=prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=30)
            Path(a.output).with_suffix(f".{name}.txt").write_text(
                "Five graph replays; input copies/output clones included.\n"+
                "\n".join(line.rstrip() for line in table.splitlines())+"\n")
            prof.export_chrome_trace(str(Path(a.output).with_suffix(f".{name}.json.gz")))
            print(name,report["results"][name],flush=True)
            del graph
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
