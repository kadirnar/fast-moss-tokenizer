"""Matched batch-size throughput and latency; does not conflate sequential baselines."""
import argparse
import json
from pathlib import Path
import torch
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from benchmarks.baseline import measure
from benchmarks.compare import difference


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batches",type=int,nargs="+",default=[1,2,4,8,16,32,64,128])
    p.add_argument("--frames",type=int,default=3)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--attention-mask-backend", choices=["none", "triton"], default="none")
    p.add_argument("--output",default="results/batching.json")
    a=p.parse_args()
    model=load_model()
    torch.manual_seed(921)
    report={"scope":"matched full-checkpoint offline batch throughput", "revision":REVISION,
            "gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"dtype":"float32","tf32":False,
            "quantizers":32,"frames":a.frames,"seconds_per_lane":a.frames*.08,
            "attention_mask_backend":a.attention_mask_backend,"share_rope_tables":True,"records":[]}
    for batch in a.batches:
        torch.cuda.reset_peak_memory_stats()
        x=torch.randn(batch,1,a.frames*1920,device="cuda")*.05
        encode=lambda v:(model._encode_frame(v).audio_codes,)
        codes=encode(x)[0]
        decode=lambda v:(model._decode_frame(v).audio,)
        for direction,fn,inp in [("encode",encode,x),("decode",decode,codes)]:
            reference=fn(inp)[0]
            baseline=measure(lambda:fn(inp),repeats=a.repeats)
            with optimized(model,residual_backend="triton",rope_backend="triton",kv_backend="triton",share_rope_tables=True,
                           attention_mask_backend=a.attention_mask_backend):
                graph=GraphedCallable(fn,inp)
                fidelity=difference(reference,graph(inp)[0])
                candidate=measure(lambda:graph(inp),repeats=a.repeats)
                del graph
            total_audio_seconds=batch*a.frames*.08
            record={"batch":batch,"direction":direction,"fidelity":fidelity,
                    "baseline":baseline,"optimized":candidate,
                    "matched_batch_speedup":baseline["wall_ms_median"]/candidate["wall_ms_median"],
                    "baseline_audio_seconds_per_second":total_audio_seconds/(baseline["wall_ms_median"]*.001),
                    "optimized_audio_seconds_per_second":total_audio_seconds/(candidate["wall_ms_median"]*.001),
                    "peak_memory_bytes":torch.cuda.max_memory_allocated()}
            report["records"].append(record)
            print(batch,direction,"ms",round(candidate["wall_ms_median"],3),"speedup",round(record["matched_batch_speedup"],2),
                  "exact",fidelity["exact"],flush=True)
            Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
        del reference,x,codes
    report["all_exact"]=all(r["fidelity"]["exact"] for r in report["records"])
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    if not report["all_exact"]:raise SystemExit("Batch fidelity failed")


if __name__=="__main__":
    main()
