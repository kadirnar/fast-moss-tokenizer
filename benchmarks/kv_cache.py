"""Ring-update microbenchmark; isolated component timings, not model speedup."""
import json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from benchmarks.baseline import measure
from fast_moss.kv_cache import complete
from upstream.modeling_moss_audio_tokenizer import RingKVCache


@torch.inference_mode()
def main():
    report={"scope":"KV cache update only", "gpu":torch.cuda.get_device_name(),"records":[]}
    for b,h,t,d,c in [(1,20,1,64,125),(1,12,8,64,1007),(2,12,8,64,1007),(8,12,8,64,1007)]:
        packed=torch.randn(b,t,3,h,d,device="cuda").permute(2,0,3,1,4)
        k,v=packed[1],packed[2]
        active=torch.ones(b,device="cuda",dtype=torch.bool)
        for name in ["reference","triton"]:
            cache=RingKVCache(b,h,d,c,dtype=torch.float32)
            fn=(lambda:cache.complete(k,v,active)) if name=="reference" else (lambda:complete(cache,k,v,active))
            record={"shape":[b,h,t,d],"capacity":c,"backend":name,
                    "eager":measure(fn,repeats=20),
                    "amortized_graph_ms":do_bench_cudagraph(fn,rep=20,return_mode="median")}
            print(name,record["shape"],"graph us",record["amortized_graph_ms"]*1000,flush=True)
            report["records"].append(record)
    Path("results/kv_cache.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
