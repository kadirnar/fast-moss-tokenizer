"""Actual checkpoint activations/weights: GEMM layouts, padding, and Triton IEEE.

Warm-cache microbenchmarks can overstate model-level performance. A separate
cache-evicted replay measurement is reported; neither is a whole-model speedup.
"""
import argparse
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from fast_moss.loading import load_model,REVISION,strict_precision
from fast_moss.graphs import GraphedCallable
from benchmarks.experimental_mm import linear,gemv_rows
from benchmarks.compare import difference


def evicted_replay(graph,flush,repeats=15):
    elapsed=[]
    for _ in range(repeats):
        flush.zero_()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();graph.graph.replay();end.record();end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return {"gpu_ms_median":statistics.median(elapsed),"samples":elapsed}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--frames",type=int,nargs="+",default=[1,3])
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--output",default="results/matrices.json")
    p.add_argument("--gemv-only",action="store_true")
    p.add_argument("--batches",type=int,nargs="+",default=[1])
    p.add_argument("--cublaslt",action="store_true")
    p.add_argument("--save-inputs")
    p.add_argument("--inputs")
    p.add_argument("--rows",type=int,nargs="+")
    a=p.parse_args()
    strict_precision()
    cases={};hooks=[]

    def collect(name):
        def hook(module,args):
            x=args[0].reshape(-1,args[0].shape[-1])
            w=module.weight
            key=(x.shape[0],*w.shape)
            if key not in cases:
                cases[key]={"name":name,"x":x.contiguous().clone(),"weight":w,"calls":0}
            cases[key]["calls"]+=1
        return hook
    if a.inputs:
        cases=torch.load(a.inputs,weights_only=True)
    else:
        model=load_model()
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Linear) and module.bias is None:
                hooks.append(module.register_forward_pre_hook(collect(name)))
        torch.manual_seed(2026)
        for batch in a.batches:
            for frames in a.frames:
                codes=model.encode(torch.randn(batch,1,1920*frames,device="cuda")*.05,return_dict=True).audio_codes
                model.decode(codes,return_dict=True)
        for hook in hooks:hook.remove()
        if a.save_inputs:torch.save(cases,a.save_inputs)
        del model
    report={"scope":"isolated actual-weight matrix operations, not whole-model speedups",
            "revision":REVISION,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
            "dtype":"float32","tf32":False,"triton_dot_precision":"ieee",
            "cublaslt_compute":"CUBLAS_COMPUTE_32F_PEDANTIC" if a.cublaslt else None,"records":[],"unavailable":[]}
    if a.cublaslt:
        from benchmarks.cublaslt import library
        report['cublaslt_version']=library().cublasLtGetVersion()
    flush=torch.empty(128*1024*1024,device="cuda",dtype=torch.uint8)
    selected=sorted(cases.items(),key=lambda item:item[1]["calls"]*item[0][1]*item[0][2],reverse=True)
    if a.rows:selected=[item for item in selected if item[0][0] in a.rows]
    if a.limit:selected=selected[:a.limit]
    for shape,case in selected:
        x,w=case["x"],case["weight"]
        gold=x.double()@w.double().T
        ref=F.linear(x,w)
        packed=w.T.contiguous()
        candidates={"torch":lambda z:F.linear(z,w),"torch_packed":lambda z:z@packed}
        # Pad rows at the end and keep them in the GEMM, then discard the extra outputs.
        candidates["torch_pad8"]=lambda z:F.linear(F.pad(z,(0,0,0,(-z.shape[0])%8)),w)[:z.shape[0]]
        plans=[];metadata={}
        if a.cublaslt:
            from benchmarks.cublaslt import LinearPlan
            candidates={"torch":candidates["torch"]}
            for layout in ['col','row','packed']:
                try:
                    plan=LinearPlan(w,x.shape[0],layout)
                except RuntimeError as error:
                    report['unavailable'].append({'shape_MNK':shape,'layout':layout,'error':str(error)})
                    continue
                plans.append(plan)
                for index in range(len(plan.algorithms)):
                    name=f'lt_{layout}_{index}'
                    candidates[name]=lambda z,plan=plan,index=index:plan(z,index)
                    metadata[name]=plan.metadata(index)
        elif a.gemv_only:
            candidates={"torch":candidates["torch"]}
            for bn in [2,4,8]:
                candidates[f"triton_gemv_{bn}"]=lambda z,bn=bn:gemv_rows(z,w,bn)
        else:
            for tile in [(16,32,32),(16,64,32),(16,64,64),(16,32,64)]:
                candidates[f"triton_{tile}"]=lambda z,tile=tile:linear(z,w,tile)
        for name,fn in candidates.items():
            out=fn(x)
            graph=GraphedCallable(lambda z:(fn(z),),x)
            record={"shape_MNK":shape,"layer":case["name"],"occurrences":case["calls"],"backend":name,
                    "reference_difference":difference(ref,out),"graph_vs_eager":difference(out,graph(x)[0]),
                    "algorithm":metadata.get(name),"fp64_max_abs":(out.double()-gold).abs().max().item(),
                    "reference_fp64_max_abs":(ref.double()-gold).abs().max().item(),
                    "hot_graph_ms":do_bench_cudagraph(lambda:fn(x),rep=10,return_mode="median"),
                    "evicted_replay":evicted_replay(graph,flush)}
            report["records"].append(record)
            print(shape,name,"hot_us",round(record["hot_graph_ms"]*1000,2),
                  "exact",record["reference_difference"]["exact"],flush=True)
            del graph
        for plan in plans:plan.close()
        Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
