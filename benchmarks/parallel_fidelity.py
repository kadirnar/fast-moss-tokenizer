"""Pretrained model validation of paused/resumed/reused streaming batch lanes."""
import argparse
import json
from pathlib import Path
import torch
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession
from benchmarks.compare import difference


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--share-rope-tables",action="store_true")
    p.add_argument("--output",default="results/full_parallel_fidelity.json")
    a=p.parse_args()
    model=load_model()
    torch.manual_seed(918)
    x=torch.randn(2,1,5*1920,device="cuda")*.05
    codes=model.encode(x,return_dict=True).audio_codes
    report={"scope":"full checkpoint, independent batch lane timelines", "revision":REVISION,
            "torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"dtype":"float32","tf32":False,
            "input":"seeded Gaussian amplitude .05 and its encoded tokens","quantizers":32,"batch":2,
            "share_rope_tables":a.share_rope_tables,"results":{}}
    for direction,inp in [("encode",x),("decode",codes)]:
        chunks=list(inp.split(1920 if direction=="encode" else 1,dim=-1))
        dim=1 if direction=="encode" else 0
        with StreamingSession(model,direction,batch_size=2,use_graph=False) as session:
            reference=[session.push(chunk)[0].clone() for chunk in chunks]
        with StreamingSession(model,direction,batch_size=2,use_graph=False) as session:
            slow_reference=[session.push(chunks[i])[0].clone() for i in [0,2,4]]
        records=[]
        mask=torch.tensor([True,False],device="cuda")
        with optimized(model,residual_backend="triton",rope_backend="triton",kv_backend="triton",
                       share_rope_tables=a.share_rope_tables):
            with StreamingSession(model,direction,batch_size=2) as session:
                for i,chunk in enumerate(chunks):
                    out,lengths=session.push(chunk,active_mask=mask if i%2 else None)
                    record={"step":i,"lane0":difference(reference[i].select(dim,0),out.select(dim,0)),
                            "lengths":lengths.tolist()}
                    if i%2==0:
                        record["lane1"]=difference(slow_reference[i//2].select(dim,1),out.select(dim,1))
                    else:
                        record["paused_lane_length_zero"]=lengths[1].item()==0
                    records.append(record)
                session.reset(torch.tensor([False,True],device="cuda"))
                fresh,_=session.push(chunks[0])
                fresh_diff=difference(slow_reference[0].select(dim,1),fresh.select(dim,1))
        report["results"][direction]={"steps":records,"reused_lane":fresh_diff}
        print(direction,report["results"][direction],flush=True)
    report["all_exact"]=all(
        result["reused_lane"]["exact"] and all(
            record["lane0"]["exact"] and record.get("lane1",{"exact":True})["exact"]
            and record.get("paused_lane_length_zero",True) for record in result["steps"])
        for result in report["results"].values())
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    if not report["all_exact"]:
        raise SystemExit("Independent lane fidelity failed")


if __name__=="__main__":
    main()
