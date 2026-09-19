"""Full-checkpoint exact regression gates on real audio and synthetic edge cases."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import soundfile as sf
import torchaudio.functional as AF
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from benchmarks.compare import difference


def cases():
    for entry in json.loads(Path("data/manifest.json").read_text()):
        path=Path(entry["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError(f"Audio checksum changed: {path}")
        samples,sr=sf.read(path,dtype="float32",always_2d=True)
        x=torch.from_numpy(samples).mean(1)
        if sr!=24000:
            x=AF.resample(x,sr,24000)
        for frames in [1,10,40]:
            if x.numel()>=frames*1920:
                yield f"{path.stem}_{frames}frames",x[:frames*1920].view(1,1,-1),entry
    yield "silence",torch.zeros(1,1,1920),None
    impulse=torch.zeros(1,1,19200);impulse[...,9599]=1
    yield "impulse",impulse,None
    yield "full_scale_alternating",(torch.arange(19200)%2*2-1).float().view(1,1,-1),None
    yield "tiny_signal",torch.full((1,1,1920),1e-8),None


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/full_fidelity.json")
    p.add_argument("--backend",choices=["triton","cute"],default="triton")
    p.add_argument("--rope-backend",choices=["none","triton"],default="triton")
    a=p.parse_args()
    model=load_model()
    def run(x):
        enc=model._encode_frame(x)
        dec=model._decode_frame(enc.audio_codes)
        return enc.audio_codes,enc.encoder_hidden_states,dec.audio
    report={"scope":"full checkpoint, small regression corpus", "revision":REVISION,
            "torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"dtype":"float32","tf32":False,
            "quantizers":32,"residual_backend":a.backend,"rope_backend":a.rope_backend,"cases":[]}
    for name,cpu,source in cases():
        x=cpu.cuda()
        reference=run(x)
        record={"name":name,"samples":x.numel(),"source":source,"comparisons":{}}
        with optimized(model,residual_backend=a.backend,rope_backend=a.rope_backend,kv_backend="triton"):
            eager=run(x)
            graph=GraphedCallable(run,x)
            replay=graph(x)
            for mode,outputs in [("eager",eager),("graph",replay)]:
                record["comparisons"][mode]={label:difference(r,c) for label,r,c in
                    zip(["codes","encoder_hidden","audio"],reference,outputs)}
            del graph,eager,replay
        report["cases"].append(record)
        exact=all(v["exact"] for comparison in record["comparisons"].values() for v in comparison.values())
        print(name,"exact",exact,flush=True)
        # Save partial evidence even if a later input fails or runs out of memory.
        Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
        del reference,x
    report["all_exact"]=all(v["exact"] for record in report["cases"]
                            for comparison in record["comparisons"].values() for v in comparison.values())
    report["peak_memory_bytes"]=torch.cuda.max_memory_allocated()
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    if not report["all_exact"]:
        raise SystemExit("Exact fidelity gate failed")


if __name__=="__main__":
    main()
