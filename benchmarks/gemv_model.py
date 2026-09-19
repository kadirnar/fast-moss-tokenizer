"""Whole-model evaluation of an experimental FP32 reduction order.

This does not enable GEMV in the supported runtime. Exactness and audio error
are recorded independently; a faster kernel is not evidence of preserved quality.
"""
from contextlib import contextmanager
from types import MethodType
import json
from pathlib import Path
import torch
import soundfile as sf
import torchaudio.functional as AF
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from benchmarks.experimental_mm import gemv_rows
from benchmarks.compare import difference
from benchmarks.baseline import measure


@contextmanager
def experimental(model):
    saved=[];counter={"calls":0}
    def wrap(original):
        def forward(self,x):
            if (x.dtype==torch.float32 and x.numel()==self.in_features
                    and self.weight.numel()>=1024*1024 and self.weight.is_contiguous()):
                counter["calls"]+=1
                bn=4 if max(self.in_features,self.out_features)>=5120 else 2
                out=gemv_rows(x.reshape(1,-1).contiguous(),self.weight,bn)
                if self.bias is not None:out=out+self.bias
                return out.reshape(*x.shape[:-1],self.out_features)
            return original(x)
        return forward
    try:
        for module in model.modules():
            if isinstance(module,torch.nn.Linear):
                saved.append((module,module.forward,"forward" in module.__dict__))
                module.forward=MethodType(wrap(module.forward),module)
        yield counter
    finally:
        for module,original,existed in reversed(saved):
            if existed:module.forward=original
            else:del module.forward


@torch.inference_mode()
def main():
    model=load_model()
    inputs=[]
    for path in ["data/speech.wav","data/environment.wav","data/music.ogg"]:
        data,sr=sf.read(path,dtype="float32",always_2d=True)
        mono=torch.from_numpy(data).mean(1)
        if sr!=24000:mono=AF.resample(mono,sr,24000)
        for start in [0,19200,38400]:
            if start+1920<=mono.numel():
                inputs.append((f"{Path(path).stem}_{start}",mono[start:start+1920].view(1,1,-1).cuda()))
    torch.manual_seed(9)
    inputs.extend([("noise",torch.randn(1,1,1920,device="cuda")*.05),
                   ("silence",torch.zeros(1,1,1920,device="cuda"))])
    def run(x):
        enc=model._encode_frame(x)
        return enc.audio_codes,enc.encoder_hidden_states,model._decode_frame(enc.audio_codes).audio
    report={"scope":"experimental full-model FP32 GEMV; not promoted", "revision":REVISION,
            "torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"dtype":"float32","tf32":False,
            "quantizers":32,"cases":[]}
    with optimized(model,residual_backend="triton",rope_backend="triton",kv_backend="triton",share_rope_tables=True):
        for name,x in inputs:
            reference=run(x)
            with experimental(model) as count:
                candidate=run(x)
                record={"name":name,"gemv_calls":count["calls"],"comparisons":{
                    key:difference(r,c) for key,r,c in zip(["codes","hidden","audio"],reference,candidate)}}
                signal=reference[2].double().square().sum()
                error=(reference[2].double()-candidate[2].double()).square().sum()
                record["audio_snr_db"]=float(10*torch.log10(signal/error)) if error else None
                record["changed_codes_per_quantizer"]=(reference[0]!=candidate[0]).sum((1,2)).tolist()
            report["cases"].append(record)
            print(name,record,flush=True)
            Path("results/full_gemv_experiment.json").write_text(json.dumps(report,indent=2)+"\n")
        x=inputs[0][1]
        baseline=GraphedCallable(run,x)
        report["baseline_graph"]=measure(lambda:baseline(x),repeats=20)
        del baseline
        with experimental(model):
            candidate=GraphedCallable(run,x)
            report["experimental_graph"]=measure(lambda:candidate(x),repeats=20)
            del candidate
    report["all_codes_exact"]=all(r["comparisons"]["codes"]["exact"] for r in report["cases"])
    report["all_audio_exact"]=all(r["comparisons"]["audio"]["exact"] for r in report["cases"])
    Path("results/full_gemv_experiment.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
