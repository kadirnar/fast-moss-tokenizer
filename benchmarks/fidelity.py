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
        if path.stem=='speech':
            # This geometry exposes a quantizer boundary only 1.19e-7 apart in
            # the original FP32 calculation; tiny latent errors can cascade.
            idx=torch.arange(5760)[None]+torch.arange(8)[:,None]*1920
            yield 'speech_near_tie_b8_f3',x[idx%x.numel()][:,None],{
                **entry,'transform':'eight cyclic lanes, 5760 samples each, starts offset by 1920 samples',
                'regression':'first observed TF32x3 code flip at quantizer 5, lane 6, frame 2 (zero-based)'}
            for batch in [24, 128]:
                idx=torch.arange(1920)[None]+torch.arange(batch)[:,None]*1920
                yield f'speech_singleton_b{batch}_f1',x[idx%x.numel()][:,None],{
                    **entry,'transform':f'{batch} cyclic lanes, 1920 samples each, starts offset by 1920 samples',
                    'regression':'canonical residual output strides must preserve downstream matrix dispatch'}
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
    p.add_argument("--share-rope-tables",action="store_true")
    p.add_argument("--attention-mask-backend", choices=["none", "triton"], default="none")
    p.add_argument("--quantizer-backend", choices=["none", "triton"], default="none")
    p.add_argument("--matrix-backend", choices=["none", "cublaslt", "triton"], default="none")
    p.add_argument("--projection-backend", choices=["none", "triton"], default="none")
    a=p.parse_args()
    model=load_model()
    def run(x):
        enc=model._encode_frame(x)
        dec=model._decode_frame(enc.audio_codes)
        return enc.audio_codes,enc.encoder_hidden_states,dec.audio
    report={"scope":"full checkpoint, small regression corpus", "revision":REVISION,
            "torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"dtype":"float32","tf32":False,
            "quantizers":32,"projection_backend":a.projection_backend,"quantizer_backend":a.quantizer_backend,"matrix_backend":a.matrix_backend,"residual_backend":a.backend,"rope_backend":a.rope_backend,
            "share_rope_tables":a.share_rope_tables,"attention_mask_backend":a.attention_mask_backend,
            "attention_mask_format":"aligned_fp32_additive" if a.attention_mask_backend=="triton" else "upstream_boolean","cases":[]}
    for name,cpu,source in cases():
        x=cpu.cuda()
        reference=run(x)
        record={"name":name,"samples":x.numel(),"input_shape":list(x.shape),"source":source,"comparisons":{}}
        with optimized(model,residual_backend=a.backend,rope_backend=a.rope_backend,kv_backend="triton",
                       share_rope_tables=a.share_rope_tables,attention_mask_backend=a.attention_mask_backend,
                       quantizer_backend=a.quantizer_backend,matrix_backend=a.matrix_backend,projection_backend=a.projection_backend):
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
