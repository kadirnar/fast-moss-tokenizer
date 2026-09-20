"""Full-checkpoint streaming across the ten-second context, including batched lanes."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import soundfile as sf
import torchaudio.functional as AF
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession
from benchmarks.compare import difference


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--audio",default="data/music.ogg")
    p.add_argument("--frames",type=int,default=160)
    p.add_argument("--chunk-frames",type=int,default=4)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--output",default="results/full_streaming_fidelity.json")
    p.add_argument("--share-rope-tables",action="store_true")
    p.add_argument("--attention-mask-backend", choices=["none", "triton"], default="none")
    p.add_argument("--matrix-backend", choices=["none", "cublaslt", "triton", "cuda"], default="none")
    for name in ['quantizer','projection','ffn']:
        p.add_argument(f"--{name}-backend", choices=["none", "triton"], default="none")
    p.add_argument("--norm-backend", choices=["none", "cuda"], default="none")
    a=p.parse_args()
    extra={f'{name}_backend':getattr(a,f'{name}_backend') for name in ['matrix','quantizer','projection','ffn']}
    extra['norm_backend']=a.norm_backend
    if a.frames%a.chunk_frames:
        raise ValueError("Frame count must divide into complete chunks")
    samples,sr=sf.read(a.audio,dtype="float32",always_2d=True)
    mono=torch.from_numpy(samples).mean(1)
    if sr!=24000:
        mono=AF.resample(mono,sr,24000)
    length=a.frames*1920
    if mono.numel()<length:
        raise ValueError("Recording is too short")
    x=torch.stack([mono[:length].roll(1920*i) for i in range(a.batch)])[:,None].cuda()
    model=load_model()
    codes=model.encode(x,return_dict=True).audio_codes
    audio=model.decode(codes,return_dict=True).audio
    report={"scope":"full checkpoint, long streaming, small music regression",
            "revision":REVISION,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
            "dtype":"float32","tf32":False,"quantizers":32,"batch":a.batch,
            "frames":a.frames,"chunk_frames":a.chunk_frames,"seconds":length/24000,
            "share_rope_tables":a.share_rope_tables,"attention_mask_backend":a.attention_mask_backend,
            "attention_mask_format":"aligned_fp32_additive" if a.attention_mask_backend=="triton" else "upstream_boolean",
            "audio":a.audio,"audio_sha256":hashlib.sha256(Path(a.audio).read_bytes()).hexdigest(),
            "lane_transform":"lane i circularly shifted by i*1920 samples","results":{}}
    report.update(extra)
    for direction,inp,offline in [("encode",x,codes),("decode",codes,audio)]:
        chunk=a.chunk_frames*(1920 if direction=="encode" else 1)
        inputs=list(inp.split(chunk,dim=-1))
        with StreamingSession(model,direction,a.batch,a.chunk_frames,use_graph=False,fast_reset=False) as session:
            reference=[session.push(part)[0].clone() for part in inputs]
        print(direction,"reference finished",flush=True)
        with optimized(model,residual_backend="triton",kv_backend="triton",rope_backend="triton",
                       share_rope_tables=a.share_rope_tables,attention_mask_backend=a.attention_mask_backend,**extra):
            with StreamingSession(model,direction,a.batch,a.chunk_frames) as session:
                actual=[session.push(part)[0] for part in inputs]
            runtime=getattr(model,'_fast_matrix_runtime',None)
            small_calls=getattr(runtime,'small_calls',0)
            norm_projection_calls=getattr(runtime,'norm_gemv_calls',0)
            norm_async_calls=getattr(runtime,'norm_async_calls',0)
            residual_async_calls=getattr(runtime,'residual_async_calls',0)
            quantizer_prepare_calls=getattr(getattr(model,'_fast_norm_runtime',None),'quantizer_prepare_calls',0)
            attention_residual_calls=getattr(runtime,'attention_residual_calls',0)
        ref=torch.cat(reference,dim=-1)
        cand=torch.cat(actual,dim=-1)
        result={"graph_vs_corrected_eager":difference(ref,cand),
                "corrected_eager_vs_offline":difference(offline,ref),
                "graph_vs_offline":difference(offline,cand),
                "per_chunk_exact":[torch.equal(r,c) for r,c in zip(reference,actual)],
                "small_matrix_calls":small_calls,"norm_projection_calls":norm_projection_calls,
                "attention_residual_calls":attention_residual_calls,"norm_async_calls":norm_async_calls,
                "residual_async_calls":residual_async_calls,
                "quantizer_prepare_calls":quantizer_prepare_calls,
                "per_chunk_bits_equal":[torch.equal(r.view(torch.int32),c.view(torch.int32)) if r.dtype==torch.float32 else torch.equal(r,c) for r,c in zip(reference,actual)]}
        report["results"][direction]=result
        print(direction,result,flush=True)
        Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    report["graph_exact"]=all(r["graph_vs_corrected_eager"]["exact"] and all(r["per_chunk_bits_equal"]) for r in report["results"].values())
    report["offline_exact"]=all(r["graph_vs_offline"]["exact"] for r in report["results"].values())
    report["peak_memory_bytes"]=torch.cuda.max_memory_allocated()
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    if not report["graph_exact"]:
        raise SystemExit("Optimized streaming differs from corrected eager reference")


if __name__=="__main__":
    main()
