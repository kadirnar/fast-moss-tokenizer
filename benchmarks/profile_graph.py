"""Profile the current optimized full-model CUDA graphs to find remaining costs."""
from contextlib import ExitStack
import math
import argparse
import json
import gzip
from pathlib import Path
import torch
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from fast_moss.streaming import StreamingSession
from benchmarks.baseline import measure


def kernel_summary(path,replays=5):
    trace=json.loads(gzip.open(path,'rt').read())
    events=[e for e in trace['traceEvents'] if e.get('cat')=='kernel']
    total=sum(e['dur'] for e in events)
    groups={name:sum(e['dur'] for e in events if needle in e.get('name','').lower())
            for name,needle in [('sgemm','sgemm'),('attention','fmha')]}
    # "cublasGemmSmallNParams" itself contains "sgemm". Exclude the explicit
    # small-matrix/reduction families so the matrix groups do not overlap.
    groups['sgemm']=sum(e['dur'] for e in events if 'sgemm' in e.get('name','').lower()
        and not any(needle in e.get('name','').lower() for needle in ['gemmsn','gemvx','splitkreduce']))
    groups['ordered_matrix']=sum(e['dur'] for e in events if e.get('name') in ['_partials','_reduce','_ffn_reduce'])
    small_names=['norm_gemv','_strided_ffn_fixed','strided_ffn_cuda','_short_fixed','short_ffn_cuda','wide_cta','cta_tiled','_small_gemv','_ffn_gemv','_small_fixed','_small_grouped','_small_parts','_small_reduce']
    small_times={name:sum(e['dur'] for e in events if e.get('name')==name) for name in small_names}
    groups['small_matrix']=sum(small_times.values())
    groups['small_vendor_matrix']=sum(e['dur'] for e in events
        if any(needle in e.get('name','').lower() for needle in ['gemmsn','gemvx']))
    groups['layer_norm_cuda']=sum(e['dur'] for e in events if e.get('name')=='layer_norm')
    groups['layer_norm_native']=sum(e['dur'] for e in events if 'vectorized_layer_norm_kernel' in e.get('name',''))
    groups['vendor_split_reduce']=sum(e['dur'] for e in events if 'splitkreduce' in e.get('name','').lower())
    return {'scope':'CUDA kernel events only, excludes memcpy and host time','replays':replays,
            'matrix_grouping':'SGEMM excludes the separately counted gemmSN/GEMV and vendor split reductions',
            'total_ms_per_replay':total/replays/1000,'kernels_per_replay':len(events)/replays,
            'quantizer_select_per_replay':sum(e.get('name')=='_select' for e in events)/replays,
            'quantizer_update_per_replay':sum(e.get('name')=='_update' for e in events)/replays,
            'projection_update_per_replay':sum(e.get('name')=='_project_update' for e in events)/replays,
            'decoder_gather_per_replay':sum(e.get('name') in ['_decode','_decode_tokens'] for e in events)/replays,
            'ordered_partials_per_replay':sum(e.get('name')=='_partials' for e in events)/replays,
            'ordered_reduce_per_replay':sum(e.get('name')=='_reduce' for e in events)/replays,
            'ffn_reduce_per_replay':sum(e.get('name')=='_ffn_reduce' for e in events)/replays,
            'strided_ffn_fixed_per_replay':sum(e.get('name')=='_strided_ffn_fixed' for e in events)/replays,
            'strided_ffn_cuda_per_replay':sum(e.get('name')=='strided_ffn_cuda' for e in events)/replays,
            'short_ffn_fixed_per_replay':sum(e.get('name')=='_short_fixed' for e in events)/replays,
            'short_ffn_cuda_per_replay':sum(e.get('name')=='short_ffn_cuda' for e in events)/replays,
            'wide_cta_per_replay':sum(e.get('name')=='wide_cta' for e in events)/replays,
            'cuda_cta_per_replay':sum(e.get('name')=='cta_tiled' for e in events)/replays,
            'small_gemv_per_replay':sum(e.get('name')=='_small_gemv' for e in events)/replays,
            'layer_norm_cuda_per_replay':sum(e.get('name')=='layer_norm' for e in events)/replays,
            'layer_norm_native_per_replay':sum('vectorized_layer_norm_kernel' in e.get('name','') for e in events)/replays,
            'norm_projection_per_replay':sum(e.get('name')=='norm_gemv' for e in events)/replays,
            'ffn_gemv_per_replay':sum(e.get('name')=='_ffn_gemv' for e in events)/replays,
            'small_fixed_per_replay':sum(e.get('name')=='_small_fixed' for e in events)/replays,
            'small_grouped_per_replay':sum(e.get('name')=='_small_grouped' for e in events)/replays,
            'small_parts_per_replay':sum(e.get('name')=='_small_parts' for e in events)/replays,
            'small_reduce_per_replay':sum(e.get('name')=='_small_reduce' for e in events)/replays,
            'small_matrix_breakdown_scope':'Already included in groups.small_matrix; do not add again. Normalization/projection includes fused LayerNorm; FFN GEMV, short-row and strided FFN kernels include their fused epilogues.',
            'small_matrix_breakdown':{name:{'ms_per_replay':value/replays/1000,
                'percent':100*value/total if total else 0.} for name,value in small_times.items()},
            'groups':{name:{'ms_per_replay':value/replays/1000,'percent':100*value/total if total else 0.}
                      for name,value in groups.items()}}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seconds",type=float,default=.24)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--warmup-replays", type=int, default=0, help="Extra graph replays before measurement/profile; reported separately")
    p.add_argument("--output",default="results/full_graph_profile.json")
    p.add_argument("--share-rope-tables",action="store_true")
    p.add_argument("--attention-mask-backend", choices=["none", "triton"], default="none")
    p.add_argument("--matrix-tuning",nargs="+",help="Experimental resident FP32 matrix tuning reports")
    p.add_argument("--quantizer-backend", choices=["none", "triton"], default="none")
    p.add_argument("--matrix-backend", choices=["none", "cublaslt", "triton", "cuda"], default="none")
    p.add_argument("--projection-backend", choices=["none", "triton"], default="none")
    p.add_argument("--ffn-backend", choices=["none", "triton"], default="none")
    p.add_argument("--norm-backend", choices=["none", "cuda"], default="none")
    a=p.parse_args()
    if a.warmup_replays < 0:p.error("Warmup replays must be nonnegative")
    if a.matrix_tuning and a.matrix_backend != 'none':
        p.error('Choose either experimental tuning or the supported matrix backend')
    torch.manual_seed(2026)
    model=load_model()
    selected=None
    if a.matrix_tuning:
        from benchmarks.cublaslt_model import choices,experimental
        selected=choices(a.matrix_tuning,packed_only=True)
    x=torch.randn(a.batch,1,round(a.seconds*24000),device="cuda")*.05
    report={"scope":"full checkpoint optimized graph", "revision":REVISION,"torch":torch.__version__,
            "gpu":torch.cuda.get_device_name(),"dtype":"float32","tf32":False,"quantizers":32,
            "seconds":a.seconds,"batch":a.batch,"streaming":a.streaming,"extra_warmup_replays":a.warmup_replays,"share_rope_tables":a.share_rope_tables,"attention_mask_backend":a.attention_mask_backend,
            "quantizer_backend":a.quantizer_backend,"matrix_backend":a.matrix_backend,"projection_backend":a.projection_backend,"ffn_backend":a.ffn_backend,"norm_backend":a.norm_backend,"experimental_resident_matrices":bool(a.matrix_tuning),"matrix_tuning":a.matrix_tuning,
            "attention_mask_format":"aligned_fp32_additive" if a.attention_mask_backend=="triton" else "upstream_boolean","results":{}}
    with optimized(model,residual_backend="triton",rope_backend="triton",kv_backend="triton",
                   share_rope_tables=a.share_rope_tables,attention_mask_backend=a.attention_mask_backend,
                   quantizer_backend=a.quantizer_backend,matrix_backend=a.matrix_backend,projection_backend=a.projection_backend,ffn_backend=a.ffn_backend,norm_backend=a.norm_backend):
        codes=model._encode_frame(x).audio_codes
        for name,fn,inp in [("encode",lambda v:(model._encode_frame(v).audio_codes,),x),
                            ("decode",lambda v:(model._decode_frame(v).audio,),codes)]:
            with ExitStack() as stack:
                if selected is not None:stack.enter_context(experimental(model,selected,resident=True))
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
                for _ in range(a.warmup_replays):run()
                torch.cuda.synchronize()
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
                report["results"][name]['kernel_summary']=kernel_summary(Path(a.output).with_suffix(f".{name}.json.gz"))
                print(name,report["results"][name],flush=True)
                if not a.streaming:
                    del graph
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
