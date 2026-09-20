"""Amortized full-model graph comparison separating pipeline and epilogue gains."""
import json
from pathlib import Path
import statistics
import torch
from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from benchmarks.ffn_components import staged
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
import fast_moss.ordered_matrices as ordered
import fast_moss.ffn as ffn


@torch.inference_mode()
def main():
    model=load_model();clips,sources=audio_sources()
    idx=torch.arange(5760,device='cuda')[None]+torch.arange(8,device='cuda')[:,None]*1920
    x=clips[1][idx%clips[1].numel()][:,None]
    enc=model._encode_frame(x);codes=enc.audio_codes
    refs={'encode':(codes,enc.encoder_hidden_states),'decode':(model._decode_frame(codes).audio,)}
    report={'scope':'full checkpoint batch eight / three frames, eight rotating-order groups of 20 owned graph calls',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'sources':sources,
            'dtype':'float32','tf32':False,'quantizers':32,'replays_per_sample':20,'results':{},
            'timing_scope':'mean latency within each 20-call group, includes input copies and owned outputs, one weight lifetime and five graphs per direction; excludes load/packing/capture/restoration'}
    native,fused=ordered.linear,ffn.linear
    try:
        with optimized(model,**dict(options(),matrix_backend='triton',ffn_backend='triton')):
            runtime=model._fast_matrix_runtime
            layers=[m for m in model.modules() if hasattr(m,'_fast_ffn_fuse')]
            selected={id(m):m._fast_ffn_fuse for m in layers}
            def encode(z):
                out=model._encode_frame(z)
                return out.audio_codes,out.encoder_hidden_states
            funcs=[('encode',encode,x),('decode',lambda z:(model._decode_frame(z).audio,),codes)]
            for _,fn,inp in funcs:fn(inp)
            for name,fn,inp in funcs:
                graphs={};checks={}
                for mode in ['previous','stages2','fused_stages3','fused_stages2','selected']:
                    runtime.ffn_enabled=mode.startswith('fused') or mode=='selected'
                    for layer in layers:layer._fast_ffn_fuse=selected[id(layer)] if mode=='selected' else True
                    ordered.linear=staged if mode=='stages2' else native
                    ffn.linear=(lambda *args: fused(*args,stages=3)) if mode=='fused_stages3' else fused
                    graphs[mode]=GraphedCallable(fn,inp)
                    checks[mode]=[difference(ref,out) for ref,out in zip(refs[name],graphs[mode](inp))]
                ordered.linear,ffn.linear=native,fused
                def run(graph):
                    for _ in range(20):graph(inp)
                samples=[]
                names=list(graphs)
                for repeat in range(8):
                    order=names[repeat%len(names):]+names[:repeat%len(names)]
                    if repeat%2:order.reverse()
                    for mode in order:
                        timing=measure(lambda:run(graphs[mode]),warmup=1,repeats=1)
                        samples.append({'round':repeat,'mode':mode,'wall_ms_per_call':timing['wall_ms_median']/20,
                                        'gpu_ms_per_call':timing['gpu_ms_median']/20})
                medians={mode:statistics.median(r['wall_ms_per_call'] for r in samples if r['mode']==mode) for mode in graphs}
                report['results'][name]={'checks':checks,'samples':samples,'wall_ms_medians':medians,
                    'speedups':{mode:medians['previous']/ms for mode,ms in medians.items()}}
                print(name,report['results'][name]['speedups'],medians,flush=True)
                graphs.clear()
                Path('results/full_ffn_ablation.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        ordered.linear,ffn.linear=native,fused
    report['all_exact']=all(c['exact'] for r in report['results'].values() for checks in r['checks'].values() for c in checks)
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    Path('results/full_ffn_ablation.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('FFN ablation fidelity failed')


if __name__=='__main__':main()
