"""Full-checkpoint ablation of eight-channel projection and decoder table fusion."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.lane_completion import audio_sources
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--output',default='results/full_projection_runtime.json')
    a=p.parse_args()
    model=load_model(); clips,sources=audio_sources()
    report={'scope':'full checkpoint; paired projection backend ablation against supported matrix/quantizer runtime',
            'revision':REVISION,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'cudnn':torch.backends.cudnn.version(),'dtype':'float32','tf32':False,'quantizers':32,
            'sources':sources,'input':'cyclic speech, lanes offset by 1920 samples',
            'timing_scope':'warmed graphs, including input copies and owned outputs; excludes setup, cache building, packing, capture, and restoration',
            'cases':[]}
    for b,t in [(1,1),(1,3),(8,3),(128,3),(8,40)]:
        idx=torch.arange(t*1920,device='cuda')[None]+torch.arange(b,device='cuda')[:,None]*1920
        x=clips[1][idx%clips[1].numel()][:,None]
        original=model._encode_frame(x)
        codes=original.audio_codes
        refs={'encode':codes,'decode':model._decode_frame(codes).audio,
              'decode_quantizer':model.quantizer.decode_codes(codes)}
        rounds=[]
        for repeat in range(a.rounds):
            for backend in (['none','triton'] if repeat%2==0 else ['triton','none']):
                torch.cuda.reset_peak_memory_stats()
                with optimized(model,residual_backend='triton',kv_backend='triton',rope_backend='triton',
                               share_rope_tables=True,attention_mask_backend='triton',quantizer_backend='triton',
                               matrix_backend='cublaslt',projection_backend=backend):
                    record={'round':repeat,'backend':backend,'results':{}}
                    for name,fn,inp in [('encode',lambda z:(model._encode_frame(z).audio_codes,),x),
                                        ('decode_quantizer',lambda z:(model.quantizer.decode_codes(z),),codes),
                                        ('decode',lambda z:(model._decode_frame(z).audio,),codes)]:
                        eager=fn(inp)[0]
                        graph=GraphedCallable(fn,inp)
                        result=graph(inp)[0]
                        record['results'][name]={'eager':difference(refs[name],eager),'graph':difference(refs[name],result),
                                                 'timing':measure(lambda:graph(inp),repeats=20)}
                        del graph,eager,result
                    record['projected_cache_bytes']=(model.quantizer._fast_projected_codebooks.numel()*4 if backend=='triton' else 0)
                record['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                record['all_exact']=all(v[mode]['exact'] for v in record['results'].values() for mode in ['eager','graph'])
                rounds.append(record)
                print(b,t,repeat,backend,{name:round(v['timing']['wall_ms_median'],3) for name,v in record['results'].items()},'exact',record['all_exact'],flush=True)
        medians={scope:{backend:statistics.median(r['results'][scope]['timing']['wall_ms_median'] for r in rounds if r['backend']==backend)
                        for backend in ['none','triton']} for scope in refs}
        case={'batch':b,'frames':t,'rounds':rounds,'wall_ms_medians':medians,
              'speedups':{scope:v['none']/v['triton'] for scope,v in medians.items()},
              'all_exact':all(r['all_exact'] for r in rounds)}
        report['cases'].append(case)
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=all(c['all_exact'] for c in report['cases'])
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Projection fidelity failed')


if __name__=='__main__':main()
