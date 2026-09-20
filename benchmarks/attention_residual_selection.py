"""Rotating decoder ablation to resolve batch-eight attention epilogue selection."""
import json,statistics
from pathlib import Path
import torch
from benchmarks.attention_residual import CONFIGS
from benchmarks.attention_residual_model import selected,compare
from benchmarks.baseline import measure
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized

@torch.inference_mode()
def main():
    model=load_model();clips,sources=audio_sources();idx=torch.arange(1920,device='cuda')[None]+torch.arange(8,device='cuda')[:,None]*1920
    x=clips[1][idx%clips[1].numel()][:,None];codes=model._encode_frame(x).audio_codes
    fn=lambda z:(model._decode_frame(z).audio,);ref=fn(codes)
    opts=dict(options(),matrix_backend='cuda',ffn_backend='triton',norm_backend='cuda')
    configs={'current':{},'all':{json.dumps(s):c for s,c in CONFIGS.items()},
        'width1280':{json.dumps(s):c for s,c in CONFIGS.items() if s[1]==1280},
        'width768':{json.dumps(s):c for s,c in CONFIGS.items() if s[1]==768}}
    report={'scope':'research batch-eight one-frame decoder, rotating four-way projection selection, full-checkpoint reference',
        'previous_commit':'f104647','revision':REVISION,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'sources':sources,'options':opts,'timing_scope':'nine rotating rounds, 200 owned graph warmups, nine samples of ten graph calls; input copies and owned outputs included; loading/capture/setup excluded',
        'configs':configs,'rounds':[]}
    for repeat in range(9):
        names=list(configs);names=names[repeat%4:]+names[:repeat%4]
        for name in names:
            with optimized(model,**opts),selected(model,configs[name]) as counts:
                graph=GraphedCallable(fn,codes);checks=compare(ref,graph(codes));assert all(c['bits_equal'] for c in checks)
                for _ in range(200):graph(codes)
                torch.cuda.synchronize()
                def many():
                    for _ in range(10):graph(codes)
                timing=measure(many,warmup=2,repeats=9);samples=[v/10 for v in timing['wall_ms_samples']]
                del graph
            report['rounds'].append({'round':repeat,'variant':name,'samples_ms':samples,'checks':checks,**counts})
            print(repeat,name,statistics.median(samples),counts,flush=True)
        report['medians_ms']={name:statistics.median(v for row in report['rounds'] if row['variant']==name for v in row['samples_ms']) for name in configs}
        report['speedups']={name:report['medians_ms']['current']/v for name,v in report['medians_ms'].items()}
        Path('results/attention_residual_selection.json').write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=True;Path('results/attention_residual_selection.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
