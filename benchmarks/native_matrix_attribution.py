"""Attribute remaining vendor matrix kernels to eager ATen operations/shapes.

No hooks or forward replacements: the supported dispatch and FFN fusion remain
active. Eager profiling includes launch gaps; timings are attribution evidence,
not whole-codec graph performance measurements.
"""
import json
from pathlib import Path
from collections import defaultdict
import torch
from benchmarks.ordered_model import options
from benchmarks.lane_completion import audio_sources
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized


@torch.inference_mode()
def main():
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='triton',ffn_backend='triton',norm_backend='cuda')
    report={'scope':'warmed eager ATen/CUDA attribution, no observer hooks; not graph latency',
        'previous_commit':'c2e4722','revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'sources':sources,'options':opts,'records':[],
        'attribution_limit':'Only ATen operator entries establish operator/shape ownership. Runtime API associations for custom launches are retained as raw profiler evidence, not inferred operator ownership.'}
    for batch,frames in [(1,1),(1,3),(8,3)]:
        idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
        x=clips[1][idx%clips[1].numel()][:,None]
        with optimized(model,**opts):
            codes=model._encode_frame(x).audio_codes;model._decode_frame(codes)
            for direction,fn in [('encode',lambda:model._encode_frame(x)),('decode',lambda:model._decode_frame(codes))]:
                for _ in range(2):fn()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                    fn();torch.cuda.synchronize()
                events=[]
                for event in prof.events():
                    kernels=[k for k in event.kernels if any(word in k.name.lower() for word in ('gemm','gemv'))]
                    if not kernels:continue
                    parents=[];parent=event.cpu_parent
                    while parent is not None:
                        parents.append({'name':parent.name,'shapes':parent.input_shapes});parent=parent.cpu_parent
                    events.append({'name':event.name,'shapes':event.input_shapes,'parents':parents,
                        'kernels':[{'name':k.name,'duration_us':k.duration} for k in kernels]})
                groups=defaultdict(lambda:{'calls':0,'kernel_us':0.,'kernels':set()})
                for event in events:
                    key=json.dumps([event['name'],event['shapes']]);g=groups[key];g['calls']+=1
                    for k in event['kernels']:g['kernel_us']+=k['duration_us'];g['kernels'].add(k['name'])
                summary=[{'operator_and_shapes':json.loads(key),**{k:(sorted(v) if k=='kernels' else v) for k,v in values.items()}}
                    for key,values in groups.items()]
                summary.sort(key=lambda r:-r['kernel_us'])
                report['records'].append({'batch':batch,'frames':frames,'direction':direction,'events':events,'summary':summary,
                    'aten_summary':[r for r in summary if r['operator_and_shapes'][0].startswith('aten::')]} )
                print(batch,frames,direction,[(v['operator_and_shapes'],v['calls'],round(v['kernel_us'],2)) for v in summary],flush=True)
                Path('results/native_matrix_attribution.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
