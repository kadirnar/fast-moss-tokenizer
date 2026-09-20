"""Capture noncontiguous LayerNorm inputs through the unchanged runtime wrapper."""
import json
from pathlib import Path
import torch
from fast_moss.loading import load_model, REVISION
from fast_moss.normalization import NormalizationRuntime
from fast_moss.optimize import optimized
from benchmarks.ordered_model import options
from benchmarks.lane_completion import audio_sources


@torch.inference_mode()
def main():
    model=load_model();clips,sources=audio_sources();captured={};counts={}
    names={id(m):n for n,m in model.named_modules()}
    original=NormalizationRuntime._wrap
    def wrap(runtime,native):
        forward=original(runtime,native)
        def observe(module,x):
            out=forward(module,x)
            if not x.is_contiguous() and x.shape[-1] in (768,1280):
                key=(tuple(x.shape),tuple(x.stride()))
                counts[key]=counts.get(key,0)+1
                if key not in captured:
                    value=torch.empty_strided(x.shape,x.stride(),device=x.device,dtype=x.dtype);value.copy_(x)
                    captured[key]={'x':value,'weight':module.weight.clone(),'bias':module.bias.clone(),
                        'eps':module.eps,'name':names[id(module)],'output':out.clone()}
            return out
        return observe
    opts=dict(options(),matrix_backend='triton',ffn_backend='triton',norm_backend='cuda')
    report={'scope':'actual optimized-runtime strided LayerNorm capture, no instrumented timing',
        'revision':REVISION,'previous_commit':'6bdd709','sources':sources,'options':opts,'cases':[]}
    NormalizationRuntime._wrap=wrap
    try:
        with optimized(model,**opts):
            for batch,frames in [(1,1),(8,1),(1,3),(8,3),(2,3),(1,40),(128,1)]:
                idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
                x=clips[1][idx%clips[1].numel()][:,None];before=sum(counts.values())
                code=model._encode_frame(x).audio_codes;model._decode_frame(code)
                report['cases'].append({'batch':batch,'frames':frames,'strided_calls':sum(counts.values())-before})
    finally:NormalizationRuntime._wrap=original
    torch.save(captured,'results/strided_layer_norm_inputs.pt')
    report['records']=[{'shape':key[0],'stride':key[1],'calls':counts[key],
        **{k:v for k,v in row.items() if not isinstance(v,torch.Tensor)}} for key,row in captured.items()]
    Path('results/strided_layer_norm_inputs.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)

if __name__=='__main__':main()
