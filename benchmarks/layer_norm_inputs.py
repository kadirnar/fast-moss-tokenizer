"""Capture actual checkpoint LayerNorm geometries without timing instrumented work."""
import json
from pathlib import Path
import torch
from fast_moss.loading import load_model,REVISION
from benchmarks.lane_completion import audio_sources


@torch.inference_mode()
def main():
    model=load_model();clips,sources=audio_sources();captured={};handles=[];calls=[]
    def hook(name):
        def observe(module,args,out):
            x=args[0];n=module.normalized_shape[0];shape=(x.numel()//n,n)
            calls.append((name,shape))
            if shape not in captured:
                captured[shape]={'x':x.reshape(shape).contiguous().clone(),'weight':module.weight.clone(),
                    'bias':module.bias.clone(),'eps':module.eps,'name':name,'native_shape':list(x.shape),
                    'native_stride':list(x.stride()),'output':out.reshape(shape).contiguous().clone()}
        return observe
    for name,module in model.named_modules():
        if type(module) is torch.nn.LayerNorm and tuple(module.normalized_shape) in ((768,),(1280,)):
            handles.append(module.register_forward_hook(hook(name)))
    report={'scope':'actual FP32 LayerNorm operand capture; no instrumented timing',
            'revision':REVISION,'previous_commit':'52a2c91','sources':sources,'cases':[]}
    try:
        for batch,frames in [(1,1),(8,1),(1,3),(1,40),(128,1)]:
            idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
            x=clips[1][idx%clips[1].numel()][:,None];calls.clear()
            codes=model._encode_frame(x).audio_codes;model._decode_frame(codes)
            report['cases'].append({'batch':batch,'frames':frames,'calls':len(calls),
                'shapes':sorted(set(s for _,s in calls))})
    finally:
        for h in handles:h.remove()
    torch.save(captured,'results/layer_norm_inputs.pt')
    report['captured']=[{'shape':s,**{k:v for k,v in r.items() if not isinstance(v,torch.Tensor)}} for s,r in captured.items()]
    Path('results/layer_norm_inputs.json').write_text(json.dumps(report,indent=2)+'\n')
    print('captured',len(captured),'geometries',list(captured))

if __name__=='__main__':main()
