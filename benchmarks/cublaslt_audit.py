"""Audit every candidate linear against original-layout FP32 on identical inputs."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.cublaslt_model import choices,experimental
from benchmarks.lane_completion import audio_sources
from benchmarks.compare import difference
from fast_moss.loading import load_model
from fast_moss.optimize import optimized


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tuning',nargs='+',default=['results/matrices_cublaslt.json','results/matrices_cublaslt_codec8.json','results/matrices_cublaslt_large.json'])
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--frames',type=int,default=1)
    p.add_argument('--output',default='results/cublaslt_layer_audit.json')
    a=p.parse_args();model=load_model();clips,sources=audio_sources();selected=choices(a.tuning,packed_only=True)
    report={'scope':'per-layer same-input audit; returns reference output to avoid propagated differences',
            'batch':a.batch,'frames':a.frames,'sources':sources,'checks':[]}
    failures={};hooks=[];source=None
    def audit(name):
        def hook(module,args,out):
            x=args[0];shape=(x.numel()//x.shape[-1],*module.weight.shape)
            if shape not in selected:return
            reference=F.linear(x,module.weight.contiguous(),module.bias)
            check=difference(reference,out)
            report['checks'].append({'layer':name,'source':source,'shape_MNK':shape,
                                     'backend':selected[shape]['backend'],'difference':check,
                                     'input_stride':list(x.stride())})
            if not check['exact']:
                key=(name,source)
                failures[key]={'shape':shape,'x':x.detach().reshape(shape[0],shape[2]).contiguous().clone(),
                               'weight':module.weight.detach().contiguous().clone()}
                print('mismatch',name,shape,check,flush=True)
            return reference
        return hook
    with optimized(model,residual_backend='triton',rope_backend='triton',kv_backend='triton',
                   share_rope_tables=True,attention_mask_backend='triton'):
        with experimental(model,selected,resident=True):
            try:
                for name,module in model.named_modules():
                    if type(module) is torch.nn.Linear:hooks.append(module.register_forward_hook(audit(name)))
                for clip,meta in zip(clips,sources):
                    source=meta['path']
                    idx=torch.arange(a.frames*1920,device='cuda')[None]+torch.arange(a.batch,device='cuda')[:,None]*1920
                    x=clip[idx%clip.numel()][:,None]
                    codes=model._encode_frame(x).audio_codes
                    model._decode_frame(codes)
            finally:
                for hook in hooks:hook.remove()
    report['all_exact']=all(r['difference']['exact'] for r in report['checks'])
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    if failures:torch.save(failures,Path(a.output).with_suffix('.pt'))


if __name__=='__main__':main()
