"""Capture actual filled-ring inputs, then compare FP32 attention alternatives."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from triton.testing import do_bench_cudagraph
import soundfile as sf
import torchaudio.functional as AF
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession
from fast_moss.graphs import GraphedCallable
from benchmarks.compare import difference
from benchmarks.matrices import evicted_replay
from benchmarks.experimental_attention import split_attention


def copy_strides(t):
    result=torch.empty_strided(t.shape,t.stride(),device=t.device,dtype=t.dtype)
    return result.copy_(t)


@torch.inference_mode()
def capture(a):
    data,sr=sf.read(a.audio,dtype='float32',always_2d=True)
    mono=torch.from_numpy(data).mean(1)
    if sr!=24000:mono=AF.resample(mono,sr,24000)
    frames=max(128, math.ceil(125/a.chunk_frames)*a.chunk_frames+a.chunk_frames)
    if mono.numel()<frames*1920:raise ValueError(f'Need at least {frames*.08} seconds of audio')
    x=torch.stack([mono[:frames*1920].roll(i*1920) for i in range(a.batch)])[:,None].cuda()
    model=load_model();cases={};original=F.scaled_dot_product_attention
    collect=False;direction=''
    def wrapped(q,k,v,bias=None,**kwargs):
        if collect:
            shape=(direction,*q.shape,k.shape[2])
            if shape not in cases:
                cases[shape]={'inputs':tuple(copy_strides(z) for z in (q,k,v,bias)), 'occurrences':0}
            cases[shape]['occurrences']+=1
        return original(q,k,v,bias,**kwargs)
    with optimized(model,residual_backend='triton',rope_backend='triton',kv_backend='triton',
                   share_rope_tables=True,attention_mask_backend='triton'):
        codes=model._encode_frame(x).audio_codes
        with patch('torch.nn.functional.scaled_dot_product_attention',wrapped):
            for direction,inp in [('encode',x),('decode',codes)]:
                step=a.chunk_frames*(1920 if direction=='encode' else 1)
                with StreamingSession(model,direction,a.batch,a.chunk_frames) as session:
                    for index,part in enumerate(inp.split(step,dim=-1)):
                        if part.shape[-1]!=step:break
                        if index*a.chunk_frames>=125:
                            # Capture actual tensors after history has filled;
                            # switch to eager for this one call so Python sees them.
                            session.use_graph=False;collect=True
                            session.push(part)
                            collect=False
                            break
                        session.push(part)
    return cases


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audio',default='data/music.ogg')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--chunk-frames',type=int,default=1)
    p.add_argument('--output',default='results/attention_shapes.json')
    p.add_argument('--flex',action='store_true')
    p.add_argument('--flex-only',action='store_true')
    p.add_argument('--save-inputs')
    p.add_argument('--inputs', help='Reuse locally captured tensor cases')
    a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cases=torch.load(a.inputs,weights_only=True) if a.inputs else capture(a)
    if a.save_inputs:torch.save(cases,a.save_inputs)
    report={'scope':'actual full-checkpoint filled-ring attention components; not whole-model speedups',
            'revision':REVISION,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'dtype':'float32',
            'tf32':False,'audio':a.audio,'audio_sha256':hashlib.sha256(Path(a.audio).read_bytes()).hexdigest(),
            'batch':a.batch,'chunk_frames':a.chunk_frames,'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    original=F.scaled_dot_product_attention
    for shape,case in cases.items():
        q,k,v,bias=case['inputs']
        reference=original(q,k,v,bias)
        gold=torch.softmax((q.double()@k.double().transpose(-1,-2))*(q.shape[-1]**-.5)+bias.double(),-1)@v.double()
        def math_sdpa(q,k,v,bias):
            with sdpa_kernel(SDPBackend.MATH):return original(q,k,v,bias)
        candidates={'sdpa':original,'math_sdpa':math_sdpa}
        for block in [64,128,256]:
            candidates[f'split_{block}']=lambda q,k,v,bias,block=block:split_attention(q,k,v,bias,block)
        if a.flex or a.flex_only:
            if a.flex_only:candidates={'sdpa':original}
            from torch.nn.attention.flex_attention import flex_attention
            def flex_candidate(q,k,v,bias):
                bias3=bias[:,0]
                def score_mod(score,b,h,t,s):
                    return score+torch.ops.aten.index.Tensor(bias3,[b,t,s])
                return flex_attention(q,k,v,score_mod=score_mod,
                                      kernel_options={'FLOAT32_PRECISION':"'ieee'",'PRESCALE_QK':False})
            candidates['flex_ieee']=torch.compile(flex_candidate,fullgraph=True)
        for name,fn in candidates.items():
            out=fn(q,k,v,bias)
            graph=GraphedCallable(lambda q,k,v,bias:(fn(q,k,v,bias),),q,k,v,bias)
            record={'shape_direction_BHTDS':shape,'occurrences':case['occurrences'],'backend':name,
                    'reference_difference':difference(reference,out),
                    'graph_vs_eager':difference(out,graph(q,k,v,bias)[0]),
                    'fp64_max_abs':(out.double()-gold).abs().max().item(),
                    'reference_fp64_max_abs':(reference.double()-gold).abs().max().item(),
                    'hot_graph_ms':do_bench_cudagraph(lambda:fn(q,k,v,bias),rep=20,return_mode='median'),
                    'evicted_replay':evicted_replay(graph,flush)}
            report['records'].append(record)
            print(shape,name,'us',round(record['hot_graph_ms']*1000,2),'max diff',record['reference_difference']['max_abs'],flush=True)
            del graph
            Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    if not cases:raise RuntimeError('No attention cases captured')


if __name__=='__main__':main()
