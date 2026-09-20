"""Research-only cyclic FP32 FMA and ordered lane/vector reduction probe."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from benchmarks.compare import difference
from fast_moss.loading import strict_precision


@triton.jit
def _fold(a, AXIS:tl.constexpr, SIZE:tl.constexpr, ORDER:tl.constexpr):
    if AXIS == 1:
        lane=tl.arange(0,SIZE)[None,:,None]
    else:
        lane=tl.arange(0,SIZE)[None,None,:]
    if ORDER == 'serial':
        acc=tl.sum(tl.where(lane==0,a,0.),axis=AXIS)
        for i in tl.static_range(1,SIZE):
            acc=acc+tl.sum(tl.where(lane==i,a,0.),axis=AXIS)
        return acc
    else:
        for i in tl.static_range(0,tl.constexpr(SIZE.bit_length()-1)):
            delta=(SIZE//2 >> i) if ORDER=='down' else (1 << i)
            index=tl.broadcast_to((lane+delta)%SIZE,a.shape)
            a=a+tl.gather(a,index,axis=AXIS)
        return tl.sum(tl.where(lane==0,a,0.),axis=AXIS)


@triton.jit
def _cyclic(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            LANES:tl.constexpr,VEC:tl.constexpr,MODE:tl.constexpr,
            LORDER:tl.constexpr,VORDER:tl.constexpr,BN:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    m=tl.program_id(1)
    lane=tl.arange(0,LANES)
    if MODE=='single':
        acc=tl.full((BN,LANES),0,tl.float32)
        for block in range(tl.cdiv(K,LANES*VEC)):
            for v in tl.static_range(VEC):
                k=block*LANES*VEC+lane*VEC+v
                x=tl.load(X+m*K+k,k<K,0)
                w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
                acc=tl.fma(x[None,:],w,acc)
        out=tl.reshape(_fold(acc[:,:,None],1,LANES,LORDER),(BN,))
    else:
        vec=tl.arange(0,VEC)
        acc=tl.full((BN,LANES,VEC),0,tl.float32)
        for block in range(tl.cdiv(K,LANES*VEC)):
            k=block*LANES*VEC+lane[:,None]*VEC+vec[None,:]
            x=tl.load(X+m*K+k,k<K,0)
            w=tl.load(W+n[:,None,None]*K+k[None,:,:],(n[:,None,None]<N)&(k[None,:,:]<K),0)
            acc=tl.fma(x[None,:,:],w,acc)
        if MODE=='vector_first':
            partial=_fold(acc,2,VEC,VORDER)
            out=tl.reshape(_fold(partial[:,:,None],1,LANES,LORDER),(BN,))
        else:
            partial=_fold(acc,1,LANES,LORDER)
            out=tl.reshape(_fold(partial[:,None,:],2,VEC,VORDER),(BN,))
    tl.store(Y+m*N+n,out,n<N)


def cyclic(x,weight,config,n=None):
    m,k=x.shape;n=weight.shape[0] if n is None else n
    lanes,vec,mode,lorder,vorder=config
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    _cyclic[(triton.cdiv(n,4),m)](x,weight,out,m,n,k,lanes,vec,mode,lorder,vorder,4,
                                 num_warps=4,enable_fp_fusion=False)
    return out


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,default=3)
    parser.add_argument('--wide',action='store_true')
    parser.add_argument('--output',default='results/small_arithmetic.json')
    args=parser.parse_args();strict_precision()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    shape=(args.rows,5120,1280);x,w=cases[shape]['x'],cases[shape]['weight']
    ref=F.linear(x,w)[:,:128]
    report={'scope':'cyclic FMA order search on the first 128 outputs of the original full matrix',
            'shape':shape,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'trials':[]}
    for lanes in ([8,16,32,64] if args.wide else [16]):
        for vec in ([1,2,4,8] if args.wide else [4]):
            for mode in ['single','vector_first','lanes_first']:
                for lorder in ['down','up','serial']:
                    for vorder in (['down','up','serial'] if mode!='single' else ['down']):
                        cfg=(lanes,vec,mode,lorder,vorder)
                        out=cyclic(x,w,cfg,n=128)
                        d=difference(ref,out)
                        report['trials'].append({'config':cfg,'difference':d})
                        print(cfg,d['exact'],d['different_elements'],d['max_abs'],flush=True)
                        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
