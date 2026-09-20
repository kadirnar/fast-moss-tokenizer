"""Research-only lossless block exponent packing and fused FP32 GEMV.

All 23 fraction bits and the sign are preserved. Exponents are integer deltas
from a per-256-value minimum; the integer representation is decoded before FMA.
"""
from dataclasses import dataclass
import numpy as np
import torch
import triton
import triton.language as tl


@dataclass
class Packed:
    data:torch.Tensor
    headers:torch.Tensor
    shape:tuple
    mode:str="variable"

    @property
    def nbytes(self):
        return self.data.numel()*self.data.element_size()+self.headers.numel()*self.headers.element_size()


def pack(weight,device=None,mode="variable"):
    if mode not in ("variable","fixed28"):raise ValueError("Unknown packing format")
    if weight.dtype!=torch.float32 or weight.numel()%256:
        raise ValueError('Expected FP32 input with a multiple of 256 elements')
    device=weight.device if device is None else device
    raw=weight.detach().cpu().contiguous().numpy().view(np.uint32).reshape(-1,256)
    exponent=(raw>>23)&255
    base=exponent.min(axis=1)
    width=np.ceil(np.log2(exponent.max(axis=1).astype(np.int64)-base+1)).astype(np.uint32)+24
    if mode=="fixed28":width=np.where(width<=28,28,32).astype(np.uint32)
    words=width.astype(np.int64)*8
    offsets=np.r_[0,np.cumsum(words)[:-1]]
    payload=np.zeros(int(words.sum())+1,dtype=np.uint32)
    for b in np.unique(width):
        rows=np.flatnonzero(width==b)
        q=((raw[rows]&0x7fffff)|((raw[rows]>>8)&0x800000)|((exponent[rows]-base[rows,None])<<24)).astype(np.uint64)
        if mode=="fixed28" and b==32:q=raw[rows].astype(np.uint64)
        pos=np.arange(256,dtype=np.uint64)*int(b)
        shift=pos%32
        idx=offsets[rows,None]+(pos//32).astype(np.int64)[None,:]
        np.bitwise_or.at(payload,idx.reshape(-1),((q<<shift[None,:])&0xffffffff).astype(np.uint32).reshape(-1))
        # A zero high word may target the first word of the next block, safely.
        np.bitwise_or.at(payload,(idx+1).reshape(-1),(q>>(32-shift[None,:])).astype(np.uint32).reshape(-1))
    headers=(offsets.astype(np.uint64)<<16)|(width.astype(np.uint64)<<8)|base.astype(np.uint64)
    return Packed(torch.from_numpy(payload.view(np.int32)).to(device),
                  torch.from_numpy(headers.view(np.int64)).to(device),tuple(weight.shape),mode)


@triton.jit
def _read(DATA,offset,bits,base,local,mask):
    pos=local*bits
    shift=pos%32
    word=offset+pos//32
    a=tl.load(DATA+word,mask,0).to(tl.uint32)
    b=tl.load(DATA+word+1,mask&(shift!=0),0).to(tl.uint32)
    q=((a>>shift)|(b<<((32-shift)%32)))&(tl.full((),0xffffffff,tl.uint32)>>(32-bits))
    raw=(q&0x7fffff)|((q&0x800000)<<8)|(((q>>24)+base)<<23)
    return raw.to(tl.float32,bitcast=True)


@triton.jit
def _read_fixed(DATA,offset,bits,base,local,mask):
    packed=bits==28
    shift=tl.where(packed,(local*28)%32,0)
    word=offset+tl.where(packed,local*7//8,local)
    a=tl.load(DATA+word,mask,0).to(tl.uint32)
    b=tl.load(DATA+word+1,mask&(shift!=0),0).to(tl.uint32)
    q=((a>>shift)|(b<<((32-shift)%32)))&0xfffffff
    reconstructed=(q&0x7fffff)|((q&0x800000)<<8)|(((q>>24)+base)<<23)
    return tl.where(packed,reconstructed,a).to(tl.float32,bitcast=True)


@triton.jit
def _unpack(DATA,HEADERS,OUT,NUMEL:tl.constexpr,B:tl.constexpr,FIXED:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    h=tl.load(HEADERS+i//256,i<NUMEL,0).to(tl.uint64)
    if FIXED:
        w=_read_fixed(DATA,(h>>16).to(tl.int32),((h>>8)&255).to(tl.int32),
            (h&255).to(tl.uint32),i%256,i<NUMEL)
    else:
        w=_read(DATA,(h>>16).to(tl.int32),((h>>8)&255).to(tl.int32),
            (h&255).to(tl.uint32),i%256,i<NUMEL)
    tl.store(OUT+i,w,i<NUMEL)


def unpack(packed,return_kernel=False):
    out=torch.empty(packed.shape,device=packed.data.device,dtype=torch.float32)
    kernel=_unpack[(triton.cdiv(out.numel(),256),)](packed.data,packed.headers,out,out.numel(),256,packed.mode=="fixed28",num_warps=4)
    return (out,kernel) if return_kernel else out


@triton.jit
def _packed_gemv(X,DATA,HEADERS,Y,N:tl.constexpr,K:tl.constexpr,
                 LANES:tl.constexpr,BN:tl.constexpr,UNROLL:tl.constexpr,FIXED:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    lane=tl.arange(0,LANES)
    acc=tl.full((BN,LANES),0,tl.float32)
    for part in range(K//256):
        h=tl.load(HEADERS+n*(K//256)+part,n<N,0).to(tl.uint64)
        offset=(h>>16).to(tl.int32);bits=((h>>8)&255).to(tl.int32);base=(h&255).to(tl.uint32)
        for step in tl.range(256//LANES,loop_unroll_factor=UNROLL):
            local=step*LANES+lane
            x=tl.load(X+part*256+local)
            if FIXED:
                w=_read_fixed(DATA,offset[:,None],bits[:,None],base[:,None],local[None,:],n[:,None]<N)
            else:
                w=_read(DATA,offset[:,None],bits[:,None],base[:,None],local[None,:],n[:,None]<N)
            acc=tl.fma(x[None,:],w,acc)
    for i in tl.static_range(0,tl.constexpr(LANES.bit_length()-1)):
        delta=LANES//2 >> i
        index=tl.broadcast_to(((lane+delta)%LANES)[None,:],(BN,LANES))
        acc=acc+tl.gather(acc,index,1)
    out=tl.reshape(tl.gather(acc,tl.full((BN,1),0,tl.int32),1),(BN,))
    out=tl.inline_asm_elementwise('add.rn.f32 $0, $1, 0f00000000;',
        constraints='=f,f',args=[out],dtype=tl.float32,is_pure=True,pack=1)
    tl.store(Y+n,out,n<N)


def gemv(x,packed,config,return_kernel=False):
    n,k=packed.shape;lanes,bn,warps,u=config
    if tuple(x.shape)!=(1,k) or k%256:raise ValueError('Expected one complete GEMV row')
    out=torch.empty((1,n),device=x.device,dtype=x.dtype)
    kernel=_packed_gemv[(triton.cdiv(n,bn),)](x,packed.data,packed.headers,out,n,k,lanes,bn,u,packed.mode=="fixed28",
                                           num_warps=warps,enable_fp_fusion=False)
    return (out,kernel) if return_kernel else out
