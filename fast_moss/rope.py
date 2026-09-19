"""FP32 RoPE fusion using the upstream PyTorch trigonometric tables."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(Q,K,C,S,OQ,OK,N:tl.constexpr,H:tl.constexpr,T:tl.constexpr,D:tl.constexpr,
            QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,QS3:tl.constexpr,
            KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,KS3:tl.constexpr,
            BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    pair=i%(D//2)
    t=i//(D//2)%T
    h=i//(D//2*T)%H
    b=i//(D//2*T*H)
    qi=b*QS0+h*QS1+t*QS2+2*pair*QS3
    ki=b*KS0+h*KS1+t*KS2+2*pair*KS3
    phase=(b*T+t)*(D//2)+pair
    qr=tl.load(Q+qi,i<N,0); qj=tl.load(Q+qi+QS3,i<N,0)
    kr=tl.load(K+ki,i<N,0); kj=tl.load(K+ki+KS3,i<N,0)
    c=tl.load(C+phase,i<N,0); s=tl.load(S+phase,i<N,0)
    tl.store(OQ+2*i,qr*c-qj*s,i<N)
    tl.store(OQ+2*i+1,qr*s+qj*c,i<N)
    tl.store(OK+2*i,kr*c-kj*s,i<N)
    tl.store(OK+2*i+1,kr*s+kj*c,i<N)


def rotate(q,k,cos,sin):
    if q.shape!=k.shape or q.ndim!=4 or q.shape[-1]%2:
        raise ValueError("Expected equal (batch,heads,time,even_dim) Q/K tensors")
    b,h,t,d=q.shape
    if cos.shape!=(b,t,d//2) or sin.shape!=cos.shape:
        raise ValueError("Unexpected rotary table dimensions")
    tensors=(q,k,cos,sin)
    if not q.is_cuda or any(x.dtype!=torch.float32 or x.device!=q.device for x in tensors):
        raise ValueError("RoPE fusion requires FP32 tensors on one CUDA device")
    if not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("RoPE tables must be contiguous")
    oq=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    ok=torch.empty_like(oq)
    n=q.numel()//2
    if n:
        _rotate[(triton.cdiv(n,256),)](q,k,cos,sin,oq,ok,n,h,t,d,*q.stride(),*k.stride(),256,
                                      enable_fp_fusion=False)
    return oq,ok


def tables(self,device,b,t,d,offset):
    key=(device,d,self.max_period)
    if key not in self._fast_freqs:
        ds=torch.arange(d//2,device=device,dtype=torch.float32)
        self._fast_freqs[key]=torch.exp(ds*(-math.log(self.max_period)*2/d))
    ts=offset.float().view(-1,1)+torch.arange(t,device=device,dtype=torch.float32)
    phase=self._fast_freqs[key]*ts.view(b,t,1)
    return torch.cos(phase),torch.sin(phase)


def forward(self,q,k,offset,time_before_heads=False):
    if time_before_heads or q.dtype!=torch.float32:
        return self._fast_original_rope(q,k,offset,time_before_heads)
    b,h,t,d=q.shape
    shared=getattr(self,"_fast_tables",None)
    cos,sin=shared if shared is not None else tables(self,q.device,b,t,d,offset)
    return rotate(q,k,cos,sin)


def stage_forward(self,x,*args,**kwargs):
    """Reuse tables only where layer positions are known to advance together."""
    state=self._streaming_state
    if (x.dtype!=torch.float32 or self.positional_embedding!="rope"
            or (state is not None and not getattr(state,"_fast_synchronized",False))):
        return self._fast_original_stage(x,*args,**kwargs)
    b,t,_=x.shape
    attn=self.layers[0].self_attn
    offset=(torch.zeros(b,device=x.device,dtype=torch.long) if state is None
            else attn._streaming_state.offset)
    previous=self.rope._fast_tables
    self.rope._fast_tables=tables(self.rope,x.device,b,t,attn.embed_dim//attn.num_heads,offset)
    try:
        return self._fast_original_stage(x,*args,**kwargs)
    finally:
        self.rope._fast_tables=previous
