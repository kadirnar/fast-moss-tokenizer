"""Small CUDA cuBLASLt research binding; explicit FP32 pedantic compute only.

ABI definitions mirror installed CUDA 12.8 headers. Never mutates PyTorch's BLAS
handle or global math mode. Plans/workspace must outlive captured graph replays.
"""
import ctypes as C
from pathlib import Path
import torch
import nvidia.cublas


class Algo(C.Structure):
    _fields_=[('data',C.c_uint64*8)]


class Heuristic(C.Structure):
    _fields_=[('algo',Algo),('workspace',C.c_size_t),('state',C.c_int),
             ('waves',C.c_float),('reserved',C.c_int*4)]


P=C.c_void_p;I=C.c_int;SZ=C.c_size_t
_LIB=None


def library():
    global _LIB
    if _LIB is None:
        path=Path(nvidia.cublas.__path__[0])/'lib/libcublasLt.so.12'
        lib=C.CDLL(str(path))
        signatures={
            'cublasLtCreate':[C.POINTER(P)],'cublasLtDestroy':[P],
            'cublasLtMatmulDescCreate':[C.POINTER(P),I,I],'cublasLtMatmulDescDestroy':[P],
            'cublasLtMatmulDescSetAttribute':[P,I,P,SZ],
            'cublasLtMatrixLayoutCreate':[C.POINTER(P),I,C.c_uint64,C.c_uint64,C.c_int64],
            'cublasLtMatrixLayoutDestroy':[P],'cublasLtMatrixLayoutSetAttribute':[P,I,P,SZ],
            'cublasLtMatmulPreferenceCreate':[C.POINTER(P)],'cublasLtMatmulPreferenceDestroy':[P],
            'cublasLtMatmulPreferenceSetAttribute':[P,I,P,SZ],
            'cublasLtMatmulAlgoGetHeuristic':[P,P,P,P,P,P,P,I,C.POINTER(Heuristic),C.POINTER(I)],
            'cublasLtMatmulAlgoConfigGetAttribute':[C.POINTER(Algo),I,P,SZ,C.POINTER(SZ)],
            'cublasLtMatmulAlgoCheck':[P,P,P,P,P,P,C.POINTER(Algo),C.POINTER(Heuristic)],
            'cublasLtMatmul':[P,P,P,P,P,P,P,P,P,P,P,P,C.POINTER(Algo),P,SZ,P],
        }
        for name,args in signatures.items():
            fn=getattr(lib,name);fn.argtypes=args;fn.restype=I
        lib.cublasLtGetVersion.argtypes=[];lib.cublasLtGetVersion.restype=SZ
        _LIB=lib
    return _LIB


def check(status):
    if status:raise RuntimeError(f'cuBLASLt status {status}')


class LinearPlan:
    def __init__(self,weight,rows,layout='col',candidates=16,workspace_bytes=32*1024*1024,workspace=None):
        if (not weight.is_cuda or weight.dtype!=torch.float32 or weight.ndim!=2
                or not weight.is_contiguous() or layout not in ['col','row','packed'] or rows<1):
            raise ValueError('Expected contiguous CUDA FP32 weight and supported layout')
        if candidates<1 or workspace_bytes<0:
            raise ValueError('Positive candidate count and nonnegative workspace size required')
        # Default heuristic preferences assume 256-byte aligned operands.
        if weight.data_ptr()%256:
            raise ValueError('Weight must be 256-byte aligned')
        self.lib=library();self.resources=[]
        self.n,self.k=weight.shape;self.m=rows;self.device=weight.device
        self.layout=layout;self.weight=weight if layout!='packed' else weight.T.contiguous()
        if workspace is not None and (workspace.device!=self.device or workspace.dtype!=torch.uint8
                                      or not workspace.is_contiguous() or workspace.numel()<workspace_bytes
                                      or workspace.data_ptr()%256):
            raise ValueError('Shared workspace must be a sufficiently large contiguous CUDA byte tensor')
        self.workspace=(torch.empty(workspace_bytes,device=self.device,dtype=torch.uint8)
                        if workspace is None else workspace)
        self.alpha=C.c_float(1.);self.beta=C.c_float(0.)
        try:
            self.handle=self.create('cublasLtCreate','cublasLtDestroy')
            # CUBLAS_COMPUTE_32F_PEDANTIC = 69, CUDA_R_32F = 0.
            self.desc=self.create('cublasLtMatmulDescCreate','cublasLtMatmulDescDestroy',69,0)
            if layout=='col':
                self.attr('cublasLtMatmulDescSetAttribute',self.desc,3,I(1)) # TRANSA = T
                self.a=self.matrix(self.k,self.n,self.k,False)
                self.b=self.matrix(self.k,self.m,self.k,False)
                self.c=self.matrix(self.n,self.m,self.n,False)
            else:
                self.attr('cublasLtMatmulDescSetAttribute',self.desc,4,I(1 if layout=='row' else 0))
                self.a=self.matrix(self.m,self.k,self.k,True)
                self.b=(self.matrix(self.n,self.k,self.k,True) if layout=='row'
                        else self.matrix(self.k,self.n,self.n,True))
                self.c=self.matrix(self.m,self.n,self.n,True)
            pref=self.create('cublasLtMatmulPreferenceCreate','cublasLtMatmulPreferenceDestroy')
            self.attr('cublasLtMatmulPreferenceSetAttribute',pref,1,SZ(workspace_bytes))
            results=(Heuristic*candidates)();count=I()
            check(self.lib.cublasLtMatmulAlgoGetHeuristic(self.handle,self.desc,self.a,self.b,self.c,self.c,
                                                        pref,candidates,results,C.byref(count)))
            self.algorithms=[h for h in results[:count.value] if h.state==0]
            if not self.algorithms:raise RuntimeError('No compatible cuBLASLt algorithm')
        except BaseException:
            self.close();raise

    def create(self,create,destroy,*args):
        value=P();check(getattr(self.lib,create)(C.byref(value),*args))
        self.resources.append((destroy,value));return value

    def attr(self,method,desc,attribute,value):
        check(getattr(self.lib,method)(desc,attribute,C.byref(value),C.sizeof(value)))

    def matrix(self,rows,cols,ld,row):
        value=self.create('cublasLtMatrixLayoutCreate','cublasLtMatrixLayoutDestroy',0,rows,cols,ld)
        if row:self.attr('cublasLtMatrixLayoutSetAttribute',value,1,I(1))
        return value

    def metadata(self,index):
        h=self.algorithms[index];id=I();written=SZ()
        check(self.lib.cublasLtMatmulAlgoConfigGetAttribute(C.byref(h.algo),0,C.byref(id),C.sizeof(id),C.byref(written)))
        return {'id':id.value,'opaque':list(h.algo.data),'workspace_bytes':h.workspace,'waves':h.waves}

    def restore(self,metadata,version):
        """Restore a same-library descriptor; heuristic shortlists can vary."""
        if version!=self.lib.cublasLtGetVersion():
            raise ValueError('Serialized algorithm requires the same cuBLASLt version')
        words=metadata['opaque']
        if len(words)!=8 or any(type(v) is not int or not 0<=v<2**64 for v in words):
            raise ValueError('Expected eight unsigned 64-bit algorithm words')
        algo=Algo((C.c_uint64*8)(*words));result=Heuristic()
        check(self.lib.cublasLtMatmulAlgoCheck(self.handle,self.desc,self.a,self.b,self.c,self.c,
                                              C.byref(algo),C.byref(result)))
        check(result.state)
        if result.workspace>self.workspace.numel():
            raise ValueError('Restored algorithm exceeds available workspace')
        result.algo=algo
        self.algorithms.append(result)
        return len(self.algorithms)-1

    def __call__(self,x,index=0):
        if (x.shape!=(self.m,self.k) or x.device!=self.device or x.dtype!=torch.float32
                or not x.is_contiguous() or x.data_ptr()%256 or not self.resources):
            raise ValueError('Input must match the live plan shape/device/FP32 contiguous layout')
        out=torch.empty((self.m,self.n),device=x.device,dtype=x.dtype)
        h=self.algorithms[index]
        a,b=(self.weight,x) if self.layout=='col' else (x,self.weight)
        check(self.lib.cublasLtMatmul(self.handle,self.desc,C.byref(self.alpha),a.data_ptr(),self.a,
                                    b.data_ptr(),self.b,C.byref(self.beta),out.data_ptr(),self.c,
                                    out.data_ptr(),self.c,C.byref(h.algo),self.workspace.data_ptr(),
                                    self.workspace.numel(),torch.cuda.current_stream(x.device).cuda_stream))
        return out

    def close(self):
        for method,value in reversed(self.resources):getattr(self.lib,method)(value)
        self.resources.clear()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
