"""Explore documented cuBLASLt algorithm capabilities beyond heuristic shortlists."""
import ctypes as C
import itertools
from benchmarks.cublaslt import Algo,Heuristic,I,SZ,check


def capabilities(lib,algo,attribute,dtype=C.c_uint32):
    size=SZ()
    check(lib.cublasLtMatmulAlgoCapGetAttribute(C.byref(algo),attribute,None,0,C.byref(size)))
    if size.value%C.sizeof(dtype):raise RuntimeError('Unexpected capability ABI size')
    result=(dtype*(size.value//C.sizeof(dtype)))()
    if size.value:
        check(lib.cublasLtMatmulAlgoCapGetAttribute(C.byref(algo),attribute,result,size.value,C.byref(size)))
    return list(result)


def algorithms(plan):
    ids=(I*512)();count=I()
    check(plan.lib.cublasLtMatmulAlgoGetIds(plan.handle,69,0,0,0,0,0,len(ids),ids,C.byref(count)))
    if count.value==len(ids):raise RuntimeError('Algorithm ID buffer may be truncated')
    for id in ids[:count.value]:
        algo=Algo()
        status=plan.lib.cublasLtMatmulAlgoInit(plan.handle,69,0,0,0,0,0,id,C.byref(algo))
        if status:continue
        yield id,algo


def configure(lib,base,settings):
    algo=Algo.from_buffer_copy(base)
    for attribute,value in settings.items():
        val=I(value)
        check(lib.cublasLtMatmulAlgoConfigSetAttribute(C.byref(algo),attribute,C.byref(val),C.sizeof(val)))
    return algo


def custom_options(maximum):
    # Some kernels expose an encoded 18-bit range. This is a bounded search,
    # not an assertion that every advertised custom value was enumerated.
    if maximum<=127:return range(maximum+1)
    values=set(range(32))|{maximum}
    for bit in range(maximum.bit_length()):
        values.update([1<<bit,(1<<bit)-1])
    return sorted(v for v in values if v<=maximum)


def candidates(plan,splitk=(0,)):
    """Yield validated FP32-FMA configurations and their documented settings."""
    seen=set()
    seed_splits={}
    names=['id','tile','split_k','reduction','swizzle','custom','stage']
    # Preserve all heuristic seeds, including split factors outside the manual
    # sweep. Snapshot first: callers append yielded descriptors to the plan.
    for h in tuple(plan.algorithms):
        flags=capabilities(plan.lib,h.algo,15,C.c_uint64)[0]
        if flags&0xffffff != 0x080201:continue
        config={}
        for attr,name in enumerate(names):
            value=I();size=SZ()
            check(plan.lib.cublasLtMatmulAlgoConfigGetAttribute(C.byref(h.algo),attr,C.byref(value),C.sizeof(value),C.byref(size)))
            config[name]=value.value
        seed_splits.setdefault(config['id'],set()).add(config['split_k'])
        config['numerical_flags']=flags;config['seed']='heuristic'
        key=tuple(h.algo.data)
        if key not in seen:
            seen.add(key)
            yield Heuristic.from_buffer_copy(h),config
    for id,base in algorithms(plan):
        flags=capabilities(plan.lib,base,15,C.c_uint64)[0]
        # Require FMA, FP32 input and accumulator. No tensor-op or truncated operands.
        if flags&0xffffff != 0x080201:continue
        tiles=capabilities(plan.lib,base,6) or [0]
        stages=capabilities(plan.lib,base,13) or [0]
        custom=capabilities(plan.lib,base,7)[0]
        swizzle=capabilities(plan.lib,base,2)[0]
        splits=sorted(set(splitk)|seed_splits.get(id,set())) if capabilities(plan.lib,base,0)[0] else [0]
        reductions=capabilities(plan.lib,base,1)[0]
        for tile,stage,option,swz,split in itertools.product(tiles,stages,custom_options(custom),range(min(swizzle,1)+1),splits):
            for reduction in ([0] if split<=1 else [r for r in [1,2,4] if reductions&r]):
                settings={1:tile,2:split,3:reduction,4:swz,5:option,6:stage}
                algo=configure(plan.lib,base,settings)
                key=tuple(algo.data)
                if key in seen:continue
                seen.add(key)
                result=Heuristic()
                status=plan.lib.cublasLtMatmulAlgoCheck(plan.handle,plan.desc,plan.a,plan.b,plan.c,plan.c,
                                                       C.byref(algo),C.byref(result))
                if status or result.state or result.workspace>plan.workspace.numel():continue
                result.algo=algo
                yield result,{'id':id,'tile':tile,'stage':stage,'custom':option,'swizzle':swz,
                              'split_k':split,'reduction':reduction,'numerical_flags':flags}
