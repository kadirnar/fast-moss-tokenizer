"""Research-only pipeline/layout tuning with fixed FP32 accumulation order."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench_cudagraph
from triton.runtime.errors import OutOfResources
from fast_moss.ordered_matrices import linear as baseline, _reduce
from fast_moss.loading import strict_precision, REVISION
from fast_moss.graphs import GraphedCallable
from benchmarks.compare import difference
from benchmarks.matrices import evicted_replay


@triton.jit
def _pipeline(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, UNROLL: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in tl.range(256 // BK, loop_unroll_factor=UNROLL):
        k = part * 256 + block * BK + kk
        x = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        w = tl.load(W + k[:, None] * N + n[None, :], (n[None, :] < N) & (k[:, None] < K), 0)
        acc = tl.dot(x, w, acc, input_precision='ieee')
    tl.store(P + (part * M + m[:, None]) * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def linear(x, packed, config, return_kernel=False):
    m, k = x.shape; n = packed.shape[1]
    bm, bn, bk, warps, stages, unroll = config
    partials = torch.empty((k//256,m,n), device=x.device)
    out = torch.empty((m,n), device=x.device)
    kernel = _pipeline[(triton.cdiv(m,bm),triton.cdiv(n,bn),k//256)](
        x,packed,partials,m,n,k,bm,bn,bk,unroll,num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    _reduce[(triton.cdiv(m*n,256),)](partials,out,m*n,k//256,256,enable_fp_fusion=False)
    return (out,kernel) if return_kernel else out


@torch.inference_mode()
def main():
    strict_precision()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'fixed-order FP32 pipeline component search, not whole-model speedup',
            'revision':REVISION,'torch':torch.__version__,'triton':triton.__version__,
            'gpu':torch.cuda.get_device_name(),'records':[]}
    previous={}
    output=Path('results/ordered_pipeline.json')
    if output.exists():
        saved=json.loads(output.read_text())
        for record in saved.get('records',[])+([saved['current']] if 'current' in saved else []):
            previous[tuple(record['shape'])]=record['trials']
    configs=[]
    for bn in [64,128]:
        for stages in [1,2,3,4,5]:
            configs.append((32,bn,32,4,stages,1))
        for unroll in [2,4,8]:
            for stages in [1,2,3]:
                configs.append((32,bn,32,4,stages,unroll))
        for warps in [2,8]:
            configs.append((32,bn,32,warps,2,1))
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape in [(24,5120,1280),(24,1280,5120)]:
        x,w=cases[shape]['x'],cases[shape]['weight'];packed=w.T.contiguous();ref=F.linear(x,w)
        records=previous.get(shape,[])
        completed={tuple(r['config']) if r['config'] is not None else None for r in records}
        for cfg in [None]+configs:
            if cfg in completed:
                continue
            r={'config':cfg}
            try:
                if cfg is None:
                    fn=lambda z:baseline(z,packed)
                    out=fn(x)
                else:
                    fn=lambda z:linear(z,packed,cfg)
                    out,kernel=linear(x,packed,cfg,True)
                    r['resources']={'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared}
                r['difference']=difference(ref,out)
                if r['difference']['exact']:
                    graph=GraphedCallable(lambda z:(fn(z),),x)
                    r['graph']=difference(ref,graph(x)[0])
                    r['warm_ms']=do_bench_cudagraph(lambda:fn(x),rep=15)
                    r['cold']=evicted_replay(graph,flush,repeats=15)
                    del graph
            except (RuntimeError,triton.CompilationError,OutOfResources) as error:
                r['error']=str(error)
            records.append(r)
            print(shape,cfg,r.get('warm_ms'),r.get('resources'),r.get('error'),flush=True)
            current={'shape':shape,'trials':records}
            Path('results/ordered_pipeline.json').write_text(json.dumps({**report,'current':current},indent=2)+'\n')
        report['records'].append({'shape':shape,'trials':records})
    Path('results/ordered_pipeline.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
