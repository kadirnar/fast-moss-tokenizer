"""Probe ordered FP32 reductions beyond the two supported FFN shapes."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph
from triton.runtime.errors import OutOfResources
from benchmarks.ordered_mm import tiled, configuration
from benchmarks.compare import difference
from benchmarks.matrices import evicted_replay
from fast_moss.cublaslt import LinearPlan
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision, REVISION


@torch.inference_mode()
def main():
    strict_precision()
    cases = torch.load('results/matrix_inputs.pt', weights_only=True)
    profile = json.loads(Path('fast_moss/matrix_profile.json').read_text())
    selected = {tuple(r['shape']): r for r in profile['records']}
    shapes = [(24,3840,1280),(24,1280,1280)]
    shapes += [(m,n,k) for m in [24,48,96,192] for n,k in
               [(3072,768),(768,3072),(2304,768),(768,768)]]
    report = {'scope':'ordered FP32 component expansion, not whole-model speedup',
              'revision':REVISION,'torch':torch.__version__,'triton':triton.__version__,
              'gpu':torch.cuda.get_device_name(),'reference':'contiguous 2-D F.linear',
              'records':[]}
    flush = torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    output = Path('results/ordered_shapes.json')
    for shape in shapes:
        x,w = cases[shape]['x'],cases[shape]['weight']
        packed = w.T.contiguous();ref = F.linear(x,w)
        r = {'shape':shape,'layer':cases[shape]['name'],'partitions':[],'tiles':[]}
        with LinearPlan(w,shape[0],'packed',packed_weight=packed) as plan:
            chunks = {32,64,128,256,512,shape[-1]}
            if shape in selected:
                index = plan.restore(selected[shape]['algorithm'],profile['cublaslt_version'])
                r['configuration'] = configuration(plan,index)
                splits = max(1,r['configuration']['split_k'])
                for alignment in [8,16,32,64,128]:
                    chunks.add(triton.cdiv(triton.cdiv(shape[-1],splits),alignment)*alignment)
                baseline = lambda z:plan(z,index)
            else:
                baseline = lambda z:F.linear(z,w)
            good = []
            for chunk in sorted(chunks):
                result = difference(ref,tiled(x,packed,chunk,(32,64,32,4)))
                r['partitions'].append({'chunk':chunk,'difference':result})
                if result['exact']:good.append(chunk)
            print(shape,'exact chunks',good,flush=True)
            r['baseline_exact'] = difference(ref,baseline(x))
            graph = GraphedCallable(lambda z:(baseline(z),),x)
            r['baseline'] = {'warm_ms':do_bench_cudagraph(lambda:baseline(x),rep=20),
                             'cold':evicted_replay(graph,flush,repeats=20)}
            del graph
            for chunk in good:
                for bm,bn in [(16,64),(32,64),(32,128),(64,64),(64,128)]:
                    tile = (bm,bn,32,4)
                    t = {'chunk':chunk,'tile':tile}
                    try:
                        out,kernel = tiled(x,packed,chunk,tile,return_kernel=True)
                        t['difference'] = difference(ref,out)
                        t['resources'] = {'registers':kernel.n_regs,'spills':kernel.n_spills,
                                          'shared_bytes':kernel.metadata.shared}
                        if t['difference']['exact']:
                            fn = lambda z:tiled(z,packed,chunk,tile)
                            graph = GraphedCallable(lambda z:(fn(z),),x)
                            t['graph'] = difference(ref,graph(x)[0])
                            t['warm_ms'] = do_bench_cudagraph(lambda:fn(x),rep=20)
                            t['cold'] = evicted_replay(graph,flush,repeats=20)
                            del graph
                    except (RuntimeError,triton.CompilationError,OutOfResources) as error:
                        t['error'] = str(error)
                    r['tiles'].append(t)
                    print(' tile',tile,t.get('warm_ms'),flush=True)
            report['records'].append(r)
            output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
