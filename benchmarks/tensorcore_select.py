"""Compare tensor-core candidates directly with the existing exact matrix choice."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.tensorcore_mm import linear
from benchmarks.cublaslt_model import choices
from benchmarks.cublaslt import LinearPlan
from benchmarks.matrices import evicted_replay
from benchmarks.compare import difference
from fast_moss.loading import strict_precision
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    strict_precision()
    paths=['results/matrices_cublaslt.json','results/matrices_cublaslt_codec8.json','results/matrices_cublaslt_large.json']
    exact=choices(paths,packed_only=True);cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    grouped={}
    for path in ['results/tensorcore_shapes.json','results/tensorcore_compact.json','results/tensorcore_bf16x6.json']:
        report=json.loads(Path(path).read_text());mode=report['candidate_arithmetic'].replace('Triton ','')
        for r in report['records']:
            if r['tile'] and r['shape_MNK'][0]==24:
                grouped.setdefault(tuple(r['shape_MNK']),[]).append((r['hot_graph_ms'],mode,r['tile']))
    result={'scope':'paired tensor-core versus existing exact matrix backend; not model quality proof',
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    def bench(fn,x):
        graph=GraphedCallable(lambda z:(fn(z),),x)
        r={'warm_ms':do_bench_cudagraph(lambda:fn(x),rep=30,return_mode='median'),
           'evicted_ms':evicted_replay(graph,flush)['gpu_ms_median']}
        del graph
        return r
    for shape,variants in grouped.items():
        x,w=cases[shape]['x'],cases[shape]['weight'];plan=None
        ref_fn=lambda z:F.linear(z,w)
        if shape in exact:
            choice=exact[shape];plan=LinearPlan(w,shape[0],'packed')
            index=plan.restore(choice['algorithm'],choice['cublaslt_version'])
            ref_fn=lambda z:plan(z,index)
        try:
            # Best measured tile from each arithmetic mode, followed by fresh paired timing.
            for mode in sorted({m for _,m,_ in variants}):
                _,_,tile=min(v for v in variants if v[1]==mode)
                fn=lambda z:linear(z,w,tile,mode=mode)
                candidate=fn(x);reference=ref_fn(x)
                before=bench(ref_fn,x);timing=bench(fn,x);after=bench(ref_fn,x)
                baseline={key:(before[key]+after[key])/2 for key in before}
                r={'shape_MNK':shape,'mode':mode,'tile':tile,'reference_backend':exact[shape]['backend'] if plan else 'torch',
                   'reference':difference(reference,candidate),'baseline':baseline,'candidate':timing,
                   'warm_gain':baseline['warm_ms']/timing['warm_ms'],'evicted_gain':baseline['evicted_ms']/timing['evicted_ms']}
                result['records'].append(r);print(shape,mode,r['warm_gain'],r['evicted_gain'],flush=True)
        finally:
            if plan:plan.close()
    Path('results/tensorcore_selection.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
