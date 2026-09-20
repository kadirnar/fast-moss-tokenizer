"""Bracket finalists with fresh baselines to expose long-search timing drift."""
import argparse
import json
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.cublaslt import LinearPlan,library
from benchmarks.matrices import evicted_replay
from benchmarks.compare import difference
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/cublaslt_config_search.json')
    p.add_argument('--output',default='results/cublaslt_config_confirmation.json')
    a=p.parse_args();strict_precision()
    search=json.loads(Path(a.input).read_text())
    if (search['cublaslt_version']!=library().cublasLtGetVersion()
            or search['gpu']!=torch.cuda.get_device_name() or search['torch']!=torch.__version__):
        raise ValueError('Confirmation requires the recorded library, GPU, and PyTorch versions')
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    groups={}
    for r in search['records']:groups.setdefault(tuple(r['shape_MNK']),[]).append(r)
    report={'scope':'fresh alternating finalist/baseline component check; not model speedup',
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cublaslt_version':library().cublasLtGetVersion(),
            'source':a.input,'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape,rows in groups.items():
        case=cases[shape];x,w=case['x'],case['weight'];ref=F.linear(x,w)
        base_fn=lambda z:F.linear(z,w)
        base_graph=GraphedCallable(lambda z:(base_fn(z),),x)
        base=next(r for r in rows if r['backend']=='torch')
        candidates=[r for r in rows if r['backend'].startswith('lt_packed_')]
        if not candidates:
            del base_graph;continue
        # Confirm the warm winner and the cold winner independently.
        finalists={r['backend']:r for r in [min(candidates,key=lambda r:r['hot_graph_ms']),
                                          min(candidates,key=lambda r:r['evicted_replay']['gpu_ms_median'])]}
        with LinearPlan(w,shape[0],'packed') as plan:
            for candidate in finalists.values():
                index=plan.restore(candidate['algorithm'],search['cublaslt_version'])
                fn=lambda z:plan(z,index)
                graph=GraphedCallable(lambda z:(fn(z),),x)
                rounds=[]
                for mode in ['baseline','candidate','baseline','candidate','baseline']:
                    active=base_fn if mode=='baseline' else fn
                    active_graph=base_graph if mode=='baseline' else graph
                    rounds.append({'mode':mode,'hot_graph_ms':do_bench_cudagraph(lambda:active(x),rep=30,return_mode='median'),
                                   'evicted_replay':evicted_replay(active_graph,flush)})
                medians={mode:{'hot_ms':statistics.median(r['hot_graph_ms'] for r in rounds if r['mode']==mode),
                               'evicted_ms':statistics.median(r['evicted_replay']['gpu_ms_median'] for r in rounds if r['mode']==mode)}
                         for mode in ['baseline','candidate']}
                record={'shape_MNK':shape,'backend':candidate['backend'],'configuration':candidate['configuration'],
                        'fidelity':difference(ref,fn(x)),'search_baseline_hot_ms':base['hot_graph_ms'],
                        'rounds':rounds,'medians':medians,
                        'hot_gain':medians['baseline']['hot_ms']/medians['candidate']['hot_ms'],
                        'evicted_gain':medians['baseline']['evicted_ms']/medians['candidate']['evicted_ms']}
                report['records'].append(record)
                print(shape,candidate['configuration'],'warm',record['hot_gain'],'evicted',record['evicted_gain'],flush=True)
                del graph
        del base_graph
    report['all_exact']=all(r['fidelity']['exact'] for r in report['records'])
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
