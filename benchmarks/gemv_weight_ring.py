"""Distinct-allocation GEMV rings bridge same-weight timing and codec traffic.

Each ring contains bit-identical copies of one captured weight, not different
checkpoint layers. Copies, packing and graph setup are outside GPU timings.
"""
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.gemv_interleaved import pack,gemv
from fast_moss.graphs import GraphedCallable
from fast_moss.small_matrices import linear
from fast_moss.loading import strict_precision,REVISION


@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    confirmation=json.loads(Path('results/gemv_interleaved_confirm.json').read_text())
    report={'scope':'synthetic distinct-allocation ring of bit-identical captured FP32 weight copies; excludes packing/capture/copies; not a codec benchmark or physical DRAM counter',
        'previous_commit':'dafaa46','revision':REVISION,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'records':[]}
    for row in confirmation['records']:
        shape=tuple(row['shape'])
        if shape not in ((1,1280,5120),(1,3840,1280),(1,5120,1280)):continue
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w).view(torch.int32)
        valid=[(name,t) for name,t in row['medians_ms'].items() if name not in ('native','current') and row['exact'][name]]
        configs=list(dict.fromkeys(tuple(json.loads(min(valid,key=lambda a:a[1][metric])[0])) for metric in ('warm','cold')))
        weights=[w.clone() for _ in range(32)]
        assert len({t.data_ptr() for t in weights})==32
        packed={cfg:[pack(t,cfg[0],cfg[1]) for t in weights] for cfg in configs}
        for length in (1,4,16,32):
            fns={'current':lambda z:tuple(linear(z,t) for t in weights[:length])}
            for cfg in configs:
                name=json.dumps(cfg)
                fns[name]=lambda z,cfg=cfg:tuple(gemv(z,p,cfg[2],cfg[3]) for p in packed[cfg][:length])
            checks={}
            for name,fn in fns.items():
                graph=GraphedCallable(fn,x)
                checks[name]={'eager':all(torch.equal(t.view(torch.int32),ref) for t in fn(x)),
                              'graph':all(torch.equal(t.view(torch.int32),ref) for t in graph(x))}
                del graph
            if not all(c['eager'] and c['graph'] for c in checks.values()):raise SystemExit('Ring fidelity mismatch')
            rounds=[]
            for repeat in range(3):
                names=list(fns);names=names[repeat%len(names):]+names[:repeat%len(names)]
                for name in names:
                    ms=do_bench_cudagraph(lambda:fns[name](x),rep=30)
                    rounds.append({'round':repeat,'backend':name,'ring_ms':ms,'per_call_ms':ms/length})
            medians={name:statistics.median(r['per_call_ms'] for r in rounds if r['backend']==name) for name in fns}
            record={'shape':shape,'length':length,'logical_weight_bytes':length*w.numel()*4,
                'configs':configs,'checks':checks,'rounds':rounds,'per_call_ms':medians,
                'speedups':{name:medians['current']/v for name,v in medians.items() if name!='current'}}
            report['records'].append(record)
            print(shape,length,record['speedups'],flush=True)
            Path('results/gemv_weight_ring.json').write_text(json.dumps(report,indent=2)+'\n')
        del packed,weights,fns

if __name__=='__main__':main()
