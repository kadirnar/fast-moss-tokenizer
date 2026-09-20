"""Native launch evidence for the short-chunk FP32 matrix arithmetic search."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from fast_moss.loading import strict_precision, REVISION


SHAPES=[(m,n,k) for m in [1,3] for n,k in
        [(5120,1280),(1280,5120),(3840,1280),(1280,1280)]]
SHAPES += [(m,n,k) for m in [6,12] for n,k in
           [(3072,768),(768,3072),(2304,768),(768,768)]]


@torch.inference_mode()
def main():
    strict_precision()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'native small-row FP32 matrix launch geometry, ten calls per shape',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
    for shape in SHAPES:
        x,w=cases[shape]['x'],cases[shape]['weight']
        for _ in range(5):F.linear(x,w)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for _ in range(10):F.linear(x,w)
            torch.cuda.synchronize()
        path=Path('/tmp/moss-small-matrix-trace.json')
        prof.export_chrome_trace(str(path))
        events=json.loads(path.read_text())['traceEvents']
        kernels=[{'name':e['name'],'duration_us':e['dur'],'args':e.get('args',{})}
                 for e in events if e.get('cat')=='kernel']
        report['records'].append({'shape':shape,'kernels':kernels})
        print(shape,sorted({(e['name'],str(e['args'].get('grid')),str(e['args'].get('block')))
                            for e in kernels}),flush=True)
        Path('results/small_matrix_orders.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
