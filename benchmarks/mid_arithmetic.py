"""Full-width arithmetic gates for the remaining 16/32/64-row native kernels."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.small_tiled_arithmetic import tiled as small
from benchmarks.ordered_mm import tiled as ordered
from benchmarks.compare import difference
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    launches=json.load(open('results/mid_native_orders.json'))
    report={'scope':'actual checkpoint full-output arithmetic, native launch-guided partitions',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
    for entry in launches['records']:
        shape=tuple(entry['shape']);x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        r={'shape':shape,'trials':[]};m,n,k=shape
        family='small' if any('gemmSN' in e['name'] for e in entry['kernels']) else 'ordered'
        r['family']=family
        packed=w.T.contiguous()
        hints=set()
        for event in entry['kernels']:
            if 'cutlass::Kernel2' in event['name']:
                splits=event['args']['grid'][2]
                for align in [8,16,32]:hints.add(((k+splits-1)//splits+align-1)//align*align)
        chunks=[128,256,512] if family=='small' else sorted(hints or {32,64,96,128,192,256,384,k})
        def probe(chunk):
            out=small(x,w,chunk) if family=='small' else ordered(x,packed,chunk,(32,64,32,4))
            d=difference(ref,out);bits=torch.equal(ref.view(torch.int32),out.view(torch.int32))
            r['trials'].append({'chunk':chunk,'difference':d,'bits_equal':bits})
            print(shape,family,chunk,bits,d['different_elements'],flush=True)
            return bits
        good=[chunk for chunk in chunks if probe(chunk)]
        if family=='ordered' and not good:
            good=[chunk for chunk in range(32,k+1,32) if chunk not in chunks and probe(chunk)]
        r['exact_chunks']=good
        report['records'].append(r)
        Path('results/mid_arithmetic.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
