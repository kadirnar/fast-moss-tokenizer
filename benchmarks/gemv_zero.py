"""Check GEMV's final zero addition, including negative underflow and signed zero."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.gemv_order import gemv
from benchmarks.compare import difference
from fast_moss.loading import strict_precision

@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    search=json.loads(Path('results/gemv_order.json').read_text())
    records=[]
    for record in search['records']:
        shape=tuple(record['shape']);x,w=cases[shape]['x'],cases[shape]['weight']
        lanes=next(t['lanes'] for t in record['trials'] if t['bits_equal'])
        tiny=torch.nextafter(torch.tensor(0.,device='cuda'),torch.tensor(1.,device='cuda'))
        variants=[('actual',x,w),('negative_underflow',torch.full_like(x,-tiny.item()),torch.full_like(w,.125)),
                  ('positive_underflow',torch.full_like(x,tiny.item()),torch.full_like(w,.125)),
                  ('negative_zero',torch.full_like(x,-0.),w),('tiny_actual_weights',torch.full_like(x,-tiny.item()),w)]
        for label,z,weight in variants:
            ref=F.linear(z,weight)
            for zero in [False,True]:
                out=gemv(z,weight,lanes,zero=zero)
                bits=torch.equal(ref.view(torch.int32),out.view(torch.int32))
                records.append({'shape':shape,'input':label,'add_zero':zero,'difference':difference(ref,out),'bits_equal':bits})
                print(shape,label,zero,bits,flush=True)
    Path('results/gemv_zero.json').write_text(json.dumps({'scope':'final GEMV signed-zero behavior','records':records},indent=2)+'\n')

if __name__=='__main__':main()
