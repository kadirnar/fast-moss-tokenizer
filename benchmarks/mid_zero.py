"""Signed-zero/underflow bit audit of accepted and proposed matrix families."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.mid_rows import run
from fast_moss.loading import strict_precision
from fast_moss.graphs import GraphedCallable

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--legacy',action='store_true');parser.add_argument('--output',default='results/mid_zero_fixed.json');args=parser.parse_args()
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    snapshot=json.load(open('results/mid_baseline.json'))
    configs={**{tuple(r['shape']):tuple(r['config']) for r in snapshot['small_configs']},
             **{tuple(r['shape']):('ordered',*r['config']) for r in snapshot['ordered_configs']}}
    baseline=set(configs)
    for r in json.load(open('results/mid_arithmetic.json'))['records']:
        if not r['exact_chunks']:continue
        configs[tuple(r['shape'])]=('fixed',4,4,1,16) if r['family']=='small' else ('ordered',r['exact_chunks'][0],32,64,32,4,3)
    records=[]
    for shape,cfg in configs.items():
        if cfg[0]=='gemv':continue  # Separate GEMV signed-zero gate already covers these.
        x=torch.full_like(cases[shape]['x'],-1.401298464324817e-45)
        for label,w in [('actual_weight',cases[shape]['weight']),('positive_weight',torch.full_like(cases[shape]['weight'],.125))]:
            packed=w.T.contiguous();ref=F.linear(x,w)
            fn=lambda z:run(z,w,packed,cfg,exact_zero=not args.legacy)
            out=fn(x);graph=GraphedCallable(lambda z:(fn(z),),x);replay=graph(x)[0]
            bits=torch.equal(ref.view(torch.int32),out.view(torch.int32));gbits=torch.equal(ref.view(torch.int32),replay.view(torch.int32))
            records.append({'shape':shape,'config':cfg,'baseline':shape in baseline,'input':label,'eager_bits_equal':bits,'graph_bits_equal':gbits,
                            'reference_negative_zero':int((ref.view(torch.int32)==-2147483648).sum()),
                            'candidate_negative_zero':int((out.view(torch.int32)==-2147483648).sum())})
            del graph
            print(shape,label,bits,gbits,flush=True)
    report={'scope':'negative minimum-subnormal activations, actual and forced positive weights','signed_zero_policy':'legacy' if args.legacy else 'corrected','records':records,
            'all_bits_equal':all(r['eager_bits_equal'] and r['graph_bits_equal'] for r in records)}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
