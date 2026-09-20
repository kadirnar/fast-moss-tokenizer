"""Logical weight demand and small-matrix call attribution, not DRAM counters."""
import argparse
import json
from collections import Counter
from pathlib import Path
import torch
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from fast_moss import small_matrices as small
from benchmarks.ordered_model import options
from benchmarks.baseline import measure
from benchmarks.compare import difference


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--warmup",type=int,default=20);parser.add_argument("--output",default="results/weight_traffic.json");args=parser.parse_args()
    model=load_model();torch.manual_seed(9402)
    report={'scope':'full checkpoint logical module weight demand and supported small-matrix call attribution',
            'revision':REVISION,'previous_commit':'e87f54d','torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'dtype':'float32','timing_warmup':args.warmup,'cases':[],
            'measurement_limits':'Logical module weight bytes count each call once; not physical DRAM traffic or a bandwidth lower bound. Diagnostic module hooks can disable fusion; timings use a separate unhooked optimized context. Functional codebook products and attention state are not included in module-weight counts.'}
    opts=dict(options(),matrix_backend='triton',ffn_backend='triton')
    for batch,frames in [(1,1),(8,1),(1,3)]:
        x=torch.randn(batch,1,1920*frames,device='cuda')*.05
        e=model._encode_frame(x);codes=e.audio_codes
        for direction,fn,inp in [('encode',lambda z:(model._encode_frame(z).audio_codes,),x),('decode',lambda z:(model._decode_frame(z).audio,),codes)]:
            ref=fn(inp);calls=[];handles=[]
            for name,module in model.named_modules():
                if isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.ConvTranspose1d)):
                    def hook(mod,args,output,name=name):
                        calls.append({'name':name,'type':type(mod).__name__,'input_shape':list(args[0].shape),
                                      'weight_shape':list(mod.weight.shape),'weight_bytes':mod.weight.numel()*mod.weight.element_size()})
                    handles.append(module.register_forward_hook(hook))
            try:diagnostic=fn(inp)
            finally:
                for h in handles:h.remove()
            small_calls=[];original=small.linear
            with optimized(model,**opts):
                def tracked(z,w):
                    small_calls.append((z.shape[0],w.shape[0],z.shape[1]))
                    return original(z,w)
                small.linear=tracked
                try:observed=fn(inp)
                finally:small.linear=original
                graph=GraphedCallable(fn,inp)
                timing=measure(lambda:graph(inp),warmup=args.warmup,repeats=10)
                optimized_result=graph(inp);del graph
            checks=[difference(a,b) for out in [diagnostic,observed,optimized_result,fn(inp)] for a,b in zip(ref,out)]
            count=Counter(small_calls)
            record={'batch':batch,'frames':frames,'direction':direction,'module_calls':calls,
                    'logical_weight_bytes':sum(c['weight_bytes'] for c in calls),
                    'linear_weight_bytes':sum(c['weight_bytes'] for c in calls if c['type']=='Linear'),
                    'small_calls':[{'shape':s,'count':n,'config':small.CONFIGS[s],'logical_weight_bytes':n*s[1]*s[2]*4} for s,n in sorted(count.items())],
                    'timing':timing,'checks':checks,'all_exact':all(c['exact'] for c in checks)}
            report['cases'].append(record)
            print(batch,frames,direction,'logical GB',record['logical_weight_bytes']/1e9,'small GB',sum(c['logical_weight_bytes'] for c in record['small_calls'])/1e9,'exact',record['all_exact'],flush=True)
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
            if not record['all_exact']:raise SystemExit('Traffic diagnostic changed outputs')

if __name__=='__main__':main()
