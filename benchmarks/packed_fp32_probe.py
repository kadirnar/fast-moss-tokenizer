"""Measure exact exponent packing, standalone decode and fused GEMV."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from fast_moss.loading import strict_precision,REVISION
from fast_moss.small_matrices import CONFIGS,linear
from fast_moss.graphs import GraphedCallable
from benchmarks.packed_fp32 import pack,unpack,gemv
from benchmarks.ffn_resources import resources
from benchmarks.matrices import evicted_replay


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--format",choices=["variable","fixed28"],default="variable");parser.add_argument("--cooperative",action="store_true");parser.add_argument("--output",default="results/packed_fp32_probe.json");args=parser.parse_args()
    global gemv
    if args.cooperative:
        if args.format!="fixed28":raise ValueError("Cooperative decode requires fixed28")
        from benchmarks.packed_fp32_shuffle import gemv
    strict_precision();torch.manual_seed(3161)
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research-only lossless FP32 block storage and fused GEMV; no runtime replacement',
            'revision':REVISION,'previous_commit':'e87f54d','torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'records':[],
            'format':args.format, 'cooperative':args.cooperative, 'cooperative_backend':'cuda_nvrtc' if args.cooperative else None, 'storage':'23 fraction bits + sign + per-block exponent delta; packed 24..32-bit integer words, 64-bit header per 256 values, one guard word',
            'fixed28_policy':'28-bit sign/fraction/exponent-delta blocks; blocks whose exponent span exceeds 16 use verbatim FP32', 'timing_scope':'warm amortized CUDA graphs and 128 MiB-evicted replay; packing, copies and capture excluded; decoded/full FP32 weights kept only as reference controls'}
    patterns=torch.tensor([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0x7f800000,
                           0xff800000,0x7fc00001,0xffc00123,0x7f800001,0x3f800000,0xbf800000,0x00800001,0x80800001],dtype=torch.int64)
    special=patterns.repeat(16).to(torch.int32).view(torch.float32).reshape(16,16).cuda()
    special_packed=pack(special,mode=args.format);report['special_roundtrip_bits_equal']=torch.equal(special.view(torch.int32),unpack(special_packed).view(torch.int32))
    if not report['special_roundtrip_bits_equal']:raise SystemExit('Special-value bit roundtrip failed')
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape,config in CONFIGS.items():
        if shape[0]!=1:continue
        x,w=cases[shape]['x'],cases[shape]['weight'];p=pack(w,mode=args.format)
        decoded,dk=unpack(p,True);bits=torch.equal(w.view(torch.int32),decoded.view(torch.int32))
        if not bits:raise SystemExit('Checkpoint bit roundtrip failed')
        row={'shape':shape,'raw_bytes':w.numel()*4,'encoded_bytes':p.nbytes,'storage_ratio':w.numel()*4/p.nbytes,
             'roundtrip_bits_equal':bits,'decode_resources':resources(dk),'controls':{},'trials':[]}
        fns={'native':lambda:F.linear(x,w),'current':lambda:linear(x,w),'decode':lambda:unpack(p),
             'decode_then_current':lambda:linear(x,unpack(p))}
        for name,fn in fns.items():
            graph=GraphedCallable(lambda z:(fn(),),x)
            row['controls'][name]={'warm_ms':do_bench_cudagraph(fn,rep=20),'cold':evicted_replay(graph,flush,repeats=35)}
            del graph
        lanes=config[1]
        cfgs=list(dict.fromkeys([tuple(config[1:]),(lanes,4,1,4),(lanes,4,2,4),(lanes,8,2,16)]))
        if args.cooperative:cfgs=[(lanes,threads,u) for threads in [32,64,128] for u in [4,16]]
        variants=[('actual',x),('negated',-x),('scaled',x*.17),('zero',torch.zeros_like(x)),
                  ('subnormal',torch.randn_like(x)*1e-38),('tiny',torch.randn_like(x)*1e-20),
                  ('large',torch.randn_like(x)*1e20),('negative_minimum',torch.full_like(x,-1.401298464324817e-45)),
                  ('negative_zero',torch.full_like(x,-0.))]
        variants += [(f'random_{i}',torch.randn_like(x)) for i in range(4)]
        sparse=torch.zeros_like(x);sparse[:,::127]=torch.randn_like(sparse[:,::127]);variants.append(('sparse',sparse))
        for cfg in cfgs:
            out,kernel=gemv(x,p,cfg,True);graph=GraphedCallable(lambda z:(gemv(z,p,cfg),),x)
            checks=[]
            for name,z in variants:
                ref=F.linear(z,w)
                checks.append({'input':name,'eager_bits_equal':torch.equal(ref.view(torch.int32),gemv(z,p,cfg).view(torch.int32)),
                               'graph_bits_equal':torch.equal(ref.view(torch.int32),graph(z)[0].view(torch.int32))})
            row['trials'].append({'config':cfg,'checks':checks,'resources':kernel if isinstance(kernel,dict) else resources(kernel),
               'warm_ms':do_bench_cudagraph(lambda:gemv(x,p,cfg),rep=20),'cold':evicted_replay(graph,flush,repeats=35)})
            del graph
        report['records'].append(row)
        best=min(row['trials'],key=lambda t:t['warm_ms'])
        print(shape,'ratio',row['storage_ratio'],'current',row['controls']['current']['warm_ms'],'fused',best['warm_ms'],'exact',all(c['eager_bits_equal'] and c['graph_bits_equal'] for t in row['trials'] for c in t['checks']),flush=True)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=report['special_roundtrip_bits_equal'] and all(r['roundtrip_bits_equal'] and all(c['eager_bits_equal'] and c['graph_bits_equal'] for t in r['trials'] for c in t['checks']) for r in report['records'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Fused GEMV bit mismatch')

if __name__=='__main__':main()
