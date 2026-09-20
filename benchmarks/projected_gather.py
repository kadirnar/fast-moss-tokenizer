"""Compare exact sequential decoder gathers with output- or token-major tiles."""
import json
from pathlib import Path
import torch
import triton
from triton.testing import do_bench_cudagraph
from fast_moss.projections import _decode, _decode_tokens


@torch.inference_mode()
def main():
    torch.manual_seed(832)
    table=torch.randn(32,1024,512,device='cuda')*.05
    records=[]
    for b,t in [(1,1),(1,3),(8,3),(128,3),(8,40),(8,128)]:
        codes=torch.randint(1024,(32,b,t),device='cuda')
        out=torch.empty(b,512,t,device='cuda')
        gold=torch.zeros_like(out)
        for q in range(32):gold+=table[q][codes[q]].transpose(1,2)
        for mode in ['flat','tokens']:
            def run():
                if mode=='flat':
                    _decode[(triton.cdiv(out.numel(),256),)](codes,table,out,out.numel(),512,t,*codes.stride(),32,1024,256,enable_fp_fusion=False)
                else:
                    _decode_tokens[(b*t,2)](codes,table,out,t,*codes.stride(),32,256,enable_fp_fusion=False)
                return out
            assert torch.equal(run(),gold)
            values=[do_bench_cudagraph(run,rep=30,return_mode='median') for _ in range(3)]
            records.append({'batch':b,'frames':t,'mode':mode,'exact':True,'graph_ms':values})
            print(b,t,mode,values,flush=True)
    Path('results/projected_gather.json').write_text(json.dumps({'scope':'component only, resident random FP32 table; excludes validation, allocations and final output projection',
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':records},indent=2)+'\n')


if __name__=='__main__':main()
