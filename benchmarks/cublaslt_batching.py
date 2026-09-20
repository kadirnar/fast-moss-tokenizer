"""Matched whole-codec batching gates for resident FP32 cuBLASLt layouts."""
import argparse
import json
from pathlib import Path
import time
import torch
from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.cublaslt_model import choices,experimental
from benchmarks.lane_completion import audio_sources
from benchmarks.cublaslt import library
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tuning',nargs='+',default=['results/matrices_cublaslt.json','results/matrices_cublaslt_codec8.json','results/matrices_cublaslt_large.json'])
    p.add_argument('--batches',type=int,nargs='+',default=[1,8,128])
    p.add_argument('--frames',type=int,nargs='+',default=[1,3])
    p.add_argument('--output',default='results/full_cublaslt_batching.json')
    a=p.parse_args();selected=choices(a.tuning,packed_only=True);model=load_model();clips,sources=audio_sources()
    report={'scope':'matched full-codec resident FP32 experiment, not promoted','revision':REVISION,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cublaslt_version':library().cublasLtGetVersion(),
            'dtype':'float32','tf32':False,'quantizers':32,'sources':sources,'resident_weights':True,
            'input':'cyclic source samples, lanes offset by 1920 samples','cases':[]}
    def run(x):
        enc=model._encode_frame(x)
        return enc.audio_codes,enc.encoder_hidden_states,model._decode_frame(enc.audio_codes).audio
    with optimized(model,residual_backend='triton',rope_backend='triton',kv_backend='triton',
                   share_rope_tables=True,attention_mask_backend='triton'):
        for batch in a.batches:
            for frames in a.frames:
                idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
                inputs=[clip[idx%clip.numel()][:,None] for clip in clips]
                refs=[tuple(v.cpu() for v in run(x)) for x in inputs]
                torch.cuda.reset_peak_memory_stats()
                baseline=GraphedCallable(run,inputs[0])
                base_timing=measure(lambda:baseline(inputs[0]),repeats=15)
                base_peak=torch.cuda.max_memory_allocated()
                del baseline
                torch.cuda.reset_peak_memory_stats()
                setup=time.perf_counter()
                with experimental(model,selected,resident=True) as plans:
                    records=[]
                    for i,(x,ref) in enumerate(zip(inputs,refs)):
                        eager=run(x)
                        graph=GraphedCallable(run,x)
                        replay=graph(x)
                        checks={mode:{k:difference(old,new.cpu()) for k,old,new in zip(['codes','hidden','audio'],ref,out)}
                                for mode,out in [('eager',eager),('graph',replay)]}
                        records.append({'source':sources[i]['path'],'comparisons':checks})
                        if i==0:
                            torch.cuda.synchronize()
                            setup_seconds=time.perf_counter()-setup
                            timing=measure(lambda:graph(x),repeats=15)
                        del graph,eager,replay
                    count=len(plans)
                    unique={plan.weight.data_ptr():plan.weight.numel()*4 for plan,_ in plans.values() if plan.layout=='packed'}
                    packed_bytes=sum(unique.values())
                restored=run(inputs[0])
                restoration={k:difference(old,new.cpu()) for k,old,new in zip(['codes','hidden','audio'],refs[0],restored)}
                case={'batch':batch,'frames':frames,'audio_seconds':batch*frames*.08,'baseline_graph':base_timing,
                      'experimental_graph':timing,'setup_seconds':setup_seconds,'plan_count':count,'packed_weight_bytes':packed_bytes,
                      'additional_weight_bytes':0,'baseline_peak_bytes':base_peak,'experimental_peak_bytes':torch.cuda.max_memory_allocated(),
                      'sources':records,'restored':restoration}
                case['all_exact']=all(v['exact'] for r in records for c in r['comparisons'].values() for v in c.values()) and all(v['exact'] for v in restoration.values())
                case['speedup']=base_timing['wall_ms_median']/timing['wall_ms_median']
                case['audio_seconds_per_second']=case['audio_seconds']/(timing['wall_ms_median']/1000)
                report['cases'].append(case)
                print(batch,frames,'exact',case['all_exact'],'speedup',case['speedup'],'peak_GB',case['experimental_peak_bytes']/1e9,flush=True)
                Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
                del restored,inputs,refs
    report['all_exact']=all(c['all_exact'] for c in report['cases'])
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Batch fidelity failed')


if __name__=='__main__':main()
