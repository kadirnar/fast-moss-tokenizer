"""Alternating matched runs: new configuration search versus prior resident tuning."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from benchmarks.cublaslt_model import choices,experimental
from benchmarks.lane_completion import audio_sources
from benchmarks.baseline import measure
from benchmarks.compare import difference
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/full_cublaslt_config_ablation.json')
    a=p.parse_args()
    paths=['results/matrices_cublaslt.json','results/matrices_cublaslt_codec8.json','results/matrices_cublaslt_large.json']
    options={'previous':choices(paths,packed_only=True),'searched':choices(paths+['results/cublaslt_config_search.json'],packed_only=True)}
    model=load_model();clips,sources=audio_sources()
    report={'scope':'matched new versus previous resident FP32 configuration, not original-model speedup',
            'revision':REVISION,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'dtype':'float32','tf32':False,'quantizers':32,'source':sources[0],
            'input':'cyclic music, lane starts offset by 1920 samples','cases':[]}
    def run(x):
        enc=model._encode_frame(x)
        return enc.audio_codes,enc.encoder_hidden_states,model._decode_frame(enc.audio_codes).audio
    with optimized(model,residual_backend='triton',rope_backend='triton',kv_backend='triton',
                   share_rope_tables=True,attention_mask_backend='triton'):
        for batch,frames in [(1,1),(8,3),(128,3)]:
            idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
            x=clips[0][idx%clips[0].numel()][:,None]
            reference=run(x);case={'batch':batch,'frames':frames,'rounds':[]}
            for iteration,mode in enumerate(['previous','searched','searched','previous','previous','searched']):
                with experimental(model,options[mode],resident=True):
                    graph=GraphedCallable(run,x)
                    output=graph(x)
                    check={k:difference(old,new) for k,old,new in zip(['codes','hidden','audio'],reference,output)}
                    timing=measure(lambda:graph(x),repeats=20)
                    case['rounds'].append({'order':iteration,'mode':mode,'fidelity':check,'timing':timing})
                    del graph,output
            medians={mode:statistics.median(r['timing']['wall_ms_median'] for r in case['rounds'] if r['mode']==mode) for mode in options}
            case['median_wall_ms']=medians;case['speedup']=medians['previous']/medians['searched']
            case['all_exact']=all(v['exact'] for r in case['rounds'] for v in r['fidelity'].values())
            report['cases'].append(case)
            print(batch,frames,medians,'speedup',case['speedup'],'exact',case['all_exact'],flush=True)
            Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=all(c['all_exact'] for c in report['cases'])
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Configuration ablation fidelity failed')


if __name__=='__main__':main()
