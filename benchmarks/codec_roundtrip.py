"""Time a complete encode-to-decode invocation against original FP32 execution."""
import argparse,json,statistics
from contextlib import nullcontext
from pathlib import Path
import torch
from benchmarks.baseline import measure
from benchmarks.codec_compare import source_state
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from benchmarks.residual_gemv_async_model import compare
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/full_codec_roundtrip.json')
    parser.add_argument('--frames',type=int,nargs='+',default=[1,3])
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--warmups',type=int,default=200)
    args=parser.parse_args()
    if args.rounds<1 or args.warmups<0 or any(f<1 for f in args.frames):parser.error('Invalid measurement options')
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='cuda',norm_backend='cuda',ffn_backend='triton')
    report={'scope':'direct complete codec roundtrip: encode output codes immediately feed decode within one invocation/graph; input and model already on CUDA; graph input copies and owned codes/hidden/audio included; excludes audio I/O, resampling, checkpoint loading, optimization setup, capture and restoration',
        **source_state(),'revision':REVISION,'model_id':'OpenMOSS-Team/MOSS-Audio-Tokenizer','sampling_rate':model.sampling_rate,'channels':1,
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'float32','tf32':False,'quantizers':32,'sources':sources,'options':opts,
        'rounds':args.rounds,'graph_warmups':args.warmups,'cases':[]}
    def run(x):
        e=model._encode_frame(x)
        return e.audio_codes,e.encoder_hidden_states,model._decode_frame(e.audio_codes).audio
    modes=['original_eager','original_graph','optimized_graph']
    for frames in args.frames:
        index=torch.arange(frames*model.downsample_rate,device='cuda')
        x=clips[1][index%clips[1].numel()][None,None];ref=run(x)
        case={'batch':1,'frames':frames,'audio_seconds':x.shape[-1]/model.sampling_rate,'rounds':[]}
        for repeat in range(args.rounds):
            for mode in modes[repeat%3:]+modes[:repeat%3]:
                with optimized(model,**opts) if mode=='optimized_graph' else nullcontext():
                    run(x)
                    graph=None if mode=='original_eager' else GraphedCallable(run,x)
                    fn=run if graph is None else graph
                    checks=compare(ref,fn(x));assert all(c['bits_equal'] for c in checks)
                    if graph is not None:
                        for _ in range(args.warmups):graph(x)
                    torch.cuda.synchronize();timing=measure(lambda:fn(x),warmup=3,repeats=10)
                    del fn,graph
                restored=compare(ref,run(x));assert all(c['bits_equal'] for c in restored)
                case['rounds'].append({'round':repeat,'mode':mode,'checks':checks,'restored':restored,'timing':timing})
                print(frames,repeat,mode,timing['wall_ms_median'],flush=True)
        case['medians_ms']={m:statistics.median(v for r in case['rounds'] if r['mode']==m for v in r['timing']['wall_ms_samples']) for m in modes}
        case['speedups']={m:case['medians_ms'][m]/case['medians_ms']['optimized_graph'] for m in modes[:2]}
        case['all_exact']=all(c['bits_equal'] for r in case['rounds'] for k in ('checks','restored') for c in r[k])
        report['cases'].append(case);Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=all(c['all_exact'] for c in report['cases']);report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
