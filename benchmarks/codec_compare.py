"""Fresh original-eager/original-graph/current-graph whole-codec comparison."""
from contextlib import nullcontext
import argparse
import json
from pathlib import Path
import statistics
import torch
from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--frames',type=int,default=3)
    parser.add_argument('--batches',type=int,nargs='+',default=[1,8])
    parser.add_argument('--output',default='results/full_codec_current.json')
    parser.add_argument("--norm-backend", choices=["none", "cuda"], default="none")
    parser.add_argument("--matrix-backend", choices=["triton", "cuda"], default="triton")
    args=parser.parse_args()
    if args.frames<1 or any(b<1 for b in args.batches):parser.error('Frames and batches must be positive')
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend=args.matrix_backend,ffn_backend='triton',norm_backend=args.norm_backend)
    report={'scope':'full checkpoint, original eager/original graph/current optimized graph',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'dtype':'float32','tf32':False,'quantizers':32,'sources':sources,'options':opts,
            'input':f'speech source (index 1), cyclic {args.frames*1920}-sample windows with lane*1920-sample offsets',
            'timing_scope':'three rotating-order rounds, ten single-call samples after three warmups per direction/mode; independent restored contexts, one live graph at a time; graph times include input copies and owned outputs; excludes load, packing, capture and restoration',
            'cases':[]}
    def encode(z):
        e=model._encode_frame(z)
        return e.audio_codes,e.encoder_hidden_states
    for batch in args.batches:
        frames=args.frames
        idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
        x=clips[1][idx%clips[1].numel()][:,None]
        enc=model._encode_frame(x);codes=enc.audio_codes
        refs={'encode':(codes,enc.encoder_hidden_states),'decode':(model._decode_frame(codes).audio,)}
        funcs=[('encode',encode,x),('decode',lambda z:(model._decode_frame(z).audio,),codes)]
        case={'batch':batch,'frames':frames,'rounds':[]}
        modes=['original_eager','original_graph','optimized_graph']
        for repeat in range(3):
            order=modes[repeat:]+modes[:repeat]
            for mode in order:
                row={'round':repeat,'mode':mode,'results':{}}
                with (optimized(model,**opts) if mode=='optimized_graph' else nullcontext()):
                    # Complete all packing before graph capture.
                    for _,fn,inp in funcs:fn(inp)
                    for direction,fn,inp in funcs:
                        graph=None if mode=='original_eager' else GraphedCallable(fn,inp)
                        run=fn if graph is None else graph
                        checks=[difference(ref,out) for ref,out in zip(refs[direction],run(inp))]
                        timing=measure(lambda:run(inp),warmup=3,repeats=10)
                        row['results'][direction]={'checks':checks,'timing':timing}
                        del run,graph
                row['restored']={direction:[difference(ref,out) for ref,out in zip(refs[direction],fn(inp))]
                                 for direction,fn,inp in funcs}
                case['rounds'].append(row)
                print(batch,repeat,mode,{k:v['timing']['wall_ms_median'] for k,v in row['results'].items()},flush=True)
        case['medians_ms']={direction:{mode:statistics.median(v for r in case['rounds'] if r['mode']==mode
                                  for v in r['results'][direction]['timing']['wall_ms_samples'])
                                  for mode in modes} for direction,_,_ in funcs}
        case['speedups']={direction:{mode:times[mode]/times['optimized_graph'] for mode in modes[:2]}
                          for direction,times in case['medians_ms'].items()}
        case['all_exact']=all(c['exact'] for row in case['rounds'] for r in row['results'].values() for c in r['checks']) and all(
            c['exact'] for row in case['rounds'] for checks in row['restored'].values() for c in checks)
        report['cases'].append(case)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=all(c['all_exact'] for c in report['cases'])
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Whole-codec comparison gate failed')


if __name__=='__main__':main()
