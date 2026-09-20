"""Full-checkpoint fidelity and restored-context strided FFN epilogue ablation."""
import argparse
import json
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch

from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from fast_moss.ffn import math_library,observed,owned_norm_forward
import torch.nn.functional as F
from types import MethodType
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import REVISION, load_model
from fast_moss.optimize import optimized
RUNTIME = False
ENCODER_EPILOGUES = "both"


@contextmanager
def selected(model,configs):
    runtime=model._fast_matrix_runtime;counts={'candidate_calls':0};saved=[]
    previous=getattr(runtime,'ffn_strided_enabled',None)
    if RUNTIME:
        before=runtime.ffn_strided_calls;runtime.ffn_strided_enabled=bool(configs)
        try:yield counts
        finally:
            counts['candidate_calls']=runtime.ffn_strided_calls-before
            runtime.ffn_strided_enabled=previous
        return
    if previous is not None:runtime.ffn_strided_enabled=False
    library=math_library();owned={id(m) for m,_ in runtime.saved}
    from benchmarks.strided_ffn import linear as strided_linear
    for module in model.modules():
        if (type(module).__name__!='MossAudioTokenizerTransformerLayer'
            or id(module.linear1) not in owned or id(module.linear2) not in owned):continue
        original=module._ff_block
        def forward(self,x,original=original):
            if (not configs or observed(self) or self.activation is not F.gelu or self.gating is not None
                or self.weights_per_step or type(self.norm2) is not torch.nn.LayerNorm
                or ('forward' in self.norm2.__dict__ and not owned_norm_forward(self.norm2))
                or x.ndim!=3 or x.shape[-1] not in (768,1280) or x.shape[1]<=1
                or x.stride()!=(x.shape[1]*x.shape[2],1,x.shape[1])
                or x.dtype!=torch.float32 or x.requires_grad or torch.is_autocast_enabled('cuda')
                or not self.linear1.weight.is_contiguous() or not self.linear2.weight.is_contiguous()):return original(x)
            width=x.shape[-1];rows=x.numel()//width
            if (rows,width,4*width) not in configs:return original(x)
            normalized=self.norm2(x)
            if observed(self) or self.activation is not F.gelu:
                update=self.linear2(self.activation(self.linear1(normalized)))
                return x.to(update)+self.layer_scale_2(update)
            hidden=self.linear1(normalized,_fast_epilogue=('gelu',None,None,library))
            out=strided_linear(hidden.reshape(rows,4*width),self.linear2.weight,x,self.layer_scale_2.scale,library,runtime.cuda_bindings)
            counts['candidate_calls']+=1
            return out
        saved.append((module,original));module._ff_block=MethodType(forward,module)
    try:yield counts
    finally:
        torch.cuda.synchronize()
        for module,original in reversed(saved):module._ff_block=original
        if previous is not None:runtime.ffn_strided_enabled=previous


def compare(ref,out):
    return [{**difference(a,b),'bits_equal':torch.equal(
        a.view(torch.int32) if a.dtype==torch.float32 else a,
        b.view(torch.int32) if b.dtype==torch.float32 else b)} for a,b in zip(ref,out)]


@torch.inference_mode()
def main():
    global RUNTIME, ENCODER_EPILOGUES
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',action='store_true')
    parser.add_argument('--encoder-epilogues',choices=['both'],default='both')

    parser.add_argument('--fidelity-only', action='store_true')
    parser.add_argument('--timing-only', action='store_true')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--residual-backend', choices=['triton','cute'], default='triton')
    parser.add_argument('--output', default='results/full_strided_ffn.json')
    args=parser.parse_args();RUNTIME=args.runtime;ENCODER_EPILOGUES=args.encoder_epilogues
    if RUNTIME and ENCODER_EPILOGUES!="both":parser.error("Runtime encoder policy is selected by its implementation")
    if args.rounds<1 or (args.timing_only and args.fidelity_only):parser.error('Invalid round count or incompatible scope flags')
    from fast_moss.short_ffn import CONFIGS
    configs=dict(CONFIGS)
    if not configs:raise SystemExit('No exact faster selected configurations')
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='cuda',ffn_backend='triton',norm_backend='cuda');opts['residual_backend']=args.residual_backend
    report={'scope':('supported CUDA runtime' if RUNTIME else 'research strided FFN epilogues')+', full checkpoint and paired current-runtime ablation','candidate_backend':'cuda','encoder_epilogues':ENCODER_EPILOGUES,
        'previous_commit':'531fd32','revision':REVISION,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'sources':sources,'options':opts,
        'configs':[{'shape':s,'config':c} for s,c in configs.items()],
        'fidelity_skipped':args.timing_only,'timing_rounds':args.rounds,
        'timing_scope':'rotating independently restored contexts; five samples of ten graph calls, input copies and owned outputs included; setup/capture/restoration excluded',
        'cases':[],'timings':[],'enabled_in_runtime':RUNTIME}
    output=Path(args.output)
    def save():output.write_text(json.dumps(report,indent=2)+'\n')
    def run(x):
        e=model._encode_frame(x)
        return e.audio_codes,e.encoder_hidden_states,model._decode_frame(e.audio_codes).audio
    def inputs():
        yield from cases()
        for source,clip in zip(sources,clips):
            for batch,frames in [(2,1),(4,1),(8,1),(4,2),(1,2),(1,3),(8,3),(2,3),(1,6),(1,8),(1,12)]:
                idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
                yield f'{Path(source["path"]).stem}_b{batch}_f{frames}',clip[idx%clip.numel()][:,None],source
    for name,value,source in ([] if args.timing_only else inputs()):
        x=value.cuda();ref=run(x)
        with optimized(model,**opts),selected(model,configs) as counts:
            eager=run(x);graph=GraphedCallable(run,x)
            checks={'eager':compare(ref,eager),'graph':compare(ref,graph(x))}
            del graph,eager
        checks['restored']=compare(ref,run(x))
        exact=all(c['bits_equal'] for values in checks.values() for c in values)
        report['cases'].append({'name':name,'shape':list(x.shape),'checks':checks,
                               **counts,'all_exact':exact})
        print('fidelity',name,exact,counts,flush=True);save()
        if not exact:raise SystemExit('Model fidelity mismatch')
        del ref,x
    for batch,frames in ([] if args.fidelity_only else [(1,1),(8,1),(1,3)]):
        idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
        x=clips[1][idx%clips[1].numel()][:,None];e=model._encode_frame(x);codes=e.audio_codes
        refs={'encode':(codes,e.encoder_hidden_states),'decode':(model._decode_frame(codes).audio,)}
        def encode(z):
            e=model._encode_frame(z);return e.audio_codes,e.encoder_hidden_states
        fns=[('encode',encode,x),('decode',lambda z:(model._decode_frame(z).audio,),codes)]
        record={'batch':batch,'frames':frames,'rounds':[]}
        for repeat in range(args.rounds):
            names=['current','candidate'] if repeat%2==0 else ['candidate','current']
            for backend in names:
                entry={'round':repeat,'backend':backend,'results':{}}
                with optimized(model,**opts),selected(model,configs if backend=='candidate' else {}) as counts:
                    for _,fn,inp in fns:fn(inp)
                    for name,fn,inp in fns:
                        before=(model._fast_matrix_runtime.ffn_strided_calls if RUNTIME else counts['candidate_calls'])
                        graph=GraphedCallable(fn,inp)
                        checks=compare(refs[name],graph(inp))
                        if not all(c['bits_equal'] for c in checks):raise SystemExit('Timing fidelity mismatch')
                        def run_many():
                            for _ in range(10):graph(inp)
                        timing=measure(run_many,warmup=2,repeats=5)
                        after=(model._fast_matrix_runtime.ffn_strided_calls if RUNTIME else counts['candidate_calls'])
                        entry['results'][name]={'checks':checks,'candidate_capture_calls':after-before,'wall_ms':[v/10 for v in timing['wall_ms_samples']]}
                        del graph
                entry.update(counts);record['rounds'].append(entry)
                print('timing',batch,frames,repeat,backend,
                      {k:statistics.median(v['wall_ms']) for k,v in entry['results'].items()},flush=True)
        record['medians_ms']={name:{backend:statistics.median(v for r in record['rounds']
            if r['backend']==backend for v in r['results'][name]['wall_ms'])
            for backend in ('current','candidate')} for name,_,_ in fns}
        record['speedups']={name:t['current']/t['candidate'] for name,t in record['medians_ms'].items()}
        report['timings'].append(record);save()
    report['all_exact']=all(r['all_exact'] for r in report['cases']) and all(c['bits_equal'] for t in report['timings'] for r in t['rounds'] for v in r['results'].values() for c in v['checks'])
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()


if __name__=='__main__':main()
