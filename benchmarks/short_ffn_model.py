"""Full-checkpoint fidelity and restored-context short-row FFN epilogue ablation."""
import argparse
import json
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch

from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from benchmarks.short_ffn import linear,CONFIGS as RESEARCH_CONFIGS
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
    previous=getattr(runtime,'ffn_short_enabled',None)
    if RUNTIME:
        before=runtime.ffn_short_calls;runtime.ffn_short_enabled=bool(configs)
        try:yield counts
        finally:
            counts['candidate_calls']=runtime.ffn_short_calls-before
            runtime.ffn_short_enabled=previous
        return
    if previous is not None:runtime.ffn_short_enabled=False
    library=math_library();RESEARCH_CONFIGS.update(configs)
    encoder={id(m) for m in model.encoder.modules()}
    owned={id(m) for m,_ in runtime.saved}
    for module in model.modules():
        if (type(module).__name__!='MossAudioTokenizerTransformerLayer'
            or id(module.linear1) not in owned or id(module.linear2) not in owned):continue
        original=module._ff_block
        def forward(self,x,original=original):
            if (not configs or observed(self) or self.activation is not F.gelu or self.gating is not None
                or self.weights_per_step or type(self.norm2) is not torch.nn.LayerNorm
                or ('forward' in self.norm2.__dict__ and not owned_norm_forward(self.norm2))
                or not runtime.ffn_enabled or x.ndim<2 or x.shape[-1] not in (768,1280)
                or not x.is_contiguous() or x.dtype!=torch.float32 or x.requires_grad
                or torch.is_autocast_enabled('cuda')
                or not self.linear1.weight.is_contiguous() or not self.linear2.weight.is_contiguous()):return original(x)
            width=x.shape[-1];rows=x.numel()//width
            first=(rows,4*width,width);second=(rows,width,4*width)
            first_enabled=first in configs and (id(self) not in encoder or ENCODER_EPILOGUES in ('both','gelu'))
            second_enabled=second in configs and (id(self) not in encoder or ENCODER_EPILOGUES in ('both','residual'))
            if not first_enabled and not second_enabled:return original(x)
            normalized=self.norm2(x)
            if observed(self) or self.activation is not F.gelu:
                update=self.linear2(self.activation(self.linear1(normalized)))
                return x.to(update)+self.layer_scale_2(update)
            if first_enabled:
                hidden=linear(normalized.reshape(rows,width),self.linear1.weight,'gelu',library=library,bindings=runtime.cuda_bindings)
                hidden=hidden.reshape(*x.shape[:-1],4*width);counts['candidate_calls']+=1
            else:hidden=self.activation(self.linear1(normalized))
            if second_enabled:
                out=linear(hidden.reshape(rows,4*width),self.linear2.weight,'residual',x.reshape(rows,width),self.layer_scale_2.scale,library,runtime.cuda_bindings)
                counts['candidate_calls']+=1;return out.reshape(x.shape)
            return self._fast_scale_add(x,self.linear2(hidden),self.layer_scale_2.scale)
        saved.append((module,original));module._ff_block=MethodType(forward,module)
    try:yield counts
    finally:
        try:torch.cuda.synchronize()
        finally:
            for module,original in reversed(saved):module._ff_block=original
            if previous is not None:runtime.ffn_short_enabled=previous


def compare(ref,out):
    return [{**difference(a,b),'bits_equal':torch.equal(
        a.view(torch.int32) if a.dtype==torch.float32 else a,
        b.view(torch.int32) if b.dtype==torch.float32 else b)} for a,b in zip(ref,out)]


@torch.inference_mode()
def main():
    global RUNTIME, linear, ENCODER_EPILOGUES
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',action='store_true')
    parser.add_argument('--encoder-epilogues',choices=['both','gelu','residual','none'],default='both')

    parser.add_argument('--fidelity-only', action='store_true')
    parser.add_argument('--timing-only', action='store_true')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--residual-backend', choices=['triton','cute'], default='triton')
    parser.add_argument('--output', default='results/full_short_ffn.json')
    args=parser.parse_args();RUNTIME=args.runtime;ENCODER_EPILOGUES=args.encoder_epilogues
    if RUNTIME and ENCODER_EPILOGUES!="both":parser.error("Runtime encoder policy is selected by its implementation")
    if args.rounds<1 or (args.timing_only and args.fidelity_only):parser.error('Invalid round count or incompatible scope flags')
    confirmation=json.loads(Path('results/short_ffn_probe.json').read_text())
    configs={tuple(r['shape']):(r['family'],tuple(r['config'])) for r in confirmation['records']
        if r['all_exact'] and r['rings'][-1]['speedup']>1.005}
    retuned=json.loads(Path('results/short_ffn_confirm.json').read_text())
    for r in retuned['records']:
        speeds=r['rings'][-1]['speedups'];name=max(speeds,key=speeds.get)
        if speeds[name]>1.005:configs[tuple(r['shape'])]=('wide',tuple(json.loads(name)))
    if RUNTIME:
        from fast_moss.short_ffn import CONFIGS
        configs=dict(CONFIGS)
    if not configs:raise SystemExit('No exact faster selected configurations')
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='cuda',ffn_backend='triton',norm_backend='cuda');opts['residual_backend']=args.residual_backend
    report={'scope':('supported CUDA runtime' if RUNTIME else 'research short-row FFN epilogues')+', full checkpoint and paired current-runtime ablation','candidate_backend':'cuda','encoder_epilogues':ENCODER_EPILOGUES,
        'previous_commit':'c67e3a9','revision':REVISION,'torch':torch.__version__,
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
                        before=(model._fast_matrix_runtime.ffn_short_calls if RUNTIME else counts['candidate_calls'])
                        graph=GraphedCallable(fn,inp)
                        checks=compare(refs[name],graph(inp))
                        if not all(c['bits_equal'] for c in checks):raise SystemExit('Timing fidelity mismatch')
                        def run_many():
                            for _ in range(10):graph(inp)
                        timing=measure(run_many,warmup=2,repeats=5)
                        after=(model._fast_matrix_runtime.ffn_short_calls if RUNTIME else counts['candidate_calls'])
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
