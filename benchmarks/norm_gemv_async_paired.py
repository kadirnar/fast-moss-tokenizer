"""Interleaved owned-graph pairs with one verified stable weight lifetime."""
import json,statistics,time
from pathlib import Path
import torch
from benchmarks.norm_gemv_async_model import selected,compare
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable

@torch.inference_mode()
def main():
    selection=json.loads(Path('results/norm_gemv_async_ring_confirm.json').read_text());configs={}
    for row in selection['records']:
        ratios=row['rings'][-1]['speedups'];best=max(ratios,key=ratios.get)
        if ratios[best]>1.005:configs[row['mode']]=tuple(json.loads(best))
    model=load_model();clips,sources=audio_sources();opts=dict(options(),matrix_backend='cuda',ffn_backend='triton',norm_backend='cuda')
    report={'scope':'40 alternating graph pairs per direction and geometry, 200 warmups, five owned calls per sample; same verified weight addresses/layouts after warming both methods; setup/capture excluded; copies/clones included',
        'previous_commit':'b40b3b4','revision':REVISION,'sources':sources,'configs':configs,'options':opts,'records':[]}
    for batch,frames in [(1,1),(8,1),(1,3)]:
        idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
        x=clips[1][idx%clips[1].numel()][:,None];e=model._encode_frame(x);codes=e.audio_codes
        refs={'encode':(codes,e.encoder_hidden_states),'decode':(model._decode_frame(codes).audio,)}
        def encode(z):
            e=model._encode_frame(z);return e.audio_codes,e.encoder_hidden_states
        operations={'encode':(encode,x),'decode':(lambda z:(model._decode_frame(z).audio,),codes)}
        graphs={};counts={};checks=[];samples=[]
        with optimized(model,**opts):
            for mode in ('current','candidate'):
                with selected(model,configs if mode=='candidate' else {}):
                    for fn,value in operations.values():fn(value)
            def storage():return [(p.data_ptr(),tuple(p.stride())) for p in model.parameters()]
            pointers=storage()
            for mode in ('current','candidate'):
                with selected(model,configs if mode=='candidate' else {}) as counter:
                    for direction,(fn,value) in operations.items():
                        graph=GraphedCallable(fn,value);c=compare(refs[direction],graph(value));assert all(v['bits_equal'] for v in c)
                        checks.append({'mode':mode,'direction':direction,'checks':c});graphs[(mode,direction)]=graph
                counts[mode]=dict(counter);assert storage()==pointers
            for _ in range(200):
                for (mode,direction),graph in graphs.items():graph(operations[direction][1])
            torch.cuda.synchronize()
            for repeat in range(40):
                modes=['current','candidate'] if repeat%2==0 else ['candidate','current']
                for direction,(_,value) in operations.items():
                    for mode in modes:
                        torch.cuda.synchronize();start=time.perf_counter()
                        for _ in range(5):graphs[(mode,direction)](value)
                        torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000/5
                        samples.append({'pair':repeat,'mode':mode,'direction':direction,'ms':ms})
                assert storage()==pointers
            for (mode,direction),graph in graphs.items():assert all(c['bits_equal'] for c in compare(refs[direction],graph(operations[direction][1])))
            graphs.clear();del graph
        for direction,(fn,value) in operations.items():assert all(c['bits_equal'] for c in compare(refs[direction],fn(value)))
        medians={d:{m:statistics.median(r['ms'] for r in samples if r['direction']==d and r['mode']==m) for m in ('current','candidate')} for d in operations}
        pair_ratios={d:[next(r['ms'] for r in samples if r['pair']==p and r['mode']=='current' and r['direction']==d)/next(r['ms'] for r in samples if r['pair']==p and r['mode']=='candidate' and r['direction']==d) for p in range(40)] for d in operations}
        row={'batch':batch,'frames':frames,'checks':checks,'counts':counts,'stable_storage':True,'restored_bits_equal':True,'samples':samples,'medians_ms':medians,
            'speedups':{d:v['current']/v['candidate'] for d,v in medians.items()},'paired_speedups':pair_ratios,'candidate_wins':{d:sum(v>1 for v in values) for d,values in pair_ratios.items()}}
        report['records'].append(row);Path('results/full_norm_gemv_async_paired.json').write_text(json.dumps(report,indent=2)+'\n')
        print(batch,frames,row['speedups'],row['candidate_wins'],counts,flush=True)
    report['all_exact']=True;Path('results/full_norm_gemv_async_paired.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
