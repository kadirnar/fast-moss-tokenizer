"""Full-checkpoint long streaming gates for measured cuBLASLt FP32 choices."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from benchmarks.cublaslt import library
from benchmarks.cublaslt_model import choices,experimental
from benchmarks.lane_completion import audio_sources
from benchmarks.compare import difference
from benchmarks.baseline import measure
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession


def schedule(step,width):
    lengths=[width]*8;ends=[False]*8;reset=[]
    if step%5==1:lengths[1]=0
    if step<8:lengths[2]=0
    if step==12:lengths[3]=max(1,width//2);ends[3]=True
    if step==22:reset.append(3)
    if step==17:lengths[4]=0;ends[4]=True
    if step==24:reset.append(4)
    return lengths,ends,reset


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tuning',nargs='+',default=['results/matrices_cublaslt.json','results/matrices_cublaslt_codec8.json'])
    p.add_argument('--steps',type=int,default=54)
    p.add_argument('--output',default='results/full_cublaslt_streaming.json')
    p.add_argument('--resident',action='store_true')
    a=p.parse_args();model=load_model();selected=choices(a.tuning,packed_only=a.resident);clips,sources=audio_sources()
    width=5760
    audio=torch.stack([clips[i%3][(torch.arange(a.steps*width,device='cuda')+i*1920)%clips[i%3].numel()]
                       for i in range(8)])[:,None]
    report={'scope':'full-checkpoint cuBLASLt long-stream experiment; not promoted',
            'revision':REVISION,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'cublaslt_version':library().cublasLtGetVersion(),'dtype':'float32','tf32':False,
            'resident_weights':a.resident,
            'quantizers':32,'batch':8,'chunk_frames':3,'steps':a.steps,'sources':sources,
            'input':'cyclic source samples, lanes alternate music/speech/environment with 1920-sample offsets',
            'reference':'existing exact optimized CUDA graph, identical lane schedule',
            'continuous_lane_seconds':a.steps*.24,'results':{}}
    def run(direction,parts,candidate,refs=None):
        records=[];outputs=[];valid_elements=0
        ctx=experimental(model,selected,resident=a.resident) if candidate else nullcontext({})
        with ctx as plans:
            with StreamingSession(model,direction,8,chunk_frames=3) as session:
                first_graph=None
                for i,part in enumerate(parts):
                    lengths,ends,reset=schedule(i,part.shape[-1])
                    if reset:session.reset(torch.tensor([b in reset for b in range(8)],device='cuda'))
                    out,lens=session.push(part,valid_lengths=lengths,final_lanes=ends)
                    if first_graph is None:first_graph=session.graph
                    host=out.cpu();host_lens=lens.cpu()
                    if refs is None:outputs.append((host,host_lens))
                    else:
                        dim=1 if direction=='encode' else 0
                        checks=[]
                        for lane,n in enumerate(host_lens.tolist()):
                            if n:
                                ref=refs[i][0].select(dim,lane)[...,:n]
                                actual=host.select(dim,lane)[...,:n]
                                checks.append(difference(ref,actual));valid_elements+=actual.numel()
                        records.append({'step':i,'lengths':lengths,'final_lanes':ends,'reset':reset,
                                        'output_lengths':host_lens.tolist(),
                                        'lengths_exact':torch.equal(host_lens,refs[i][1]),
                                        'same_graph':session.graph is first_graph,'lanes':checks})
                # All lanes reopen, then fill beyond ten seconds before timing.
                session.reset()
                for _ in range(43):session.push(parts[-1])
                timing=measure(lambda:session.push(parts[-1]),repeats=20)
            plan_count=len(plans)
            unique={p.weight.data_ptr():p.weight for p,_ in plans.values() if p.layout=='packed'}
            packed_bytes=sum(p.numel()*4 for p in unique.values())
            del unique
        return outputs,{'timing':timing,'plan_count':plan_count,'packed_weight_bytes':packed_bytes,
                        'steps':records,'valid_elements':valid_elements,
                        'all_exact':all(r['lengths_exact'] and r['same_graph'] and all(c['exact'] for c in r['lanes']) for r in records)}
    with optimized(model,residual_backend='triton',rope_backend='triton',kv_backend='triton',
                   share_rope_tables=True,attention_mask_backend='triton'):
        parts=list(audio.split(width,-1))
        refs,base=run('encode',parts,False)
        print('encode reference finished',flush=True)
        _,candidate=run('encode',parts,True,refs)
        report['results']['encode']={'baseline':base,'experimental':candidate}
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
        print('encode exact',candidate['all_exact'],flush=True)
        code_parts=[out.cuda() for out,_ in refs]
        refs,base=run('decode',code_parts,False)
        print('decode reference finished',flush=True)
        _,candidate=run('decode',code_parts,True,refs)
        report['results']['decode']={'baseline':base,'experimental':candidate}
        print('decode exact',candidate['all_exact'],flush=True)
    report['all_exact']=all(r['experimental']['all_exact'] for r in report['results'].values())
    report['peak_memory_bytes']=torch.cuda.max_memory_allocated()
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Full-stream fidelity failed')


if __name__=='__main__':main()
