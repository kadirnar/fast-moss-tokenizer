"""Full-checkpoint heterogeneous final lengths, continuous lane reuse, and overhead."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import soundfile as sf
import torchaudio.functional as AF
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession
from benchmarks.lane_fixtures import schedule,independent_reference
from benchmarks.compare import difference
from benchmarks.baseline import measure


def audio_sources():
    clips=[];metadata=[]
    for path in ['data/music.ogg','data/speech.wav','data/environment.wav']:
        samples,sr=sf.read(path,dtype='float32',always_2d=True)
        x=torch.from_numpy(samples).mean(1)
        if sr!=24000:x=AF.resample(x,sr,24000)
        clips.append(x.cuda())
        metadata.append({'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    return clips,metadata


def fill_schedule(entries,direction,sources):
    positions=[0]*3;closed=[False]*3
    for entry in entries:
        for lane in entry['reset']:positions[lane]=0;closed[lane]=False
        for lane,(n,end) in enumerate(zip(entry['lengths'],entry['ends'])):
            view=entry['chunk'][lane] if direction=='encode' else entry['chunk'][:,lane]
            view.fill_(float('nan') if direction=='encode' else -999)
            if n and not closed[lane]:
                source=sources[lane]
                idx=(torch.arange(n,device='cuda')+positions[lane])%source.shape[-1]
                view[...,:n]=source[...,idx]
                positions[lane]+=n
            closed[lane] |= end


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--warm-steps',type=int,default=65)
    p.add_argument('--output',default='results/full_lane_completion.json')
    a=p.parse_args();model=load_model();clips,metadata=audio_sources()
    source_audio=torch.stack([x[torch.arange(19200,device='cuda')%x.numel()] for x in clips])[:,None]
    source_codes=model._encode_frame(source_audio).audio_codes
    report={'scope':'full checkpoint heterogeneous streaming completion and lane reuse',
            'revision':REVISION,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'dtype':'float32','tf32':False,'quantizers':32,'batch':3,'chunk_frames':2,
            'warm_steps':a.warm_steps,'continuous_lane_seconds_before_tail':a.warm_steps*.16,
            'sources':metadata,'encoder_input':'source audio repeated as needed; fresh requests restart at sample zero',
            'decoder_input':'tokens from the first 0.8 seconds of each source, repeated as needed',
            'reference':'independent eager timelines with the same batch shape, not a performance baseline',
            'results':{}}
    for direction in ['encode','decode']:
        entries=schedule(direction,a.warm_steps)
        sources=clips if direction=='encode' else [source_codes[:,i] for i in range(3)]
        fill_schedule(entries,direction,sources)
        refs=independent_reference(model,direction,entries)
        print(direction,'reference finished',flush=True)
        dim=1 if direction=='encode' else 0;records=[]
        with optimized(model,residual_backend='triton',kv_backend='triton',rope_backend='triton',
                       share_rope_tables=True,attention_mask_backend='triton'):
            with StreamingSession(model,direction,3,chunk_frames=2) as session:
                graph=None
                for i,entry in enumerate(entries):
                    if entry['reset']:
                        session.reset(torch.tensor([b in entry['reset'] for b in range(3)],device='cuda'))
                    out,lengths=session.push(entry['chunk'],valid_lengths=entry['lengths'],final_lanes=entry['ends'])
                    if graph is None:graph=session.graph
                    record={'step':i,'input_lengths':entry['lengths'],'final_lanes':entry['ends'],
                            'reset_lanes':entry['reset'],'output_lengths':lengths.tolist(),
                            'same_graph':session.graph is graph,'lanes':[]}
                    for lane,ref in refs[i].items():
                        n=0 if ref is None else ref.shape[-1]
                        record['lanes'].append({'lane':lane,'length_exact':lengths[lane].item()==n,
                                                'fidelity':difference(ref,out.select(dim,lane)[...,:n]) if n else None})
                    records.append(record)
            # A matched all-active workload isolates control overhead; it is not
            # a sequential baseline and does not time reference construction.
            timing={}
            example=(source_audio[...,:3840] if direction=='encode' else source_codes[...,:2])
            width=example.shape[-1]
            for mode in ['uniform','lane_controls']:
                with StreamingSession(model,direction,3,chunk_frames=2) as session:
                    def push():
                        return session.push(example,**({'valid_lengths':[width]*3} if mode=='lane_controls' else {}))
                    for _ in range(65):push()
                    timing[mode]=measure(push,repeats=20)
            reset_timing={}
            mask=torch.tensor([True,False,True],device='cuda')
            for fast in [False,True]:
                with StreamingSession(model,direction,3,chunk_frames=2,use_graph=False,fast_reset=fast) as session:
                    reset_timing['triton' if fast else 'upstream']=measure(lambda:session.reset(mask),repeats=30)
        report['results'][direction]={'steps':records,'timing':timing,
            'reset_timing':reset_timing,
            'all_exact':all(r['same_graph'] and all(v['length_exact'] and (v['fidelity'] is None or v['fidelity']['exact'])
                            for v in r['lanes']) for r in records)}
        print(direction,'exact',report['results'][direction]['all_exact'],'timing',timing,flush=True)
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=all(r['all_exact'] for r in report['results'].values())
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Lane-completion fidelity failed')


if __name__=='__main__':main()
