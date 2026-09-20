"""Full streaming evaluation of experimental split-cache FP32 attention.

Token changes and waveform rounding are separate metrics. This does not enable
this reduction order in fast_moss or establish perceptual quality preservation.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio.functional as AF
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession
from benchmarks.experimental_attention import split_attention
from benchmarks.compare import difference
from benchmarks.baseline import measure


@contextmanager
def experimental(enabled=True):
    original=F.scaled_dot_product_attention
    counter={'calls':0}
    def forward(q,k,v,bias=None,**kwargs):
        if (enabled and q.dtype==torch.float32 and q.shape[2]<=32 and bias is not None
                and bias.dtype==torch.float32 and bias.shape==(q.shape[0],1,q.shape[2],k.shape[2])
                and kwargs.get('dropout_p',0.)==0. and not kwargs.get('is_causal',False)
                and kwargs.get('scale') is None and not kwargs.get('enable_gqa',False)):
            counter['calls']+=1
            return split_attention(q,k,v,bias,64)
        return original(q,k,v,bias,**kwargs)
    with patch('torch.nn.functional.scaled_dot_product_attention',forward):yield counter


def audio_difference(reference,candidate):
    report=difference(reference,candidate)
    energy=reference.double().square().sum()
    error=(reference.double()-candidate.double()).square().sum()
    report['snr_db']=float(10*torch.log10(energy/error)) if error>0 and energy>0 else None
    return report


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audio',nargs='+',default=['data/music.ogg','data/speech.wav','data/environment.wav'])
    p.add_argument('--frames',type=int,default=160)
    p.add_argument('--chunk-frames',type=int,default=1)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--output',default='results/full_attention_experiment.json')
    a=p.parse_args()
    if a.frames%a.chunk_frames:raise ValueError('Use complete chunks')
    model=load_model()
    report={'scope':'experimental full-checkpoint streaming attention; not promoted', 'revision':REVISION,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'dtype':'float32','tf32':False,
            'quantizers':32,'batch':a.batch,'frames':a.frames,'chunk_frames':a.chunk_frames,'cases':[]}
    def run(direction,inp,candidate):
        step=a.chunk_frames*(1920 if direction=='encode' else 1)
        parts=list(inp.split(step,-1))
        with experimental(candidate) as counter:
            with StreamingSession(model,direction,a.batch,a.chunk_frames) as session:
                outputs=[session.push(part)[0] for part in parts]
                timing=measure(lambda:session.push(parts[-1]),repeats=20)
                # Count is Python capture/eager calls; replay invokes device kernels directly.
                captures=counter['calls']
        return torch.cat(outputs,-1),timing,captures
    with optimized(model,residual_backend='triton',rope_backend='triton',kv_backend='triton',
                   share_rope_tables=True,attention_mask_backend='triton'):
        for path in a.audio:
            data,sr=sf.read(path,dtype='float32',always_2d=True)
            mono=torch.from_numpy(data).mean(1)
            if sr!=24000:mono=AF.resample(mono,sr,24000)
            length=a.frames*1920
            repeats=(length+mono.numel()-1)//mono.numel()
            mono=mono.repeat(repeats)[:length]
            x=torch.stack([mono.roll(i*1920) for i in range(a.batch)])[:,None].cuda()
            record={'audio':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    'repetitions_to_fill_duration':repeats,'seconds_per_lane':length/24000}
            ref_codes,ref_enc,_=run('encode',x,False)
            cand_codes,cand_enc,calls=run('encode',x,True)
            record['codes']=difference(ref_codes,cand_codes)
            record['changed_codes_per_quantizer']=(ref_codes!=cand_codes).sum((1,2)).tolist()
            record['encoder']={'reference_graph':ref_enc,'experimental_graph':cand_enc,'capture_calls':calls}
            print(path,'codes',record['codes'],flush=True)
            ref_audio,ref_dec,_=run('decode',ref_codes,False)
            cand_audio,cand_dec,calls=run('decode',ref_codes,True)
            record['decoder_fixed_codes']=audio_difference(ref_audio,cand_audio)
            record['decoder']={'reference_graph':ref_dec,'experimental_graph':cand_dec,'capture_calls':calls}
            if torch.equal(ref_codes,cand_codes):
                record['roundtrip']=record['decoder_fixed_codes']
            else:
                roundtrip,_,_=run('decode',cand_codes,True)
                record['roundtrip']=audio_difference(ref_audio,roundtrip)
            report['cases'].append(record)
            print(path,'audio',record['decoder_fixed_codes'],'roundtrip',record['roundtrip'],flush=True)
            Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    report['all_codes_exact']=all(c['codes']['exact'] for c in report['cases'])
    report['all_audio_exact']=all(c['roundtrip']['exact'] for c in report['cases'])
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
