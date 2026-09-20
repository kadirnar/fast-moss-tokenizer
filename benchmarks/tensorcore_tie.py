"""Inspect the first discrete decision changed by the tensor-core codec probe."""
from contextlib import contextmanager
from types import MethodType
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.tensorcore_model import tensorcore
from benchmarks.cublaslt_model import choices,experimental
from benchmarks.lane_completion import audio_sources
from fast_moss.loading import load_model
from fast_moss.optimize import optimized


@torch.inference_mode()
def main():
    model=load_model();clips,sources=audio_sources()
    selected=choices(['results/matrices_cublaslt.json','results/matrices_cublaslt_codec8.json','results/matrices_cublaslt_large.json'],packed_only=True)
    probe=json.loads(Path('results/full_tensorcore_probe.json').read_text())
    failed=next(c for c in probe['cases'] if c['name']=='data/speech.wav_b8_f3')
    if not failed['changed_code_locations_quantizer_lane_frame']:
        raise RuntimeError('The recorded speech regression no longer changes codes')
    quantizer,lane,frame=failed['changed_code_locations_quantizer_lane_frame'][0]
    tile=probe['tile'];idx=torch.arange(5760,device='cuda')[None]+torch.arange(8,device='cuda')[:,None]*1920
    x=clips[1][idx%clips[1].numel()][:,None]
    module=next(m for name,m in model.named_modules() if name.endswith(f'quantizers.{quantizer}') and type(m).__name__=='MossAudioTokenizerLFQ')
    collected={};mode='reference'
    @contextmanager
    def observe():
        original=module.decode_latents;existed='decode_latents' in module.__dict__
        def wrapped(self,latents):
            enc=F.normalize(latents.transpose(1,2).reshape(-1,latents.shape[1]).float())
            dist=enc.pow(2).sum(1,keepdim=True)-2*enc@self._fast_codebook.t()+self._fast_codebook_norm
            output=original(latents)
            values,ids=dist[lane*latents.shape[-1]+frame].topk(4,largest=False)
            collected[mode]={'chosen':int(output[1][lane,frame]),'nearest_ids':ids.tolist(),'nearest_distances':values.tolist(),
                             'margin':float(values[1]-values[0]),'normalized_latent':enc[lane*latents.shape[-1]+frame].tolist()}
            return output
        module.decode_latents=MethodType(wrapped,module)
        try:yield
        finally:
            if existed:module.decode_latents=original
            else:del module.decode_latents
    with optimized(model,residual_backend='triton',rope_backend='triton',kv_backend='triton',share_rope_tables=True,attention_mask_backend='triton'):
        with experimental(model,selected,resident=True),observe():model._encode_frame(x)
        mode='candidate'
        with experimental(model,selected,resident=True),tensorcore(model,tile),observe():model._encode_frame(x)
    report={'scope':'same full-batch distance arithmetic at first changed quantizer decision',
            'source':sources[1],'quantizer_zero_based':quantizer,'lane_zero_based':lane,'frame_zero_based':frame,
            'results':collected}
    Path('results/tensorcore_quantizer_tie.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':main()
