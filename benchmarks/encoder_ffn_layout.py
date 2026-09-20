"""Record baseline FFN input/output layouts without installing module observers."""
import json
from pathlib import Path
from types import MethodType
from collections import Counter
import torch
from fast_moss.loading import load_model
from fast_moss.optimize import optimized
from benchmarks.ordered_model import options
records=[]
@torch.inference_mode()
def main():
 model=load_model();opts=dict(options(),matrix_backend='cuda',ffn_backend='triton',norm_backend='cuda')
 for frames in [1,3]:
  with optimized(model,**opts):
   model._fast_matrix_runtime.ffn_strided_enabled=False
   for name,module in model.named_modules():
    if not hasattr(module,'_fast_ffn_runtime'):continue
    original=module._ff_block
    def wrapped(self,x,original=original,name=name):
     before=self._fast_ffn_runtime.ffn_short_calls
     out=original(x)
     records.append(dict(frames=frames,name=name,shape=list(x.shape),stride=list(x.stride()),output_stride=list(out.stride()),calls=self._fast_ffn_runtime.ffn_short_calls-before))
     return out
    module._ff_block=MethodType(wrapped,module)
   x=torch.randn(1,1,frames*1920,device='cuda');e=model._encode_frame(x);model._decode_frame(e.audio_codes)
 Path('results/encoder_ffn_layout.json').write_text(json.dumps(records,indent=2)+'\n')
 for key,n in Counter((r['frames'],r['name'].split('.')[0],tuple(r['shape']),tuple(r['stride']),tuple(r['output_stride']),r['calls']) for r in records).items():print(key,n)
if __name__=='__main__':main()
