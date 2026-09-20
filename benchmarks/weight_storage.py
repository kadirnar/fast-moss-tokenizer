"""CPU-only exact representation audit of captured checkpoint GEMV weights."""
import json
from pathlib import Path
import torch
from fast_moss.loading import REVISION


def main():
    cases=torch.load('results/matrix_inputs.pt',weights_only=True,map_location='cpu')
    records=[]
    for shape,case in sorted(cases.items()):
        if shape[0]!=1:continue
        w=case['weight'];bits=w.view(torch.int32);exponent=((bits>>23)&255).reshape(-1)
        counts=torch.bincount(exponent,minlength=256);p=counts[counts>0].double()/w.numel()
        entropy=-(p*p.log2()).sum().item()
        blocks=exponent.reshape(-1,256)
        ranges=blocks.max(1).values-blocks.min(1).values+1
        widths=torch.ceil(torch.log2(ranges.double())).long()
        # Representation cost estimate only: preserve sign+23 fraction bits,
        # block minimum exponent and delta width, and a four-byte block offset.
        encoded_bytes=w.numel()*3+(widths*32).sum().item()+len(widths)*6
        records.append({'shape':shape,'elements':w.numel(),
            'bf16_representable_elements':((bits&65535)==0).sum().item(),
            'fp16_roundtrip_exact_elements':(w==w.half().float()).sum().item(),
            'lowest_byte_zero_elements':((bits&255)==0).sum().item(),
            'exponent_entropy_bits':entropy,'block256_delta_width_histogram':torch.bincount(widths,minlength=9).tolist(),
            'estimated_block256_encoded_bytes':encoded_bytes,'fp32_bytes':w.numel()*4,
            'estimated_storage_ratio':w.numel()*4/encoded_bytes})
    report={'scope':'CPU-only exact storage-representation audit of six captured one-row checkpoint weights',
            'revision':REVISION,'compression_status':'No encoder/decoder implemented; block-format sizes are arithmetic estimates, not measured GPU traffic or speedup.',
            'records':records}
    Path('results/weight_storage_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in records:print(r['shape'],'exact BF16 fraction',r['bf16_representable_elements']/r['elements'],'estimated storage ratio',r['estimated_storage_ratio'])

if __name__=='__main__':main()
