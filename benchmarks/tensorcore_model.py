"""Full-codec numerical/latency probe for the strongest warm tensor-core candidate.

It did not win the eviction check. This probe tests whether warm component gains
survive the codec and records rounding changes; it is not a runtime promotion.
"""
from contextlib import contextmanager
from types import MethodType
import json
from pathlib import Path
import torch
from benchmarks.tensorcore_mm import linear
from benchmarks.cublaslt_model import choices,experimental,folds_to_mm
from benchmarks.lane_completion import audio_sources
from benchmarks.compare import difference
from benchmarks.baseline import measure
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable


@contextmanager
def tensorcore(model,tile):
    saved=[];count={'calls':0}
    def wrap(original):
        def forward(self,x):
            if (x.dtype==torch.float32 and x.is_contiguous() and self.weight.is_contiguous()
                    and self.bias is None and folds_to_mm(x) and (x.numel()//x.shape[-1],*self.weight.shape)==(24,3840,1280)):
                count['calls']+=1
                return linear(x.reshape(24,1280),self.weight,tile,mode='tf32x3').reshape(*x.shape[:-1],3840)
            return original(x)
        return forward
    try:
        for module in model.modules():
            if type(module) is torch.nn.Linear:
                saved.append((module,module.forward,'forward' in module.__dict__))
                module.forward=MethodType(wrap(module.forward),module)
        yield count
    finally:
        for module,original,existed in reversed(saved):
            if existed:module.forward=original
            else:del module.forward


@torch.inference_mode()
def main():
    model=load_model();clips,sources=audio_sources()
    selected=choices(['results/matrices_cublaslt.json','results/matrices_cublaslt_codec8.json','results/matrices_cublaslt_large.json'],packed_only=True)
    selection=json.loads(Path('results/tensorcore_selection.json').read_text())
    tile=next(r['tile'] for r in selection['records'] if r['mode']=='tf32x3' and r['shape_MNK']==[24,3840,1280])
    report={'scope':'full-codec TF32x3 QKV probe; not promoted','revision':REVISION,'gpu':torch.cuda.get_device_name(),
            'torch':torch.__version__,'weight_storage':'float32','arithmetic':'tf32x3 for selected QKV, existing exact FP32 elsewhere',
            'quantizers':32,'sources':sources,'tile':tile,'input':'cyclic source samples; lanes offset by 1920 samples','cases':[]}
    inputs=[]
    for clip,source in zip(clips,sources):
        for batch,frames in [(8,3),(4,6),(2,12),(1,24)]:
            idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
            inputs.append((f"{source['path']}_b{batch}_f{frames}",clip[idx%clip.numel()][:,None]))
    inputs += [('silence',torch.zeros(8,1,5760,device='cuda')),('tiny',torch.full((8,1,5760),1e-38,device='cuda'))]
    def run(x):
        enc=model._encode_frame(x)
        return enc.audio_codes,enc.encoder_hidden_states,model._decode_frame(enc.audio_codes).audio
    with optimized(model,residual_backend='triton',rope_backend='triton',kv_backend='triton',share_rope_tables=True,attention_mask_backend='triton'):
        with experimental(model,selected,resident=True):
            refs=[run(x) for _,x in inputs]
            graph=GraphedCallable(run,inputs[0][1]);report['baseline_graph']=measure(lambda:graph(inputs[0][1]),repeats=30);del graph
        with experimental(model,selected,resident=True),tensorcore(model,tile) as count:
            for (name,x),ref in zip(inputs,refs):
                eager=run(x);graph=GraphedCallable(run,x);out=graph(x)
                checks={mode:{key:difference(old,new) for key,old,new in zip(['codes','hidden','audio'],ref,values)} for mode,values in [('eager',eager),('graph',out)]}
                checks['graph_vs_eager']={key:difference(old,new) for key,old,new in zip(['codes','hidden','audio'],eager,out)}
                d=(ref[2].double()-out[2].double()).square().sum();energy=ref[2].double().square().sum()
                fixed=model._decode_frame(ref[0]).audio
                fixed_difference=difference(ref[2],fixed)
                changed=(ref[0]!=out[0])
                report['cases'].append({'name':name,'comparisons':checks,
                    'fixed_code_decoder':fixed_difference,
                    'changed_code_locations_quantizer_lane_frame':changed.nonzero().tolist(),
                    'changed_codes_per_quantizer':changed.sum((1,2)).tolist(),
                    'audio_snr_db':float(10*torch.log10(energy/d)) if d>0 and energy>0 else None})
                print(name,checks['graph']['codes']['exact'],'audio_max',checks['graph']['audio']['max_abs'],flush=True)
                if name==inputs[0][0]:report['candidate_graph']=measure(lambda:graph(x),repeats=30)
                del graph
                Path('results/full_tensorcore_probe.json').write_text(json.dumps(report,indent=2)+'\n')
            report['qkv_capture_and_eager_calls']=count['calls']
    report['all_codes_exact']=all(r['comparisons']['graph']['codes']['exact'] for r in report['cases'])
    report['all_audio_exact']=all(r['comparisons']['graph']['audio']['exact'] for r in report['cases'])
    Path('results/full_tensorcore_probe.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
