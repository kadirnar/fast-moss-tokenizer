"""Full-codec gates for measured exact cuBLASLt choices; research only."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import MethodType
import torch
import soundfile as sf
import torchaudio.functional as AF
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from benchmarks.cublaslt import LinearPlan,library
from benchmarks.packed_weights import PackedWeights
from benchmarks.compare import difference
from benchmarks.baseline import measure


def choices(paths,packed_only=False):
    groups={}
    for path in paths:
        current={}
        report=json.loads(Path(path).read_text())
        if (report['cublaslt_version']!=library().cublasLtGetVersion()
                or report['gpu']!=torch.cuda.get_device_name() or report['torch']!=torch.__version__):
            raise ValueError('Tuning requires the recorded GPU, PyTorch, and cuBLASLt versions')
        for r in report['records']:
            r['cublaslt_version']=report['cublaslt_version']
            current.setdefault(tuple(r['shape_MNK']),[]).append(r)
        groups.update(current)
    selected={}
    for shape,rows in groups.items():
        base=next(r for r in rows if r['backend']=='torch')
        good=[r for r in rows if r['backend'].startswith('lt_') and r['reference_difference']['exact']
              and (not packed_only or r['backend'].startswith('lt_packed_'))
              and r['graph_vs_eager']['exact'] and r['hot_graph_ms']<base['hot_graph_ms']/1.05
              and r['evicted_replay']['gpu_ms_median']<base['evicted_replay']['gpu_ms_median']/1.1]
        if good:selected[shape]=min(good,key=lambda r:r['evicted_replay']['gpu_ms_median'])
    return selected


def folds_to_mm(x):
    # PyTorch 2.8 should_fold(), for a frozen 2-D right operand. Even a tensor
    # reported contiguous can fail this check when singleton strides differ.
    return all(x.stride(i)==x.stride(i+1)*x.shape[i+1] for i in range(x.ndim-2))


@contextmanager
def experimental(model,selected,resident=False):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Experimental matrix replacement requires a frozen evaluation model')
    if getattr(model,'_fast_cublaslt_active',False):
        raise RuntimeError('This model already has a matrix replacement context')
    if resident and any(v['backend'].split('_')[1]!='packed' for v in selected.values()):
        raise ValueError('Resident mode currently requires packed algorithms')
    saved=[];plans={};workspace=torch.empty(32*1024*1024,device=next(model.parameters()).device,dtype=torch.uint8)
    packing=PackedWeights(model,resident)
    def wrap(original):
        def forward(self,x):
            shape=(x.numel()//x.shape[-1],*self.weight.shape)
            if (shape not in selected or self.bias is not None or x.dtype!=torch.float32
                    or not x.is_contiguous() or not folds_to_mm(x) or x.data_ptr()%256
                    or x.requires_grad or torch.is_autocast_enabled('cuda')):
                if id(self.weight) in packing.entries and resident:
                    return torch.nn.functional.linear(x,self.weight.contiguous(),self.bias)
                return original(x)
            key=(id(self),shape)
            if key not in plans:
                record=selected[shape];layout=record['backend'].split('_')[1]
                packed=packing.get(self.weight) if layout=='packed' else None
                plan=LinearPlan(self.weight,shape[0],layout,workspace=workspace,packed_weight=packed)
                try:index=plan.restore(record['algorithm'],record['cublaslt_version'])
                except BaseException:
                    plan.close();raise
                plans[key]=(plan,index)
            plan,index=plans[key]
            return plan(x.reshape(shape[0],shape[-1]),index).reshape(*x.shape[:-1],self.out_features)
        return forward
    try:
        model._fast_cublaslt_active=True
        for module in model.modules():
            if type(module) is torch.nn.Linear:
                saved.append((module,module.forward,'forward' in module.__dict__))
                module.forward=MethodType(wrap(module.forward),module)
        yield plans
    finally:
        for module,original,existed in reversed(saved):
            if existed:module.forward=original
            else:del module.forward
        for plan,index in plans.values():plan.close()
        plans.clear()
        # Drop the loop's final reference before restoring one weight at a time.
        if 'plan' in locals():del plan
        packing.close()
        del model._fast_cublaslt_active


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tuning',nargs='+',default=['results/matrices_cublaslt.json','results/matrices_cublaslt_codec8.json'])
    p.add_argument('--output',default='results/full_cublaslt_experiment.json')
    p.add_argument('--resident',action='store_true')
    a=p.parse_args();selected=choices(a.tuning,packed_only=a.resident);model=load_model()
    report={'scope':'full-codec cuBLASLt pedantic FP32 experiment, not promoted','revision':REVISION,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'dtype':'float32','tf32':False,
            'cublaslt_version':library().cublasLtGetVersion(),
            'resident_weights':a.resident,
            'quantizers':32,'selection_threshold':'exact component output, >5% warm and >10% cold gain',
            'selected':[{'shape':k,'backend':v['backend'],'algorithm':v['algorithm']} for k,v in selected.items()],
            'cases':[]}
    inputs=[]
    for path in ['data/music.ogg','data/speech.wav','data/environment.wav']:
        y,sr=sf.read(path,dtype='float32',always_2d=True);y=torch.from_numpy(y).mean(1)
        if sr!=24000:y=AF.resample(y,sr,24000)
        for batch,frames in [(8,3),(4,6),(2,12),(1,24)]:
            offsets=torch.arange(frames*1920)[None]+torch.arange(batch)[:,None]*1920
            inputs.append((f'{Path(path).stem}_b{batch}_f{frames}',y[offsets%y.numel()][:,None].cuda(),
                           {'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                            'transform':'cyclic samples, lane starts offset by 1920 samples'}))
    inputs.extend([('silence',torch.zeros(8,1,5760,device='cuda'),None),
                   ('tiny_signal',torch.full((8,1,5760),1e-38,device='cuda'),None)])
    def run(x):
        enc=model._encode_frame(x)
        return enc.audio_codes,enc.encoder_hidden_states,model._decode_frame(enc.audio_codes).audio
    with optimized(model,residual_backend='triton',rope_backend='triton',kv_backend='triton',
                   share_rope_tables=True,attention_mask_backend='triton'):
        references=[run(x) for _,x,_ in inputs]
        base=GraphedCallable(run,inputs[0][1])
        report['baseline_graph']=measure(lambda:base(inputs[0][1]),repeats=20)
        del base
        with experimental(model,selected,resident=a.resident) as plans:
            for (name,x,source),ref in zip(inputs,references):
                eager=run(x)
                graph=GraphedCallable(run,x)
                replay=graph(x)
                r={'name':name,'input_shape':list(x.shape),'source':source,'comparisons':{}}
                for mode,outputs in [('eager',eager),('graph',replay)]:
                    r['comparisons'][mode]={k:difference(old,new) for k,old,new in zip(['codes','hidden','audio'],ref,outputs)}
                report['cases'].append(r)
                print(name,'exact',all(v['exact'] for c in r['comparisons'].values() for v in c.values()),flush=True)
                if name==inputs[0][0]:report['experimental_graph']=measure(lambda:graph(x),repeats=20)
                del graph
                Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
            report['plan_count']=len(plans)
            unique={p.weight.data_ptr():p.weight for p,_ in plans.values() if p.layout=='packed'}
            report['packed_weight_bytes']=sum(p.numel()*4 for p in unique.values())
            report['additional_weight_bytes']=0 if a.resident else report['packed_weight_bytes']
            del unique
        restored=run(inputs[0][1])
        report['restored']={k:difference(old,new) for k,old,new in zip(['codes','hidden','audio'],references[0],restored)}
    report['all_exact']=all(v['exact'] for c in report['cases'] for mode in c['comparisons'].values() for v in mode.values())
    report['all_exact'] &= all(v['exact'] for v in report['restored'].values())
    report['peak_memory_bytes']=torch.cuda.max_memory_allocated()
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Full-codec fidelity failed')


if __name__=='__main__':main()
