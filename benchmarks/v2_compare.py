"""Original v2 tensor arithmetic versus graph adapter and cached v2 runtime."""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics

import torch

from benchmarks.baseline import measure
from benchmarks.codec_compare import source_state
from benchmarks.residual_gemv_async_model import compare
from benchmarks.v2_baseline import stereo_source
from fast_moss.graphs import GraphedCallable
from fast_moss.v2 import codec, load_model, optimized
from fast_moss.v2_loading import MODEL_ID, REVISION


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',default='results/v2_compare.json')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--frames',type=int,nargs='+',default=[1,3])
    p.add_argument('--batches',type=int,nargs='+',default=[1,2])
    p.add_argument('--warmups',type=int,default=100)
    args = p.parse_args()
    model = load_model()
    clip,source = stereo_source()
    def native(x):
        enc = model._encode_frame(x)
        return enc.audio_codes,enc.encoder_hidden_states,model._decode_frame(enc.audio_codes).audio
    def fixed(x): return codec(model,x)
    modes = ['original_eager','original_graph_adapter','optimized_graph']
    report = {'scope':'48 kHz stereo v2 direct encode-to-decode; original eager vs same original tensor arithmetic with fixed full-frame graph adapter vs cached native BF16 linear casts and fused FP32 quantizer',
              **source_state(), 'model_id':MODEL_ID, 'revision':REVISION,
              'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(),
              'sampling_rate':48000, 'channels':2, 'dtype_policy':model.get_codec_dtype_summary(),
              'attention_implementation':model.attention_implementation,'quantizers':32,'source':source,
              'rounds':args.rounds,'graph_warmups':args.warmups,
              'timing_scope':'rotating modes; ten owned-output calls per mode/round; GPU input/model resident; excludes loading, resampling, cache construction and graph capture',
              'cases':[]}
    def save(): Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    for batch in args.batches:
        for frames in args.frames:
            index = torch.arange(frames*3840,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*3840
            x = clip[:,index%clip.shape[1]].permute(1,0,2).contiguous()
            ref = native(x)
            initial = compare(ref,fixed(x))
            assert all(c['bits_equal'] for c in initial),initial
            record = {'batch':batch,'frames':frames,'input_shape':list(x.shape),
                      'audio_seconds':frames*.08,'adapter_checks':initial,'rounds':[]}
            for repeat in range(args.rounds):
                for mode in modes[repeat%3:]+modes[:repeat%3]:
                    gc.collect();torch.cuda.empty_cache()
                    with optimized(model) if mode=='optimized_graph' else nullcontext() as owner:
                        graph = None if mode=='original_eager' else GraphedCallable(fixed,x)
                        fn = native if graph is None else graph
                        checks = compare(ref,fn(x))
                        assert all(c['bits_equal'] for c in checks),(mode,checks)
                        if graph is not None:
                            for _ in range(args.warmups): graph(x)
                        torch.cuda.synchronize()
                        timing = measure(lambda:fn(x),warmup=3,repeats=10)
                        counts = ({k:getattr(owner,k) for k in ['cached_bytes','linear_modules','conv_modules','quantizer_modules',
                                                              'linear_calls','prepare_calls','select_calls']} if owner else {})
                        del fn,graph
                    restored = compare(ref,native(x))
                    assert all(c['bits_equal'] for c in restored)
                    record['rounds'].append({'round':repeat,'mode':mode,'checks':checks,'restored':restored,
                                             'timing':timing,'counts':counts})
                    print(batch,frames,repeat,mode,timing['wall_ms_median'],counts,flush=True)
            record['medians_ms'] = {mode:statistics.median(v for r in record['rounds'] if r['mode']==mode
                                                         for v in r['timing']['wall_ms_samples']) for mode in modes}
            record['speedups'] = {mode:record['medians_ms'][mode]/record['medians_ms']['optimized_graph'] for mode in modes[:2]}
            record['all_exact'] = all(c['bits_equal'] for r in record['rounds'] for k in ['checks','restored'] for c in r[k])
            report['cases'].append(record)
            save()
    report['all_exact'] = all(c['all_exact'] for c in report['cases'])
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    save()


if __name__=='__main__':
    main()
