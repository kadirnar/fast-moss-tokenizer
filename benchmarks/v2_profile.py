"""Profile original and optimized v2 roundtrip graphs without changing precision."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import gc
import gzip
import json
from pathlib import Path

import torch

from benchmarks.codec_compare import source_state
from benchmarks.residual_gemv_async_model import compare
from benchmarks.v2_baseline import stereo_source
from fast_moss.graphs import GraphedCallable
from fast_moss.v2 import codec, load_model, optimized
from fast_moss.v2_loading import MODEL_ID, REVISION


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',default='results/v2_profile.json')
    args = p.parse_args()
    model = load_model()
    clip,source = stereo_source()
    x = clip[:,:3840][None].contiguous()
    enc = model._encode_frame(x)
    ref = enc.audio_codes,enc.encoder_hidden_states,model._decode_frame(enc.audio_codes).audio
    report = {'scope':'batch-one 80 ms v2 direct encode/decode graphs; CUDA kernel durations under profiler, not headline wall latency',
              **source_state(),'model_id':MODEL_ID,'revision':REVISION,'source':source,
              'dtype_policy':model.get_codec_dtype_summary(),'attention_implementation':model.attention_implementation,
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'replays':5,'warmups':200,'modes':{}}
    for mode in ['original_graph_adapter','optimized_graph']:
        gc.collect();torch.cuda.empty_cache()
        with optimized(model) if mode=='optimized_graph' else nullcontext():
            graph = GraphedCallable(lambda z:codec(model,z),x)
            checks = compare(ref,graph(x))
            assert all(c['bits_equal'] for c in checks)
            for _ in range(200): graph(x)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(5): graph(x)
                torch.cuda.synchronize()
            trace = Path(args.output).with_suffix('.'+mode+'.json.gz')
            prof.export_chrome_trace(str(trace))
            events = json.loads(gzip.open(trace,'rt').read())['traceEvents']
            kernels = [e for e in events if e.get('cat')=='kernel']
            grouped = defaultdict(lambda:[0,0.])
            for event in kernels:
                grouped[event['name']][0] += 1
                grouped[event['name']][1] += event['dur']
            total = sum(e['dur'] for e in kernels)
            rows = [{'name':name,'calls_per_replay':count/5,'ms_per_replay':duration/5000,
                     'percent_kernel_time':100*duration/total} for name,(count,duration) in grouped.items()]
            rows.sort(key=lambda row:row['ms_per_replay'],reverse=True)
            report['modes'][mode] = {'checks':checks,'kernel_count_per_replay':len(kernels)/5,
                                    'kernel_ms_per_replay':total/5000,'kernels':rows}
            del graph
        print(mode,len(kernels)/5,total/5000,flush=True)
    report['all_exact'] = all(c['bits_equal'] for m in report['modes'].values() for c in m['checks'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
