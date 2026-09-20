"""Host-driven fragmented input packing, including allocation and metadata upload."""
import json
from pathlib import Path
import statistics

import torch

from benchmarks.baseline import measure
from fast_moss.batching import _gather_chunks


@torch.inference_mode()
def main():
    torch.manual_seed(8302)
    report = {'scope': 'packing component only; eager calls include output allocation and GPU metadata upload',
              'baseline': 'allocate zero batch, concatenate each active lane, copy into batch',
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'records': []}
    for encode in [True, False]:
        channels, width = (1, 5760) if encode else (32, 3)
        dtype = torch.float32 if encode else torch.long
        for batch in [8, 32, 128]:
            segments = []
            for lane in range(batch):
                parts = []
                if lane % 4 != 3:
                    lengths = [1, max(2, width // 2) - 1, width - max(2, width // 2)]
                    if lane % 4 == 2:
                        lengths = lengths[:2]
                    for n in lengths:
                        value = (torch.randn(channels, n + 2, device='cuda') if encode
                                 else torch.randint(1024, (channels, n + 2), device='cuda'))
                        parts.append((value, 1, n))
                segments.append(parts)
            shape = (batch, 1, width) if encode else (channels, batch, width)
            def concatenate():
                result = torch.zeros(shape, device='cuda', dtype=dtype)
                for lane, parts in enumerate(segments):
                    if parts:
                        joined = torch.cat([x[..., start:start+n] for x, start, n in parts], -1)
                        destination = result[lane] if encode else result[:, lane]
                        destination[..., :joined.shape[-1]].copy_(joined)
                return result
            def fused():
                return _gather_chunks(segments, width=width, channels=channels, encode=encode, device='cuda')
            assert torch.equal(concatenate(), fused())
            rounds = {'concatenate': [], 'fused': []}
            for repeat in range(3):
                methods = [('concatenate', concatenate), ('fused', fused)]
                if repeat % 2:
                    methods.reverse()
                for name, fn in methods:
                    rounds[name].append(measure(fn, warmup=3, repeats=30))
            medians = {name: statistics.median(r['wall_ms_median'] for r in entries)
                       for name, entries in rounds.items()}
            entry = {'direction': 'encode' if encode else 'decode', 'batch': batch,
                     'width': width, 'exact': True, 'rounds': rounds, 'wall_ms_medians': medians,
                     'component_gain': medians['concatenate'] / medians['fused']}
            report['records'].append(entry)
            print(entry['direction'], batch, medians, entry['component_gain'], flush=True)
    Path('results/fragment_gather.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
