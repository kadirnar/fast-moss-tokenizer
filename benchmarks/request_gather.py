"""Isolate the request packing kernel; whole-queue gains use request_batching."""
import json
from pathlib import Path
import torch
import triton
from triton.testing import do_bench_cudagraph
from fast_moss.batching import _gather
from benchmarks.baseline import measure


@torch.inference_mode()
def main():
    report = {'scope': 'request gather component, resident metadata, no model speed claim',
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'records': []}
    torch.manual_seed(394)
    for direction in ['encode', 'decode']:
        encode = direction == 'encode'
        channels, width = (1, 5760) if encode else (32, 3)
        dtype = torch.float32 if encode else torch.long
        for batch in [8, 32, 128]:
            requests = [torch.randn(channels, width * 3, device='cuda') if encode
                        else torch.randint(1024, (channels, width * 3), device='cuda') for _ in range(batch)]
            lengths = [width if i % 3 == 0 else max(1, width // 2) if i % 3 == 1 else 0 for i in range(batch)]
            shape = (batch, 1, width) if encode else (channels, batch, width)
            meta = torch.tensor([[x.data_ptr(), x.shape[-1], width, n] for x, n in zip(requests, lengths)],
                                device='cuda', dtype=torch.long)
            expected = torch.empty(shape, device='cuda', dtype=dtype)
            actual = torch.empty_like(expected)
            def copies():
                expected.zero_()
                for lane, (value, n) in enumerate(zip(requests, lengths)):
                    view = expected[lane] if encode else expected[:, lane]
                    if n: view[..., :n].copy_(value[..., width:width + n])
                return expected
            def fused():
                _gather[(batch, channels, triton.cdiv(width, 256))](
                    meta, actual, width, channels, batch, encode, 256)
                return actual
            assert torch.equal(copies(), fused())
            record = {'direction': direction, 'batch': batch, 'shape': shape, 'exact': True,
                      'metadata_bytes': meta.numel() * meta.element_size(), 'timing': {}}
            for name, fn in [('copies', copies), ('fused', fused)]:
                record['timing'][name] = {'graph_ms': do_bench_cudagraph(fn, rep=30, return_mode='median'),
                                         'eager': measure(fn, repeats=30)}
            record['graph_gain'] = record['timing']['copies']['graph_ms'] / record['timing']['fused']['graph_ms']
            report['records'].append(record)
            print(direction, batch, 'gain', record['graph_gain'], flush=True)
    Path('results/request_gather.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
