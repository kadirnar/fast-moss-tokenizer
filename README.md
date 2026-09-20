# Fast-MOSS-Tokenizer

Fast inference engine for [MOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2), a neural audio codec that compresses and reconstructs **48 kHz stereo** audio. This library accelerates end-to-end encoding and decoding up to **7.52x** through CUDA Graph streaming, weight caching and fused kernels — with unchanged model weights and bit-identical outputs in validation.

## Benchmark

NVIDIA RTX 5070 Ti 16 GB | `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` (2.12B params) | 48 kHz stereo | Batch 1

Native BF16 compute, FP32 weights and quantizer, all 32 quantizers.

### End-to-End — 5s Audio

| Method | Latency | Speedup | Real-time Factor |
|--------|:-------:|:-------:|:----------------:|
| PyTorch | 10.092 s | 1.00x | 0.50x |
| **+ weight caching + fused kernels + graph** | **1.344 s** | **7.51x** | **3.72x** |

### End-to-End — 10s Audio

| Method | Latency | Speedup | Real-time Factor |
|--------|:-------:|:-------:|:----------------:|
| PyTorch | 19.935 s | 1.00x | 0.50x |
| **+ weight caching + fused kernels + graph** | **2.653 s** | **7.52x** | **3.77x** |

### End-to-End — 50s Audio

| Method | Latency | Speedup | Real-time Factor |
|--------|:-------:|:-------:|:----------------:|
| PyTorch | 99.004 s | 1.00x | 0.51x |
| **+ weight caching + fused kernels + graph** | **13.169 s** | **7.52x** | **3.80x** |

Median of three full encode → decode runs with GPU-resident inputs; excludes model loading, graph setup and file I/O. Real-time factor = audio duration / latency; higher is faster.

## Quick Start

```bash
pip install git+https://github.com/kadirnar/fast-moss-tokenizer.git
```

```bash
python -m fast_moss input.wav output.wav
```

```python
from fast_moss import load_model, optimized, StreamingCodec
import torch

model = load_model()
audio = torch.randn(1, 2, 2400000, device="cuda")  # 50s, 48 kHz stereo

with optimized(model), StreamingCodec(model) as codec:
    codes, hidden_states, output = codec(audio)
    # Reuse codec for more files; each call starts a fresh stream.
```

## Requirements

- Python 3.12, Linux
- PyTorch 2.8.0 / CUDA 12.8
- NVIDIA RTX 5070 Ti 16 GB

## License

Apache 2.0
