# Fast-MOSS-Tokenizer

Fast inference engine for [MOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2), a neural audio codec that compresses and reconstructs **48 kHz stereo** audio. Accelerates end-to-end encoding and decoding up to **8.99x** on RTX 5070 Ti through weight caching, fused CUDA/Triton kernels and CUDA graphs, with unchanged model weights and bit-identical outputs in validation.

## Benchmark

NVIDIA RTX 5070 Ti 16 GB | `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` (2.12B params) | 48 kHz stereo | Batch 1

All methods use the model's native **BF16 compute, FP32 weights and FP32 quantizer**, with SDPA and all 32 quantizers.

### End-to-End: 80 ms Audio

| Method | Latency | Speedup | Real-time Speed |
|--------|:-------:|:-------:|:---------------:|
| PyTorch | 139.262 ms | 1.00x | 0.57x |
| + CUDA Graph | 30.795 ms | 4.52x | 2.60x |
| **+ weight caching + fused kernels** | **15.492 ms** | **8.99x** | **5.16x** |

### End-to-End: 240 ms Audio

| Method | Latency | Speedup | Real-time Speed |
|--------|:-------:|:-------:|:---------------:|
| PyTorch | 140.755 ms | 1.00x | 1.71x |
| + CUDA Graph | 32.116 ms | 4.38x | 7.47x |
| **+ weight caching + fused kernels** | **16.352 ms** | **8.61x** | **14.68x** |

End-to-end means **audio → encode → tokens → decode → audio**, measured in one call. Speedup is relative to PyTorch; real-time speed is audio duration divided by latency. Higher is faster.

Steady-state medians with GPU-resident inputs. Includes graph input/output copies; excludes file I/O, resampling, loading and setup. CUDA Graph uses a fixed-frame adapter. Tokens, hidden states and reconstructed audio matched bit for bit in validation.

## Quick Start

```bash
uv venv --python 3.12
uv pip install git+https://github.com/kadirnar/fast-moss-tokenizer.git
```

Run with `.venv/bin/python`:

```python
import torch
from fast_moss import load_model, optimized, codec, GraphedCallable

model = load_model()
audio = torch.zeros(1, 2, 3840, device="cuda")  # 80 ms at 48 kHz, stereo.

with torch.inference_mode(), optimized(model):
    replay = GraphedCallable(lambda x: codec(model, x), audio)
    codes, hidden_states, reconstructed_audio = replay(audio)
    # Reuse replay with new audio of the same shape, dtype and device.
    del replay
```

The graph path accepts equal-length batch items in complete 80 ms multiples. Keep the graph inside the optimization context.

To process a **48 kHz stereo WAV** of any length:

```bash
.venv/bin/python -m fast_moss input.wav output.wav
```

The file command uses native streaming in 80 ms chunks and saves an FP32 WAV with the original length and both channels. The tables above measure the fixed-frame graph path.

## Requirements

- Linux, Python 3.12, **NVIDIA RTX 5070 Ti 16 GB**.
- PyTorch **2.8.0 / CUDA 12.8**, Triton **3.4.0** and NVRTC **12.8.93** (installed with the package).
- Approximately **13.15 GB** peak allocated GPU memory in the benchmark.

## License

[Apache 2.0](LICENSE).
