# Fast-MOSS-Tokenizer

Fast inference engine for [MOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2), a neural audio codec for **48 kHz stereo** audio. Speeds up full-file encoding and decoding by about **7.5x** on RTX 5070 Ti using CUDA Graph streaming, weight caching and fused kernels. Model weights stay unchanged, and outputs match bit for bit in validation. Independent short blocks reach up to **8.99x** speedup.

## Benchmark

NVIDIA RTX 5070 Ti 16 GB | `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` (2.12B params) | 48 kHz stereo | Batch 1

All methods use the model's native **BF16 compute, FP32 weights and FP32 quantizer**, with SDPA and all 32 quantizers.

### End-to-End: Long Audio (File Command)

Both methods preserve the model's native streaming history in 80 ms chunks. Fast-MOSS replays encoding and decoding with CUDA Graph, weight caching and fused kernels. This is the path used by `python -m fast_moss`.

| Audio Duration | PyTorch | Fast-MOSS | Speedup | Fast-MOSS Real-time Speed |
|----------------|:-------:|:---------:|:-------:|:------------------------:|
| 5 s | 10.092 s | 1.344 s | **7.51x** | 3.72x |
| 10 s | 19.935 s | 2.653 s | **7.52x** | 3.77x |
| 50 s | 99.004 s | 13.169 s | **7.52x** | 3.80x |

Median of three complete encode → decode runs per method, with alternating method order and a 0.96 s warmup before each run. Synchronized wall time includes state reset, streaming history updates, graph input/output copies, padding, concatenation and cropping. Excludes model loading, setup, file I/O, resampling and CPU output transfer; model and input are already on the GPU. One-time weight-cache and streaming-graph setup takes **0.81 s** (median) and can be reused across files.

One fresh-process CLI run on the 50 s file took **19.64 s total**, including Python startup, model loading, graph setup and file I/O, with checkpoint weights already downloaded.

Inputs are the first 5, 10 and 50 seconds of a [116-second stereo recording](https://commons.wikimedia.org/wiki/File:Anthem_of_Europe_(US_Navy_instrumental_long_version).ogg), resampled from 44.1 to 48 kHz with both channels preserved and no looping. The 5 s input is padded to 5.04 s internally and cropped back to 5 s. Tokens, hidden states and reconstructed audio matched bit for bit in all runs; peak allocated GPU memory was **13.26 GB**.

### End-to-End: 80 ms Audio (CUDA Graph)

| Method | Latency | Speedup | Real-time Speed |
|--------|:-------:|:-------:|:---------------:|
| PyTorch | 139.262 ms | 1.00x | 0.57x |
| + CUDA Graph | 30.795 ms | 4.52x | 2.60x |
| **+ weight caching + fused kernels** | **15.492 ms** | **8.99x** | **5.16x** |

### End-to-End: 240 ms Audio (CUDA Graph)

| Method | Latency | Speedup | Real-time Speed |
|--------|:-------:|:-------:|:---------------:|
| PyTorch | 140.755 ms | 1.00x | 1.71x |
| + CUDA Graph | 32.116 ms | 4.38x | 7.47x |
| **+ weight caching + fused kernels** | **16.352 ms** | **8.61x** | **14.68x** |

End-to-end means **audio → encode → tokens → decode → audio**, measured in one call. Speedup is relative to PyTorch; real-time speed is audio duration divided by latency. Higher is faster.

The 80 ms and 240 ms tables show steady-state medians with GPU-resident inputs. Includes graph input/output copies; excludes file I/O, resampling, loading and setup. CUDA Graph uses a fixed-frame adapter. Tokens, hidden states and reconstructed audio matched bit for bit in validation.

## Quick Start

```bash
uv venv --python 3.12
uv pip install git+https://github.com/kadirnar/fast-moss-tokenizer.git
```

Process a **48 kHz stereo WAV** of any length:

```bash
.venv/bin/python -m fast_moss input.wav output.wav
```

The command uses CUDA Graph streaming in 80 ms chunks and saves an FP32 WAV with the original length and both channels. The first file includes model loading and graph setup.

To reuse the model and graph across files:

```python
import soundfile as sf
import torch
from fast_moss import load_model, optimized, StreamingCodec

model = load_model()
with optimized(model), StreamingCodec(model) as codec:
    audio, sample_rate = sf.read("input.wav", dtype="float32", always_2d=True)
    assert sample_rate == 48000 and audio.shape[1] == 2
    waveform = torch.from_numpy(audio.T.copy()).unsqueeze(0).cuda()
    codes, hidden_states, reconstructed = codec(waveform)
    sf.write("output.wav", reconstructed[0].cpu().T.numpy(), 48000, subtype="FLOAT")
    # Call codec again for another file; history resets at the start of each call.
```

`StreamingCodec` accepts one FP32 stereo CUDA waveform at a time, shaped `(1, 2, samples)`, with any positive length. Keep it inside the optimization context and use the same thread and CUDA stream. Model weights and settings must stay fixed while the context is active.

For independent fixed-length blocks, `codec` and `GraphedCallable` remain available:

```python
from fast_moss import codec, GraphedCallable

with torch.inference_mode(), optimized(model):
    audio = torch.zeros(1, 2, 3840, device="cuda")  # 80 ms stereo.
    replay = GraphedCallable(lambda x: codec(model, x), audio)
    codes, hidden_states, reconstructed = replay(audio)
    del replay
```

This fixed-frame path accepts equal-length batch items in complete 80 ms multiples and does not carry streaming history between calls.

## Requirements

- Linux, Python 3.12, **NVIDIA RTX 5070 Ti 16 GB**.
- PyTorch **2.8.0 / CUDA 12.8**, Triton **3.4.0** and NVRTC **12.8.93** (installed with the package).
- Approximately **13.26 GB** peak allocated GPU memory in the long-audio benchmark.

## License

[Apache 2.0](LICENSE).
