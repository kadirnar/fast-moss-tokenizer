# Fast-MOSS-Tokenizer

Fast inference for [MOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2), a neural audio codec that converts **48 kHz stereo** audio into tokens and reconstructs it. Accelerates **encode → decode up to 8.99×** on RTX 5070 Ti using cached weights, CUDA/Triton kernels and CUDA graphs. Outputs match the original v2 model bit for bit on the validation corpus.

Uses v2's native BF16 codec compute with FP32 weights and quantizer. No retraining, distillation, FP8 or FP4.

## End-to-End Benchmark (Encode → Decode)

**NVIDIA RTX 5070 Ti 16 GB** | **MOSS-Audio-Tokenizer-v2, 2.12B parameters** | **48 kHz stereo** | **32 quantizers** | **SDPA**

Direct end-to-end **encode → decode**, with the encoder's tokens fed to the decoder in one call:

| Batch | Audio per item | Original eager | Original graph adapter | Optimized graph | vs. eager | vs. graph |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 80 ms | 139.262 ms | 30.795 ms | **15.492 ms** | **8.99×** | 1.99× |
| 1 | 240 ms | 140.755 ms | 32.116 ms | **16.352 ms** | **8.61×** | 1.96× |
| 2 | 80 ms | 140.508 ms | 31.110 ms | **15.745 ms** | **8.92×** | 1.98× |
| 2 | 240 ms | 141.661 ms | 33.489 ms | **17.699 ms** | **8.00×** | 1.89× |

Three rotating rounds, 200 graph warmups and 30 timed calls per method/shape. Both paths use the same native precision. Graph timings include input/output copies. File I/O, resampling, model loading and setup are excluded. The graph adapter removes a host length synchronization for complete frames.

Token, hidden-state and stereo waveform outputs matched the original v2 model bit for bit in the validation runs.

## Quick Start

```bash
git clone --depth 1 https://github.com/kadirnar/fast-moss-tokenizer.git
cd fast-moss-tokenizer
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock -e .
```

Process a **48 kHz stereo WAV** file:

```bash
.venv/bin/python -m fast_moss input.wav output.wav
```

The command uses native streaming with 80 ms chunks and writes an FP32 WAV, preserving the input length and both channels.

## Python API

For the fixed-frame CUDA graph path measured above, run with `.venv/bin/python`:

```python
import torch
from fast_moss import load_model, optimized, codec, GraphedCallable

model = load_model()  # Pinned v2 checkpoint; 48 kHz stereo.
audio = torch.zeros(1, 2, 3840, device="cuda")  # 80 ms, two channels.

with torch.inference_mode(), optimized(model):
    replay = GraphedCallable(lambda x: codec(model, x), audio)
    codes, hidden_states, reconstructed_audio = replay(audio)
    # Reuse with new audio of the same shape, dtype and device.
    del replay
```

The graph API accepts complete 80 ms multiples with equal-length batch items. For arbitrary lengths or native streaming, use the model's `encode` / `decode` APIs inside `optimized(model)`.

## Requirements

- Linux, Python 3.12, **RTX 5070 Ti 16 GB**.
- Pinned PyTorch **2.8.0 / CUDA 12.8**, Triton **3.4.0**, NVRTC and CUDA bindings from `requirements.lock`.
- About **13.15 GB** peak allocated GPU memory in the benchmark. The runtime validates the tested GPU and checkpoint.

## Project Structure

| Module | Responsibility |
| --- | --- |
| [v2.py](fast_moss/v2.py) | Public API |
| [v2_loading.py](fast_moss/v2_loading.py) | Pinned checkpoint and precision settings |
| [v2_codec.py](fast_moss/v2_codec.py) | Fixed-frame encode/decode adapters |
| [v2_runtime.py](fast_moss/v2_runtime.py) | Reversible caches and optimization context |
| [v2_pointwise.py](fast_moss/v2_pointwise.py) | Exact RoPE and residual Triton kernels |

## License

[Apache 2.0](LICENSE). Based on [OpenMOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2).
