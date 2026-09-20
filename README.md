# Fast-MOSS-Tokenizer

Fast inference for [MOSS Audio Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer), a neural audio codec that converts audio into tokens and reconstructs it. Accelerates encoding up to **7.24×** and decoding up to **6.44×** using CUDA, Triton and CUDA graphs, with unchanged FP32 weights and bit-identical outputs on the validation corpus.

Supports incremental streaming, parallel batch lanes and queued requests. No distillation, FP8 or FP4.

## Benchmark

**NVIDIA RTX 5070 Ti 16 GB** | Original checkpoint | FP32 | 32 quantizers | Mono 24 kHz | Batch 1

| Audio length | Operation | PyTorch eager | PyTorch graph | Optimized graph | vs. eager | vs. graph |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 80 ms | Encode | 46.738 ms | 10.343 ms | **6.455 ms** | **7.24×** | 1.60× |
| 80 ms | Decode | 37.016 ms | 9.104 ms | **5.744 ms** | **6.44×** | 1.58× |
| 240 ms | Encode | 47.555 ms | 11.806 ms | **7.357 ms** | **6.46×** | 1.60× |
| 240 ms | Decode | 37.921 ms | 10.170 ms | **6.184 ms** | **6.13×** | 1.64× |

Steady-state medians from three rotating rounds, with 200 extra graph warmups. Graph timings include input copies and owned outputs; model loading, packing and graph capture are excluded. [80 ms results](results/full_codec_norm_async_f1.json) · [240 ms results](results/full_codec_norm_async_f3.json).

**Validation:** 1,055 tests pass; full-checkpoint gates compare tokens, hidden states and waveform bits. Long streaming checks match corrected eager streaming. Streaming and offline decoding have an existing small rounding difference, documented in the [streaming reference](docs/implementation.md#streaming). Experimental kernels are excluded from the table. A 100× whole-model speedup has not been achieved.

## Quick Start

```bash
git clone https://github.com/kadirnar/fast-moss-tokenizer.git
cd fast-moss-tokenizer
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock
```

Run with `.venv/bin/python`:

```python
import torch
from fast_moss import load_model
from fast_moss.graphs import GraphedCallable
from fast_moss.optimize import optimized

model = load_model()  # Downloads the pinned original FP32 checkpoint.
audio = torch.zeros(1, 1, 1920, device="cuda")  # 80 ms at 24 kHz.


def encode_decode(audio):
    codes = model._encode_frame(audio).audio_codes
    return codes, model._decode_frame(codes).audio


with torch.inference_mode(), optimized(
    model,
    residual_backend="triton", rope_backend="triton", kv_backend="triton",
    share_rope_tables=True, attention_mask_backend="triton",
    quantizer_backend="triton", projection_backend="triton",
    matrix_backend="cuda", ffn_backend="triton", norm_backend="cuda",
):
    replay = GraphedCallable(encode_decode, audio)
    codes, reconstructed_audio = replay(audio)
    # Reuse replay with new audio of the same shape, dtype and device.
    del replay
```

Keep graphs and streaming sessions inside the optimization context. Model weights and methods are restored on exit.

## Streaming

`StreamingSession` processes 80 ms chunks while retaining causal history. Batch lanes can pause, finish and restart independently. `StreamingBatcher` queues requests of different lengths and fills available lanes. Both run on one GPU.

See the [streaming example](docs/implementation.md#streaming) and [queued request example](docs/implementation.md#queued-streaming-requests).

## Requirements

- Linux, Python 3.12 and an NVIDIA GPU.
- The benchmark configuration requires **RTX 5070 Ti (SM120)**, **PyTorch 2.8.0 / CUDA 12.8** and **Triton 3.4.0**. Tuned backends check the GPU, checkpoint and library versions.
- Install the pinned dependencies in `requirements.lock`; CuTe DSL 4.7.1 is included for the optional CuTe residual backend.

## Development

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m benchmarks.fetch_audio
.venv/bin/python -m benchmarks.codec_compare --frames 1 --matrix-backend cuda --norm-backend cuda --extra-warmup-replays 200
```

[Implementation and API details](docs/implementation.md) · [Research notes](docs/research.md) · [Development log](docs/progress.md)

## License

[Apache 2.0](LICENSE). Based on the original [OpenMOSS Audio Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer).
