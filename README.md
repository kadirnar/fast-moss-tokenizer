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

Three rotating rounds, 200 graph warmups and 30 timed calls per method/shape. Both paths use the same native precision. Graph timings include input/output copies. File I/O, resampling, model loading and setup are excluded. The graph adapter removes a host length synchronization for complete frames. [Measurements](results/v2_pointwise_compare.json) · [Methodology](docs/v2.md).

**Validation:** **1,168 tests pass**. Tokens, encoder hidden states and stereo waveforms are bit-identical across 14 corpus cases, changed graph inputs and native streaming checks. See [offline results](results/v2_pointwise_fidelity.json) and [streaming results](results/v2_pointwise_streaming_b1.json). A 100× whole-model speedup has not been achieved.

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
from fast_moss.v2 import load_model, optimized, codec
from fast_moss.graphs import GraphedCallable

model = load_model()  # Pinned v2 checkpoint; 48 kHz stereo.
audio = torch.zeros(1, 2, 3840, device="cuda")  # 80 ms, two channels.

with torch.inference_mode(), optimized(model):
    replay = GraphedCallable(lambda x: codec(model, x), audio)
    codes, hidden_states, reconstructed_audio = replay(audio)
    # Reuse with new audio of the same shape, dtype and device.
    del replay
```

The graph API accepts complete 80 ms multiples with equal-length batch items. For arbitrary lengths or native streaming, use the model's `encode` / `decode` APIs inside `optimized(model)`. [Streaming example and API details](docs/v2.md).

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
| `benchmarks/` · `tests/` | Reproducible measurements and correctness checks |

## Development

```bash
.venv/bin/python -m benchmarks.fetch_audio
.venv/bin/python -m benchmarks.v2_compare --warmups 200 --output results/v2_pointwise_compare.json
.venv/bin/python -m benchmarks.v2_fidelity --output results/v2_pointwise_fidelity.json
.venv/bin/python -m pytest -q
```

[V2 implementation](docs/v2.md) · [Legacy v1: 24 kHz mono](docs/v1.md) · [Development log](docs/progress.md)

## License

[Apache 2.0](LICENSE). Based on [OpenMOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2).
