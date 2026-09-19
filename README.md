# Fast MOSS Audio Tokenizer

Ongoing GPU optimization of the **original 1.6B MOSS Audio Tokenizer**, retaining FP32 weights, all 32 quantizers, and its learned architecture. No distillation, FP8, or FP4. **100× whole-model acceleration has not been demonstrated.**

Implemented: reversible inference caches for normalized codebooks and convolution weights; bitwise FP32 residual fusion in Triton and CuTe DSL (explicit CUDA PTX rounding); Triton RoPE with optional stage-shared tables, fused/shared attention masks, and ring-cache kernels; CUDA graphs; incremental encoder/decoder sessions with independently pausable/resettable batch lanes.

Full-checkpoint measurements on the RTX 5070 Ti, batch 1, 240 ms input, FP32, all 32 codebooks:

| Operation | Upstream eager | Optimized graph | Speedup |
| --- | ---: | ---: | ---: |
| Encode | 48.369 ms | 9.491 ms | 5.10× |
| Decode | 38.571 ms | 7.821 ms | 4.93× |

These matched measurements include graph input copies and owned output tensors. Setup is separate. Tokens and waveforms match exactly in this run. [Raw samples and setup costs](results/full_compare_masks_240ms.json) include reference graph-only and caching ablations. **This is not a 100× result.**

Both Triton and CuTe residual paths, combined with Triton RoPE, shared tables, and fused attention masks, pass exact token/hidden-state/audio checks on 12 initial real-audio and edge-case inputs: [Triton](results/full_fidelity_masks.json), [CuTe](results/full_fidelity_masks_cute.json). Two 12.8-second music streams cross the ten-second cache context with exact tokens versus offline encoding and exact waveform equality versus corrected eager streaming. Streamed audio differs from offline decoding by at most 1.70e-6, identically in the corrected eager and optimized paths. [Long-stream evidence](results/full_streaming_masks.json). Paused/resumed/reused lanes also match independent timelines on the full model: [evidence](results/full_parallel_masks.json).

At batch 128 and 240 ms per lane, aggregate throughput reaches 355.3 audio seconds/s encode and 367.1 decode. Matched-batch speedups are only 1.06× and 1.09×: throughput relative to real time is a different quantity. The fused-mask path retains exact outputs at batches 1, 8, and 128 ([measurements](results/batching_masks.json)); the earlier shared-RoPE sweep covers all powers of two from 1–128 ([measurements](results/batching.json)).

Files prefixed `structural_` use a **reduced random test model** and are not checkpoint performance evidence. Residual kernels are roughly 1.8–2.6× faster inside amortized graphs, but eager launches regress. Fusion remains opt-in; plain `optimized(model)` only caches frozen values.

## Setup

Tested on Linux, RTX 5070 Ti 16 GB (SM120), Python 3.12, PyTorch 2.8.0 CUDA 12.8, Triton 3.4.0, CuTe DSL 4.7.1. The pinned upstream code needs a newer Transformers tokenizer base than 4.57 provides; the exact compatible Transformers commit is pinned in the dependency manifest.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python -m pytest -q
.venv/bin/python -m benchmarks.baseline --seconds .08 --profile
.venv/bin/python -m benchmarks.compare --seconds .24 --stream-chunks 3 --rope-backend triton --kv-backend triton --share-rope-tables --attention-mask-backend triton
.venv/bin/python -m benchmarks.kernels
.venv/bin/python -m benchmarks.fetch_audio
.venv/bin/python -m benchmarks.fidelity --share-rope-tables --attention-mask-backend triton
.venv/bin/python -m benchmarks.streaming_fidelity --share-rope-tables --attention-mask-backend triton
.venv/bin/python -m benchmarks.batching --batches 1 8 128 --attention-mask-backend triton
```

The model loader pins Hugging Face revision `3cd226ba2947efa357ef453bcad111b6eafba782`, requests FP32, and disables TF32. `upstream/` contains unmodified source/configuration for inspection and small structural tests, with the upstream Apache-2.0 headers preserved. Checkpoint files stay in the Hugging Face cache.

## Streaming

```python
import torch
from fast_moss import load_model
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession

model = load_model()
with torch.inference_mode(), optimized(model, residual_backend="triton",
                                      rope_backend="triton", kv_backend="triton",
                                      share_rope_tables=True, attention_mask_backend="triton"):
    # Two independent streams advance together, 80 ms per push.
    with StreamingSession(model, "encode", batch_size=2) as session:
        chunk = torch.zeros(2, 1, 1920, device="cuda")
        codes, lengths = session.push(chunk)
        # Next push retains causal KV history; outputs do not alias graph buffers.
        last_codes, last_lengths = session.push(chunk[..., :960], final=True)
```

`attention_mask_backend="triton"` constructs the same FP32 zero/negative-infinity attention bias with aligned row strides and shares it within synchronized stages. This removes repeated mask construction, conversion, and padding while preserving the attention calculation.

Shared RoPE tables are opt-in and require `rope_backend="triton"`. They reuse identical positions within each transformer stage, including synchronized session lanes. External upstream streaming falls back to per-layer tables because its offsets may differ.

Use `direction="decode"` with input shape `(32, batch, chunk_frames)`, dtype `torch.int64`. Keep graphs/sessions within the optimization context that created their cached tensors. A session owns model state and its CUDA stream; overlapping sessions on one model are rejected. Pass a CUDA boolean `active_mask` to `push` to pause lanes; their KV history/position stays unchanged and their returned length is zero. Omit the mask to advance all lanes. `reset(mask)` reuses selected lanes for new streams; `reset()` resets all lanes. All active lanes process the same chunk length. The first push includes graph warmup and capture; benchmark reports separate setup from steady state.

Partial final encoder frames are explicitly zero-padded and retained. Save the original sample count to trim reconstructed audio: a codec frame always decodes to 1920 samples. This deliberately avoids upstream's floor-length behavior, which can mark a partial final frame invalid. Cache capacity is extended by `chunk_tokens - 1` at each transformer resolution so chunk insertion cannot overwrite required history. The attention context remains unchanged. This corrects upstream behavior after the ring fills; streaming benchmarks explicitly compare graph execution against the corrected eager session. `final=True` finishes the entire session. Arbitrary per-lane final lengths and multi-GPU execution remain unimplemented.

## Fidelity and benchmark scope

`benchmarks.compare` records exact equality, mismatching element counts, maximum error, and RMSE, and exits with failure on any mismatch. It compares equal batches, audio lengths, FP32 arithmetic, and quantizer counts. Wall-clock and CUDA-event samples include graph input copies/output ownership; amortized kernel-only measurements are labeled separately. Synthetic inputs and small structural tests are development gates, not proof of quality across real speech, music, or sound effects.

Outstanding: broader audio corpus coverage, more streaming durations/batch schedules, arbitrary per-lane final lengths, multi-GPU execution, matrix-kernel optimization, and a defensible matched-workload 100× result. See [research notes](docs/research.md) and [work log](docs/progress.md).
