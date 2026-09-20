# Fast MOSS Audio Tokenizer

Ongoing GPU optimization of the **original 1.6B MOSS Audio Tokenizer**, retaining FP32 weights, all 32 quantizers, and its learned architecture. No distillation, FP8, or FP4. **100× whole-model acceleration has not been demonstrated.**

Implemented: reversible inference caches for normalized codebooks and convolution weights; bitwise FP32 residual fusion in Triton and CuTe DSL (explicit CUDA PTX rounding); Triton RoPE with optional stage-shared tables, fused/shared attention masks, and ring-cache kernels; CUDA graphs; incremental encoder/decoder sessions with independently pausable, finishable, and reusable batch lanes; fused lane reset; incremental request scheduling with fused fragment gather and optional input byte limits; optional exact quantizer fusion.

Full-checkpoint measurements on the RTX 5070 Ti, batch 1, 240 ms input, FP32, all 32 codebooks:

| Operation | Upstream eager | Optimized graph | Speedup |
| --- | ---: | ---: | ---: |
| Encode | 47.894 ms | 9.491 ms | 5.05× |
| Decode | 38.317 ms | 7.820 ms | 4.90× |

These matched measurements include graph input copies and owned output tensors. Setup is separate. Tokens and waveforms match exactly in this run. [Raw samples and setup costs](results/full_compare_lane_reset_240ms.json) include reference graph-only and caching ablations. **This is not a 100× result.**

Both Triton and CuTe residual paths, combined with Triton RoPE, shared tables, and fused attention masks, pass exact token/hidden-state/audio checks on 12 initial real-audio and edge-case inputs: [Triton](results/full_fidelity_masks.json), [CuTe](results/full_fidelity_masks_cute.json). Two 12.8-second music streams cross the ten-second cache context with exact tokens versus offline encoding and exact waveform equality versus corrected eager streaming. Streamed audio differs from offline decoding by at most 1.70e-6, identically in the corrected eager and optimized paths. [Long-stream evidence](results/full_streaming_masks.json). Paused/resumed/reused lanes also match independent timelines on the full model: [evidence](results/full_parallel_masks.json).

The expanded 13-case fidelity corpus includes an eight-lane speech case whose closest codebook distances differ by only 1.19e-7. Both supported backends remain exact: [Triton](results/full_fidelity_near_tie.json), [CuTe](results/full_fidelity_near_tie_cute.json).

`quantizer_backend="triton"` adds exact distance selection, embedding/straight-through fusion, and residual updates. In a separate paired ablation, batch-one / 240 ms encoder time falls **9.488 → 9.012 ms** (5.0% lower) against the preceding optimized graph; batch eight falls **16.406 → 15.805 ms**. All 32 codes are still computed. Both residual backends pass the 13-case corpus, and the long queued-stream test remains exact. This option requires cached codebooks and leaves vendor FP32 GEMM and normalization unchanged. [Ablation](results/full_quantizer.json), [Triton fidelity](results/full_fidelity_quantizer.json), [CuTe fidelity](results/full_fidelity_quantizer_cute.json), [streaming queue](results/full_request_batching_quantizer.json).

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
        tail_codes, tail_lengths = session.push(
            chunk, valid_lengths=[960, 1920], final_lanes=[True, False])
        # Lane 0 is now finished; lane 1 can continue in the same graph.
        session.reset(torch.tensor([True, False], device="cuda"))
        next_codes, next_lengths = session.push(chunk)
```

`attention_mask_backend="triton"` constructs the same FP32 zero/negative-infinity attention bias with aligned row strides and shares it within synchronized stages. This removes repeated mask construction, conversion, and padding while preserving the attention calculation.

Quantizer fusion is opt-in through `optimized(..., quantizer_backend="triton")`. Direct quantizer calls retain their quantized vectors, codes, and lengths. The pinned encoder omits accumulated vectors and the final projection it does not return; installed forward hooks retain the complete path. The fused selector preserves both distance-rounding steps and first-index tie behavior, including the near-tie speech regression. Unsupported latent layouts keep the original straight-through additions. Decoder arithmetic is unchanged.

Shared RoPE tables are opt-in and require `rope_backend="triton"`. They reuse identical positions within each transformer stage, including synchronized session lanes. External upstream streaming falls back to per-layer tables because its offsets may differ.

Use `direction="decode"` with input shape `(32, batch, chunk_frames)`, dtype `torch.int64`. Keep graphs/sessions within the optimization context that created their cached tensors. A session owns model state and its CUDA stream; overlapping sessions on one model are rejected. Pass a CUDA boolean `active_mask` to `push` to pause lanes; their KV history/position stays unchanged and their returned length is zero. Omit the mask to advance all lanes. `reset(mask)` reuses selected lanes for new streams; `reset()` resets all lanes. Continuing lanes process the same configured chunk length. For different final tails, pass host sequences `valid_lengths` (samples for encode, code frames for decode) and `final_lanes` (booleans). Zero length pauses a lane; a short positive length must finish it. Padding beyond each length is ignored, including invalid code indices or NaNs. Finished lanes return length zero until reset. An all-zero-length push skips model execution. The first push includes graph warmup and capture; benchmark reports separate setup from steady state.

Partial final encoder frames are explicitly zero-padded and retained. Save the original sample count to trim reconstructed audio: a codec frame always decodes to 1920 samples. This deliberately avoids upstream's floor-length behavior, which can mark a partial final frame invalid. Cache capacity is extended by `chunk_tokens - 1` at each transformer resolution so chunk insertion cannot overwrite required history. The attention context remains unchanged. This corrects upstream behavior after the ring fills; streaming benchmarks explicitly compare graph execution against the corrected eager session. `final=True` finishes the entire session. Per-lane final lengths, empty completion, and lane reuse retain the captured graph. Multi-GPU execution remains unimplemented.

Known CUDA streaming states reset together in one Triton launch. Use `fast_reset=False` on `StreamingSession` to retain the original reset methods for diagnostics. On the full model, resetting selected lanes takes about 0.063 ms versus 6.38 ms previously—roughly 100× for reset alone, not for encoding or decoding. A 71-step, three-lane schedule crosses the cache context and retains exact valid outputs ([evidence](results/full_lane_completion.json)).

## Queued streaming requests

`StreamingBatcher` assigns complete or incrementally supplied requests to stable lanes, emits available chunks, and refills finished lanes from a queue. Ready queued requests enter in FIFO order; requests waiting for more input can be bypassed until they enter a lane. A Triton gather kernel packs the next chunk directly from its input fragments. It uses the same streaming cache, reset, and graph paths described above.

```python
from fast_moss.batching import StreamingBatcher

# Unbatched mono FP32 CUDA audio; requests can have different lengths.
audio_requests = [torch.zeros(1, 24000, device="cuda"),
                  torch.zeros(1, 5777, device="cuda")]
outputs = {}
with torch.inference_mode(), optimized(model, residual_backend="triton",
                                      rope_backend="triton", kv_backend="triton",
                                      share_rope_tables=True, attention_mask_backend="triton"):
    with StreamingBatcher(model, "encode", batch_size=8, chunk_frames=3) as queue:
        for audio in audio_requests:
            request_id = queue.submit(audio)
            outputs[request_id] = []
        while queue.pending:
            for chunk in queue.step():
                outputs[chunk.request_id].append(chunk.data)
                # chunk.offset locates this output; chunk.final marks completion.
codes = {request_id: torch.cat(parts, dim=-1) for request_id, parts in outputs.items()}
```

Decoder requests use shape `(32, frames)` and int64 CUDA data. `submit()` and `append()` own contiguous copies; returned chunks are also owned. New requests and fragments may arrive between `step()` calls. `submit(data)` defaults to a complete input. Open an incremental request with `submit(data, final=False)` or `submit(final=False)`, then append fragments as they arrive:

```python
with optimized(model, kv_backend="triton"), StreamingBatcher(model, "encode") as queue:
    request_id = queue.submit(final=False)
    collected = []
    for fragment in audio_requests[0].split(1400, dim=-1):
        queue.append(request_id, fragment)
        while queue.can_step:
            collected.extend(queue.step())
    queue.append(request_id, final=True)
    while queue.can_step:
        collected.extend(queue.step())
```

A nonfinal request waits for a full configured chunk. `append(id, fragment, final=True)` flushes its final tail; `append(id, final=True)` closes input without adding data. A late final notification after the last full chunk produces an empty final output at the existing output offset. Empty requests also produce one empty final output. Waiting active requests keep their lane and history. `step()` can return `[]` while `pending > 0`; use `can_step` to avoid polling until more input arrives. If every lane belongs to a waiting request, queued work needs an active request to resume, finish, or be cancelled before it can enter.

`max_pending` bounds active and waiting request count (default 128). Optional `max_buffered_bytes` bounds retained input tensor payloads; exceeding either limit raises `BufferError` before modifying request data. `buffered_bytes` counts entire retained fragments, including consumed prefixes until that fragment is fully consumed. It excludes allocator overhead, model state, and caller-owned outputs. Size the limit to accommodate incoming fragments and partially retained inputs. `cancel(request_id)` discards remaining input and releases its lane. Use one host thread and CUDA stream per batcher. Retain each encoder request's original sample count for trimming decoded audio. Network serving is not implemented.

The incremental full-checkpoint gate uses ten requests, including two over ten seconds, with uneven arrivals, pauses, late final notifications, and lane reuse. All **11,072 tokens and 664,320 waveform samples** match independent original eager timelines at batch eight / three frames. Nine requests emit output before their last input arrival. In three paired rounds, splitting each eligible arrival into three fragments changes median encode time **980.42 → 982.02 ms** and decode time **933.58 → 934.93 ms**, approximately **0.16%/0.14% overhead**. These are host-driven logical arrivals without network waits, not measurements of network latency or additional model speedup. [Incremental evidence](results/full_incremental_batching.json).

On a reproducible uneven queue of 16 requests, batch eight / 240 ms chunks, immediate refill uses **44 model steps versus 86** when each group of eight must finish before the next starts. Median queue time falls **1837 → 942 ms encode** and **1670 → 857 ms decode**, about **1.95×** against the same optimized runtime and batch size. All **13,632 tokens and 817,920 waveform samples** match independent original eager timelines at the same batch shape. Two requests cross the ten-second cache context. Timings include request copies, gathering, resets, owned outputs, and concatenation after graph warmup. This is a workload-dependent queue-completion gain, not a new single-request kernel speedup or a 100× model result. [Triton evidence](results/full_request_batching.json), [CuTe evidence](results/full_request_batching_cute.json).

## Fidelity and benchmark scope

`benchmarks.compare` records exact equality, mismatching element counts, maximum error, and RMSE, and exits with failure on any mismatch. It compares equal batches, audio lengths, FP32 arithmetic, and quantizer counts. Wall-clock and CUDA-event samples include graph input copies/output ownership; amortized kernel-only measurements are labeled separately. Synthetic inputs and small structural tests are development gates, not proof of quality across real speech, music, or sound effects.

A separate split-cache FP32 attention prototype improves filled-ring streaming by 1.24–1.26× encode and about 1.29× decode relative to the already optimized runtime. All 30,720 tokens match across three 12.8-second cases, but waveform rounding changes (maximum 8.51e-5). It remains under `benchmarks/` and is not part of the supported runtime or the speedup table above. [Experimental evidence](results/full_attention_experiment.json).

A cuBLASLt pedantic-FP32 prototype selects measured algorithms and transposed FP32 weight layouts. Its new single-copy mode retains the packed storage and restores the original layout on exit. At batch eight / 240 ms, combined encode/decode falls from **30.916 to 26.019 ms (1.19×)** against the already optimized graph, with peak allocated memory around **7.46 GB** instead of the earlier duplicated-weight prototype's 13.25 GB. Batch 128 improves **169.831 → 146.979 ms (1.16×)** with peak allocation **7.63 GB**. All 18 cases in the batch sweep retain exact tokens, hidden states, audio, and restored-model outputs. [Batch evidence](results/full_cublaslt_batching.json).

The eight-lane 12.96-second stream with pauses, completion, and reuse also remains exact in single-copy mode, with peak allocated memory **8.74 GB**. [Streaming evidence](results/full_cublaslt_resident_streaming.json). The broader tuning passes [684 matrix stress checks](results/cublaslt_large_fidelity.json). This version/hardware-specific experiment stays under `benchmarks/`; use `--resident` with `benchmarks.cublaslt_model` or `benchmarks.cublaslt_streaming`. It temporarily changes parameter storage/strides, so existing graphs and external weight aliases must not cross that context. The supported runtime and speedup table above remain unchanged.

A later search screened 12,422 vendor configurations. Its selected alternatives preserve tested outputs but change whole-codec latency by less than 1% against the previous prototype, so they have not replaced the default tuning. [Matched comparison](results/full_cublaslt_config_ablation.json), [fresh component confirmation](results/cublaslt_config_confirmation.json).

Multi-pass tensor-core experiments keep FP32 weight storage but change arithmetic. A TF32x3 QKV probe changes 24 tokens in the near-tie speech case and produces up to 0.152 waveform error, despite small hidden-state errors. It is rejected as an exact replacement and remains outside the runtime. [Full probe](results/full_tensorcore_probe.json), [first changed decision](results/tensorcore_quantizer_tie.json).

Lossless weight-storage experiments preserve every FP32 bit but have not produced a runtime improvement. Block exponent packing reduces stored bytes by roughly 12%, while separate reconstruction plus GEMM is 1.24–2.50× slower under the eviction protocol. Verified CUDA compressible allocations help the zero-matrix control but give no material benefit on the tested checkpoint matrices. Both remain research-only. [Software codec measurements](results/lossless_shapes.json), [hardware allocation comparison](results/compressible_shapes.json).

Outstanding: broader audio corpus coverage, more streaming durations/batch schedules, network serving, multi-GPU execution, matrix-kernel optimization, and a defensible matched-workload 100× result. See [research notes](docs/research.md) and [work log](docs/progress.md).
