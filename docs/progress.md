# Active optimization log

## 2026-09-20, initial implementation turn

The goal remains active. No 100× full-checkpoint result or comprehensive no-quality-loss result exists. This turn is **progress**: it created code, GPU tests, measured kernel/fixture evidence, and confirmed/fixed a streaming-history defect. The initial workspace was empty; no previous goal-turn artifact was available to classify independently.

Hardware: RTX 5070 Ti 16 GB, SM120, driver 615.71.09. The requested checkpoint is pinned at `3cd226ba2947efa357ef453bcad111b6eafba782`, with 1,774,566,400 parameters / 7,098,265,600 bytes according to its index. The second shard is downloaded; the first is still transferring. Preserve the live download; do not restart just because a tool observation expires.

Implementation and evidence:

- Reversible FP32 codebook-normalization and convolution-weight caches. A reduced-architecture profile exposed 66 weight-normalization calls per encode. Caching preserves exact operations/values; no checkpoint mutation.
- Triton and CuTe DSL residual kernels. CuTe uses explicitly rounded CUDA PTX multiply/add. Cancellation, tail masking, and multiple real transformer widths pass bitwise checks.
- CUDA graph wrapper with fixed input validation, stream ownership, warmup, and owned output tensors.
- Batched incremental encoding/decoding with state cleanup, graph reset, retained final partial frames, and sufficient history capacity at all resolutions.
- `results/tests.txt`: 20 passed. Structural tests retain quantizer depth/width but reduce transformer depth/width; they do not validate trained model quality.
- `results/kernels.json`: full samples for residual-only timing. Roughly 1.8–2.6× device speedup with amortized graph timing; eager fusion regresses due to Python launch overhead. Default optimization therefore only caches; fusion is explicit.
- `results/structural_compare.json`: exact-output factorial comparisons for reference/cached/fused, eager/graph, plus corrected eager versus graph streaming. Fixture speedups are not full-model evidence.
- `results/structural_reference.txt` and `structural_optimized.txt`: exploratory traces before frozen-weight caching; the latter records an eager regression, not the current final configuration.

Live processes started this turn (verify their handles before acting):

- Tool exec session **85876**, original download Python PID **58738**, `snapshot_download` of the pinned checkpoint. Poll with `write_stdin`. Writes `upstream/snapshot.txt` on success.
- Tool exec session **95421**, `.venv/bin/python -u -m benchmarks.baseline --seconds .08 --profile --output results/full_baseline_80ms.json`, log `results/full_baseline_80ms.log`. It waits on the same Hugging Face cache lock while the original downloader runs, then loads and profiles the full FP32 checkpoint. Do not start a competing GPU benchmark while it runs.

Next actions:

1. Poll those live handles. On baseline completion, inspect `full_baseline_80ms.json` and encode/decode operator tables. Fix actual load/runtime errors if any; do not count successful downloads as inference validation.
2. Run full `benchmarks.compare` with matched input/batch/precision/all 32 codebooks; reject any exact-output mismatch, and inspect reference/cached/fused plus graph-only ablations. Record graph setup and memory cost.
3. Collect real speech/music/environmental audio fixtures and full-checkpoint corpus fidelity gates, including edge amplitudes, quantizer near-ties, silence, and partial frames. No corpus quality claim is supported yet.
4. Validate long streaming history against offline causal attention across wrap boundaries and multi-frame chunks. Existing corrected-session graph equivalence is necessary but not sufficient. Extend to variable-length independent scheduling and parallel execution beyond lockstep batches.
5. Use the full profile to select the next bottleneck (RoPE/mask/cache fusion, quantizer operations, GEMM). Keep FP32 first; do not silently substitute reduced precision or codebooks. Evaluate achievable hardware bounds before making a 100× claim.

The compatible Transformers source was installed from a clean local checkout at `4ac58b368aff171c8b45a976b0de9ec3b22a4bdd`; manifests now pin that upstream Git commit for reproduction. The local environment's installed package is a built copy, not an editable link to the unrelated checkout.

## 2026-09-20, full-checkpoint validation and streaming optimization

Previous goal turn classification: **progress**, verified against commit `ca6e53c`, GPU test results, and the running download/baseline handles. This turn also made **progress**: the actual checkpoint completed, full-model inference/profiling ran, two new kernel paths were implemented, and pretrained fidelity/performance evidence was produced. The full goal remains active; 100× is not achieved and broad quality/parallelism requirements remain incomplete.

All jobs listed in the initial entry have now **completed successfully**. Download session 85876 and baseline session 95421 returned exit code 0. Subsequent full compare, both corpus fidelity runs, long streaming, independent-lane verification, tests, and optimized graph profiling also completed. There are no known live benchmark/download jobs left from this turn. Revalidate current processes before launching future concurrent GPU work.

New implementation:

- `fast_moss/kv_cache.py`: two-launch Triton KV write/position update, correct packed-QKV strides, 64-bit offsets, execution-mask-aware writes. Active lanes match upstream exact cache contents through wrap. Paused lanes preserve history instead of being overwritten.
- `fast_moss/rope.py`: fused Q/K rotation, with original PyTorch trigonometric tables, cached FP32 frequencies, and disabled FMA contraction. Packed strides and large positions pass exact checks.
- `StreamingSession.push(active_mask=...)` pauses individual lanes and returns zero valid lengths for them; omitted masks resume all lanes. `reset(mask)` reuses selected lanes. Shared mask storage avoids per-layer copies. Masks require the Triton KV backend; short final lengths still apply to the whole batch.

Authoritative results:

- `results/tests.txt`: **38 passed** (kernels, graph ownership/shape validation, wrap/reset/tail handling, independently advancing lanes, and rejection/cleanup of nested optimization contexts).
- `results/full_baseline_80ms.json` and its encode/decode tables: actual 1,774,566,400-parameter checkpoint, FP32, 32 codebooks, batch 1. Upstream eager 46.15 ms encode / 45.17 ms decode. Encoder matrix operations account for 49.1% of profiled device operator time; decoder 55.8%.
- `results/full_compare_240ms.json`: first full-model factorial comparison, all exact. Residual fusion alone had little graph-level benefit; graph launch reduction dominated.
- `results/full_compare_rope_240ms.json`: RoPE-enabled, same full checkpoint and matched 240 ms input. Eager encode **47.872 ms → 10.516 ms**, decode **38.019 ms → 8.731 ms**, approximately **4.55× / 4.35×**. All offline and corrected-streaming comparisons exact. Setup, individual timing samples, and peak memory are recorded.
- `results/kv_cache.json`: isolated cache update speedups approximately **2.7–14.7×**, not whole-model speedups.
- `results/full_fidelity_triton.json` and `full_fidelity_cute.json`: **12 cases each**, all exact codes, encoder hidden states, and reconstructed samples in eager and graph modes. Includes speech/music/environmental assets at available lengths, silence, impulse, alternating full scale, tiny signal. Audio hashes/URLs are in `data/manifest.json`; these are small regression sets, not comprehensive corpus coverage.
- `results/full_streaming_fidelity.json`: two 12.8-second music lanes, 320 ms chunks, crossing the 10-second context. **10,240 codes identical to offline**, **614,400 decoded samples identical to corrected eager streaming**. Streaming versus offline decode is not bitwise equal: maximum **1.6987e-6**, RMSE **4.9938e-8**, equal in both eager and optimized paths. Do not call offline/streaming waveforms identical.
- `results/full_parallel_fidelity.json`: full-checkpoint two-lane pause/resume and per-lane reset matches independent timelines exactly for encoder/decoder.
- `results/full_graph_profile.json` and associated tables: current optimized graph profile, five replays per direction. Main SGEMM kernel groups dominate (about 58% of encode device time before quantizer convolutions, 68% of decode). Remaining work should focus on dense matrix execution and batching, not extrapolate pointwise microbenchmarks.
- `results/weight_storage.json`: sampled FP32 weights generally are not exactly FP16/BF16 representable. Each direction has about 3.545 GB of dense matrix weights. At a theoretical 896 GB/s, one dense weight read costs approximately 3.96 ms; this conditional hardware estimate is not a universal impossibility proof.

Next useful work:

1. Build a shape-specific FP32 matrix benchmark from actual encoder/decoder layers and investigate CUDA/CuTe/Triton alternatives while retaining the exact gates. Profile dominant small-M SGEMM variants; verify numerical reduction differences rather than silently accepting them.
2. Share repeated RoPE tables and causal masks across layers within each transformer stage, with synchronized streaming offset invariants and rollback on any fidelity failure.
3. Measure matched batched offline/streaming throughput across batch sizes on the available single GPU. Separate per-stream latency, aggregate throughput, and batch-one versus batch-N comparisons explicitly.
4. Extend real-audio and long-stream quality coverage, arbitrary per-lane final lengths, request scheduling, and multi-GPU support where test hardware allows it. Retain the full objective; none of the current evidence proves 100× or universal quality preservation.


## 2026-09-20, shared tables and matrix/batch experiments

Previous goal turn classification: **progress**, verified against commit `052cdb4` and its full-checkpoint evidence. This turn also made **progress**: shared RoPE tables improved exact inference, matrix alternatives were measured without being silently promoted, and matched batch throughput was profiled. The goal remains active; 100× and comprehensive quality coverage remain unachieved.

Supported implementation: `optimized(..., rope_backend="triton", share_rope_tables=True)` computes sin/cos once per transformer stage. Offline positions are shared; `StreamingSession` marks its synchronized layer states. External upstream streaming falls back to per-layer tables. Temporary tables are cleaned up on exceptions. Defaults remain unchanged.

Completed evidence:

- `results/tests.txt`: **40 passed**, including three-layer shared-table eager/graph equality, exception cleanup, divergent-offset fallback, and independently advancing lanes.
- `results/full_compare_shared_240ms.json`: all comparisons exact. Matched batch-one upstream eager encode **47.757 → 10.158 ms (4.70×)**; decode **38.168 → 8.391 ms (4.55×)**, FP32/all 32 quantizers. Setup is separate; steady-state includes copies and owned outputs.
- `results/full_fidelity_shared.json` and `full_fidelity_shared_cute.json`: **12 cases each**, exact codes, hidden states, and waveforms for both residual backends with shared tables.
- `results/full_streaming_shared.json`: long streaming still matches corrected eager exactly and offline tokens exactly. The same pre-existing maximum 1.6987e-6 streaming/offline waveform difference remains.
- `results/full_parallel_shared.json`: paused/resumed/reused lanes match independent full-model timelines exactly.
- `results/matrices.json` and `matrices_gemv.json`: actual checkpoint activation/weight shapes; IEEE Triton tiles, packed weights, row padding, and row-reduction GEMV. Warm-cache component wins shrink substantially after cache eviction. Tiled IEEE GEMM generally regresses; changed reduction orders are recorded against both vendor FP32 and FP64 diagnostics.
- `results/full_gemv_experiment.json`: experimental GEMV preserves all 352 codes on 11 short inputs but changes hidden/audio values (maximum waveform difference 5.9232e-7). Combined graph latency improves only 16.077 → 15.807 ms, about 1.7%. Kept exclusively under `benchmarks/`; this is neither proof of perceptual loss nor sufficient evidence for a supported no-quality-loss optimization.
- `results/batching.json`: batches 1, 2, 4, 8, 16, 32, 64, 128; all encode/decode outputs exact against equal-batch upstream eager. Batch 128 delivers 353.5/364.1 audio seconds per second, with 1.06×/1.08× matched speedups and about 7.5 GB peak allocated memory. Do not label real-time throughput as whole-model acceleration.
- `results/full_graph_batch128.json` and operator tables: roughly 80% device time is SIMT SGEMM and 10% efficient attention. Larger batches amortize launches but expose arithmetic throughput limits.

All benchmark/test jobs from this turn completed successfully, including CuTe shared-table fidelity session 12547. No benchmark process is intentionally left running.

Next useful work: optimize dominant matrix shapes on SM120 with documented numerical behavior; share repeated causal masks under the same synchronized-state invariants; extend independent streaming schedules/final lengths and audio corpus coverage. NVIDIA cuBLAS BF16x9 FP32 emulation is currently listed only for SM10.0/10.3, not this SM12.0 GPU; a library upgrade alone does not enable that route. Continue with the full objective and matched baselines.
