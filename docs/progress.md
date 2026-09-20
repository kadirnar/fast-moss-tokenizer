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


## 2026-09-20, fused/shared attention masks and filled-ring profiling

Previous goal turn classification: **progress**, verified by clean worktree commit `f86766d`, shared-table code, and its checkpoint/batch results. This turn also made **progress**: it implemented a new exact mask kernel and stage sharing, measured the full checkpoint, broadened initial-pause coverage, and profiled a filled streaming cache. The full goal remains active; no 100× result or universal no-quality-loss proof exists.

Implementation: `optimized(..., attention_mask_backend="triton")` computes causal/context masks with 64-bit integer positions, writes the same FP32 zero/negative-infinity bias consumed by the tested PyTorch attention path, and aligns rows to eight elements. The mask is reused within one synchronized transformer-stage invocation. Projection and attention arithmetic are unchanged. Stage reuse now lives in `fast_moss/transformer.py`, supports mask sharing independently of shared RoPE, restores temporary state even after exceptions, and falls back to per-layer computation for external streaming with unknown alignment. No weights, codebook counts, or numerical precision were changed.

Completed evidence:

- `results/tests.txt`: **57 passed**. Includes strided/expanded key positions, offsets through 2**40, finite/unbounded context, additive-mask values/alignment, one construction per stage, three-layer eager/graph equality, exception cleanup, divergent-offset fallback, and initially paused lanes.
- `results/full_compare_boolean_masks_240ms.json`: intermediate shared-boolean masks, all exact; graph encode/decode 9.719/7.945 ms. Retained as an ablation, not the final runtime format.
- `results/full_compare_masks_240ms.json`: final aligned additive masks, all exact; upstream eager **48.369 → 9.491 ms encode (5.10×)**, **38.571 → 7.821 ms decode (4.93×)**. Batch one, 240 ms, FP32/all 32 quantizers, copies/owned outputs included, setup separate. Three-chunk streaming graph passes total 38.885/35.486 ms including reset.
- `results/full_fidelity_masks.json` and `full_fidelity_masks_cute.json`: **12 cases per backend**, exact token/hidden/audio comparisons in eager and graph modes.
- `results/full_streaming_masks.json`: two 12.8-second lanes cross the ten-second context, with 10,240 exact offline tokens and 614,400 samples exact against corrected eager streaming. Streaming/offline decoder difference remains unchanged: maximum 1.6987e-6, RMSE 4.9938e-8.
- `results/full_parallel_masks.json`: independently paused/resumed/reused lanes remain exact, including a lane paused before its first chunk and later started. This exercises an entirely masked empty cache.
- `results/batching_masks.json`: matched batches 1, 8, 128, all exact. Batch 128 reaches 355.3/367.1 audio seconds per second, with only 1.06×/1.09× matched-batch acceleration. Peak allocated memory around 7.42 GB. Throughput is not a 355× model speedup.
- `results/full_streaming_profile_masks.json` and encode/decode tables: filled ten-second ring, batch two, 80 ms chunks. Per-push latency 11.665/10.208 ms; attention takes 26.6%/29.7% of device time, while dense matrix groups remain roughly half. `benchmarks.profile_graph --streaming` now explicitly fills the ring before measurement.

Reproduce the main comparison with `benchmarks.compare --seconds .24 --stream-chunks 3 --rope-backend triton --kv-backend triton --share-rope-tables --attention-mask-backend triton`. All full-model fidelity scripts accept the new mask flag. The default remains opt-in. PyTorch source evidence and the FP32 backend scope are recorded in `docs/research.md`.

All jobs in this turn completed successfully, including final tests session 96771. No benchmark/download job is intentionally left running. Next work should investigate FP32 small-query streaming attention on actual stage shapes, alongside matrix execution and its bandwidth/reduction-order limits. Batched request scheduling, arbitrary per-lane final lengths, broader audio coverage, and multi-GPU work remain part of the original objective.


## 2026-09-20, experimental FP32 split-cache attention

Previous goal turn classification: **progress**, verified against clean commit `ed45da9`, the mask implementation, and its full-checkpoint results. This turn is also **progress**: actual streaming attention shapes were captured, a faster hardware kernel was implemented and evaluated on the full checkpoint, a library alternative was investigated, and numerical/graph-layout issues were tested and addressed. The goal remains active. The supported runtime's matched batch-one 240 ms result is still 5.10×/4.93×; no 100× result or comprehensive no-quality-loss proof exists.

New research code stays under `benchmarks/`:

- `experimental_attention.py`: FP32 split-cache attention. Each query/head computes per-key-split maxima, denominators, and weighted-value sums; a second kernel merges them. Fully masked rows yield zero. No low-precision operands or TF32. The reduction order differs from vendor attention, so it is not promoted to `fast_moss`.
- `attention_shapes.py`: captures actual post-wrap Q/K/V/bias tensors for four stage geometries per direction. Compares vendor SDPA, its math backend, split sizes 64/128/256, and optional FlexAttention in IEEE FP32. It records FP64 error diagnostics and separate warm/cache-evicted component timings, plus eager/graph equality. Locally cached inputs (`results/attention_inputs.pt`) are ignored and can be regenerated with `--save-inputs`.
- `attention_model.py`: three full streaming cases, reference versus experimental encode, fixed-code decode, and round-trip output. All 32 quantizers and FP32 weights remain active. Music is continuous; speech and environmental recordings are explicitly repeated four/six times to fill 12.8 seconds per lane. Repetition counts and hashes are recorded.

Authoritative evidence:

- `results/attention_shapes.json`: 40 component comparisons; 64-key splitting gives roughly 3–7× warm attention speedups, with smaller cache-evicted gains. Errors against FP64 are often lower than vendor FP32 but not uniformly. None of the changed arithmetic paths is vendor-bitwise.
- `results/attention_flex.json`: 16 comparisons. FlexAttention with explicit IEEE FP32 is faster than SDPA but slower than the custom 64-key split here; it also changes rounding. Initial compilation issues (scalar indexing and the installed version's quoted precision-option literal) were resolved. The final report completed successfully.
- Every candidate in both final component reports is bitwise equal between its eager output and its graph replay. Targeted tests exposed stride-dependent reduction trees, including bias padding in 256-key blocks. Q/K/V normalization and dynamic bias strides fix these cases; model Q/K/V are already contiguous and incur no extra copies.
- `results/full_attention_experiment.json`: **30,720 tokens exact** across three two-lane 12.8-second streams. Waveforms are not bitwise equal: maximum errors 1.69e-6 music / 2.10e-5 speech / 8.51e-5 environment; signal-to-error ratios 126.4 / 118.2 / 102.0 dB. These are numerical measurements, not a perceptual quality proof.
- Full filled-ring graph encode improves **11.64–11.78 → 9.35–9.37 ms**, or **1.24–1.26×**, versus the already optimized stream. Decode improves **10.12–10.13 → 7.84–7.85 ms**, about **1.29×**. Those mean approximately 20–21% / 22–23% lower latency. Do not multiply these by the different batch-one/offline benchmark or label component wins as whole-model gains.

Reproduce with `.venv/bin/python -m benchmarks.attention_shapes` and `.venv/bin/python -m benchmarks.attention_model`; add `--flex-only` to the former for the alternative library. No supported runtime behavior changed in this turn. Further work should explore shared Q/K/V loads across short query tiles, broaden speech/music/environmental and near-tie quantizer coverage, validate irregular lane schedules with the experimental kernel, and investigate exact reduction matching. Dense matrix execution remains the dominant remaining cost after the experimental attention improvement.


Final verification: `results/tests.txt` records **69 passed**, including 12 new tests across split sizes, noncontiguous inputs, masked tails, empty rows, large scores, and graph replay. Final component/full-model rerun session 92121 and test session 14737 completed successfully. No GPU benchmark or download is intentionally left running.


## 2026-09-20, independent completion and fused lane reset

Previous goal turn classification: **progress**, verified against clean commit `2af03f9` and its full streaming attention experiment. This turn is also **progress**: heterogeneous final lengths and reusable completed lanes now work in the supported exact runtime, a new reset kernel removes significant request-boundary overhead, and full-checkpoint validation crosses the cache context. The goal remains active. Reset alone now exceeds 100× acceleration, but the requested 100× whole-model acceleration is still unachieved.

Supported changes:

- `StreamingSession.push(..., valid_lengths=[...], final_lanes=[...])` takes host metadata, with samples for encode and frames for decode. Zero length pauses a lane. Short positive tails must finish the lane; finished lanes stay paused until selected by `reset(mask)`. Global `final=True` retains session-wide completion. The same graph handles all tail lengths, closure, and reuse.
- `fast_moss/stream_inputs.py`: fused masked loading, zero padding, effective length calculation, and activity updates. Padding never reaches the encoder or code lookup, including NaNs/invalid code IDs. Strided audio/codes and strided/expanded execution masks are tested. No host read of GPU state is required.
- Empty completion returns empty outputs/zero lengths without executing or capturing the model. Completion before first capture survives capture warmup/reset. Returned outputs remain owned.
- `fast_moss/stream_reset.py`: validated GPU address table for per-lane int64 offsets, shared masks, and completion flags. One launch resets selected lanes. CPU bookkeeping matches upstream; unknown layouts/state types fall back. Enabled by default for known CUDA states; `fast_reset=False` preserves the reference path. Session exit releases the reset plan's state references.
- Stream ownership checks now precede preparation/reset, preventing an invalid-stream call from partially changing session state.

Authoritative evidence:

- `results/tests.txt`: **86 passed**. New cases compare fused reset directly with upstream reset methods at batches 1/3/17 and offsets above 2**40, and cover variable tails, empty completion, sticky closure, replay reuse, malformed metadata, wrong-stream rejection, noncontiguous preparation, and invalid padding.
- `results/full_lane_completion.json`: full FP32 checkpoint/all 32 quantizers, three lanes, 160 ms chunks, 71 schedule steps. A continuing lane processes 13 seconds before tail/reuse events, crossing the ten-second context. **8,736 tokens and 522,240 samples match independent eager timelines exactly**; output lengths and graph reuse match at every step. References use original reset methods and identical batch shapes; they are not a sequential speed baseline. Source hashes and repetition behavior are recorded.
- Same-shaped, all-active timing isolates lane-control overhead: encode **13.657 → 13.708 ms**, decode **11.712 → 11.810 ms**, both below 1% overhead in this run.
- Selected-lane reset: encode **6.383 → 0.0629 ms (101.5×)**, decode **6.368 → 0.0623 ms (102.2×)**. This is reset-only acceleration, not codec speedup.
- `results/full_compare_lane_reset_240ms.json`: all comparisons exact. Latest matched batch-one offline result **47.894 → 9.491 ms encode (5.05×)** and **38.317 → 7.820 ms decode (4.90×)**. Offline kernels are unchanged; small baseline differences from the earlier 5.10×/4.93× run are timing variation. Three-chunk graph streaming passes including reset are now **32.421/29.030 ms** encode/decode, versus the prior **38.885/35.486 ms**. The new matched corrected-eager pass timings are 185.870/157.325 ms.

Reproduce heterogeneous completion with `.venv/bin/python -m benchmarks.lane_completion`. Existing comparison/fidelity scripts explicitly disable fast reset in eager references. No experimental attention reduction was enabled, and no weights/precision/quantizer counts changed. Automatic request scheduling, broader corpus/experimental-attention quality checks, matrix execution, and multi-GPU execution remain useful next work toward the original objective.

All benchmark/test jobs completed successfully, including lane validation session 2680, test session 48905, and final comparison session 5981. No GPU benchmark or download is intentionally left running.
