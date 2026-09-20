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


## 2026-09-20, exact FP32 cuBLASLt layout and algorithm research

Previous goal turn classification: **progress**, verified against clean commit `371d242`, the lane completion/reset implementation, and its full-checkpoint results. This turn is also **progress**: a vendor-kernel binding and actual-weight tuning pipeline found useful exact matrix replacements, and full-codec/long-stream checks establish a measured improvement on the tested hardware. The goal remains active. The supported batch-one result is still approximately 5×, and the requested 100× whole-model result is unachieved.

Research changes, all under `benchmarks/`:

- `cublaslt.py`: ctypes binding matching the installed 12.8 headers; own cuBLASLt handle, strict `CUBLAS_COMPUTE_32F_PEDANTIC`, unchanged FP32 weights/inputs, current stream, validated alignment, shared workspace, and explicit descriptor cleanup. Original, row, and transposed-contiguous weight layouts can be measured. It does not change PyTorch's global BLAS math mode.
- `matrices.py`: batch sweeps, cached actual inputs, row filters, all available heuristic candidates, and separate warm/cache-evicted component timing, with FP64 diagnostics and graph fidelity.
- `cublaslt_model.py`: reversible experimental linear replacement. It selects only exact component/graph choices exceeding both timing thresholds and falls back for other shapes/layouts. Every layer retains its original learned weights; selected layers gain an additional FP32 transpose. Shared workspace limits memory overhead.
- A first streaming attempt failed when a fresh heuristic shortlist omitted a measured algorithm; a diagnostic retry succeeded, showing the shortlist is not a stable lookup mechanism here. The final implementation restores the measured descriptor, rejects different GPU/PyTorch/cuBLASLt versions, and validates it through `cublasLtMatmulAlgoCheck`. Final stress/offline/streaming jobs reran after this change.
- `cublaslt_fidelity.py`: twelve input variants per selected actual-weight matrix, including subnormal/tiny/large values, sparsity, and four random inputs.
- `cublaslt_streaming.py`: same-shaped optimized graph reference, eight lanes, 240 ms chunks, 54 schedule steps, plus separately filled-history timing. Includes independently paused lanes, delayed starts, short tails, empty completion, and reset/reuse. Continuous lanes reach 12.96 seconds, crossing the ten-second cache context. Audio source hashes, cyclic filling/offsets, and per-step schedules are recorded.

Authoritative evidence:

- `results/matrices_cublaslt.json`: 276 measurements across 16 shape groups, selected from batches 1/8/128 and one/three codec frames. `results/matrices_cublaslt_codec8.json`: 842 measurements across all 34 cached stage shapes for batch eight / three frames. Some groups overlap. Several selected exact matrix components improve roughly 1.3–1.8×; changed-rounding alternatives are excluded.
- `results/cublaslt_fidelity.json`: all **324 comparisons**, across 27 selected shapes and twelve activation variants, exactly match both PyTorch and their graph replays. This finite gate does not prove universal equality.
- `results/full_cublaslt_experiment.json`: **14 cases**, all exact in eager and graph modes. Music, speech, and environmental audio each use batch/frame pairs (8,3), (4,6), (2,12), (1,24); silence and tiny signals add two cases. Per execution mode, **10,752 tokens and 645,120 waveform samples** match, as do encoder hidden states. Combined batch-eight encode/decode **30.913 → 26.043 ms (1.187×, 15.8% lower latency)** against the already optimized graph. Copies/owned outputs are included; packing/setup is separate. 315 plans add **5,692,227,584 bytes** of packed FP32 weights; peak allocated memory **13,252,793,856 bytes**.
- `results/full_cublaslt_streaming.json`: all valid outputs/lengths and graph reuse exact: **38,080 tokens and 2,282,880 waveform samples**. Filled-history encode **21.361 → 18.908 ms (1.130×)**; decode **19.364 → 16.856 ms (1.149×)**. These are **11.5% / 13.0% lower latency**, not multipliers for the different batch-one benchmark. Each direction packs roughly 2.85 GB separately; peak allocated memory **11,639,704,576 bytes**.
- `results/tests.txt`: **90 passed in 10.69 seconds**. New tests cover ABI/layout interpretation, graph replay, explicit pedantic behavior while PyTorch TF32 is enabled, preservation of global math mode, serialized descriptor validation, shared workspace, and invalid input/alignment/lifetime rejection.

Reproduction (run GPU jobs sequentially):

```bash
.venv/bin/python -m benchmarks.matrices --cublaslt --batches 1 8 128 --frames 1 3 --limit 16 --save-inputs results/matrix_inputs.pt --output results/matrices_cublaslt.json
.venv/bin/python -m benchmarks.matrices --cublaslt --inputs results/matrix_inputs.pt --rows 24 48 96 192 --output results/matrices_cublaslt_codec8.json
.venv/bin/python -m benchmarks.cublaslt_fidelity
.venv/bin/python -m benchmarks.cublaslt_model
.venv/bin/python -m benchmarks.cublaslt_streaming
```

No supported runtime defaults changed, no experimental attention was enabled, and no lower-precision checkpoint or operands were introduced. These improvements are specific to measured shapes and the installed RTX 5070 Ti / PyTorch 2.8 / cuBLASLt 12.8.4 environment. Further work should investigate reducing packed-weight memory overhead, extending exact algorithm coverage to small batches and larger batched workloads, and a validated optional runtime interface. Automatic request scheduling, multi-GPU execution, broader corpus checks, and the original 100× objective remain outstanding.

Final sequential validation session `94535` completed successfully, including final stress, offline, streaming, and full tests. No GPU benchmark or download is intentionally left running.


## 2026-09-20, single-copy FP32 weights and broader matrix coverage

Previous goal turn classification: **progress**, verified against clean commit `9f55340`, its cuBLASLt code, 90-test result, and exact offline/streaming reports. This turn is also **progress**: the experimental matrix path now avoids duplicated GPU weights, covers more useful shapes, passes broader full-model gates, and has a new profile identifying its remaining dominant cost. The goal remains active. No 100× whole-model result has been achieved, and the supported batch-one runtime remains approximately 5×.

Changes:

- `benchmarks/packed_weights.py` shares each FP32 packed tensor across row-count plans. Optional resident mode keeps only that storage, with the original logical weight exposed as its transposed view. It preserves Parameter identity/values, uses no CPU offload or precision conversion, and restores contiguous weights one at a time after releasing plans. Original storage pointers/strides are not retained. Existing graphs/external aliases must not cross this research context; known distinct-Parameter storage aliases are rejected.
- `LinearPlan` accepts validated prepacked weights. The reversible experimental wrapper enforces frozen evaluation, rejects overlapping contexts, preserves autocast/input-gradient fallbacks, and releases its plan/packed caches on exit. Untuned calls on resident weights reconstruct the original contiguous layout before ordinary linear execution.
- `cublaslt_batching.py` evaluates three real sources at batches 1/8/128 and one/three frames, comparing eager and CUDA graph outputs against identical optimized reference workloads. It also checks outputs after restoring original weight storage. Setup and peak allocated memory are recorded separately from steady-state timing.
- `cublaslt_audit.py` compares each candidate linear operation with original-layout PyTorch on the same input, returning the reference to prevent cascading differences. The first batch-128 one-frame attempt exposed one dispatch error at `encoder.7.output_proj`: a contiguous tensor with singleton strides failed PyTorch's leading-stride folding condition. Flattened GEMM matched the component reference but changed native batched-GEMM rounding. The rejected report is preserved in `full_cublaslt_batching_initial.json`; tokens/audio were exact but hidden-state maximum error reached 5.72e-6.
- The wrapper now follows the pinned PyTorch 2.8 leading-stride folding rule and retains native batched GEMM where appropriate, without a layer-name exception. Unit tests reproduce this layout both before packing and after an earlier call has packed the same weight. Matrix reports now explicitly label their flattened 2-D reference and future captures include native input shape/stride metadata.
- `profile_graph --matrix-tuning ...` profiles the resident experiment and persists kernel-group totals in addition to raw operator tables/traces.

Authoritative evidence:

- `results/full_cublaslt_resident.json`: the prior 14-case corpus remains exact in eager/graph execution and after restoration. Combined batch-eight timing **30.867 → 25.988 ms**. Peak allocated GPU memory **7,446,239,232 bytes**, versus the previous duplicated-weight run's **13,252,793,856 bytes**. Logical packed-weight content remains 5.69 GB, but it replaces original storage instead of adding another full copy.
- `results/full_cublaslt_resident_streaming.json`: exact eight-lane, 54-step, 12.96-second schedule, including pauses/tails/empty completion/reuse. **38,080 tokens and 2,282,880 valid waveform samples** match. Encode **21.402 → 18.978 ms**, decode **19.471 → 16.805 ms**. Peak allocation **8,744,888,832 bytes**, versus 11,639,704,576 bytes in the prior copy-based run.
- `results/matrices_cublaslt_large.json`: **1,290 component measurements across 52 shape groups** spanning 128–3,072 matrix rows. Together with the earlier reports, resident selection admits 57 packed shape choices. `results/cublaslt_large_fidelity.json`: **684 exact actual-weight comparisons and graph replays**, including subnormal/tiny/large/sparse/random inputs. Flattened-component equality alone is explicitly insufficient for native dispatch equivalence.
- `results/cublaslt_layer_audit.json` isolates the initial mismatch; `results/cublaslt_layer_audit_fixed.json` is entirely exact after the stride-dispatch correction.
- `results/full_cublaslt_batching.json`: **18 real-audio cases**, all exact in eager/graph execution and after restoring the model, comprising **52,608 tokens and 3,156,480 waveform samples per execution mode**, plus hidden states. At 240 ms per lane, combined encode/decode improves **30.916 → 26.019 ms (1.188×)** at batch eight and **169.831 → 146.979 ms (1.155×)** at batch 128. The latter is a **13.46% latency reduction** against the already optimized graph, with **7.633 GB peak allocation**. At batch 128 / 80 ms, **81.671 → 72.843 ms (1.121×)**. Batch one / 240 ms gains approximately 4.1%; batch one / 80 ms and batch eight / 80 ms have no meaningful gain. Setup for the measured case spans approximately 0.4–1.0 seconds and is excluded from steady-state latency.
- `results/full_cublaslt_large_profile.json` and its encode/decode tables: batch 128 / 240 ms, FP32/all quantizers. SGEMM still accounts for **79.97% encode / 82.51% decode** of CUDA kernel time; attention accounts for **11.80% / 11.89%**. Profiled kernel totals are not substituted for unprofiled end-to-end timing.
- `results/tests.txt`: **96 passed in 16.57 seconds**, including shared packed storage across row plans, graph replay/fallback, Parameter/value restoration, injected plan-validation failure cleanup, alias rejection, and the native batched-GEMM regression.

Reproduce memory and broader coverage with sequential GPU jobs:

```bash
.venv/bin/python -m benchmarks.cublaslt_model --resident --output results/full_cublaslt_resident.json
.venv/bin/python -m benchmarks.cublaslt_streaming --resident --output results/full_cublaslt_resident_streaming.json
.venv/bin/python -m benchmarks.matrices --cublaslt --inputs results/matrix_inputs.pt --rows 128 256 384 512 768 1024 1536 3072 --output results/matrices_cublaslt_large.json
.venv/bin/python -m benchmarks.cublaslt_fidelity --packed-only --tuning results/matrices_cublaslt.json results/matrices_cublaslt_codec8.json results/matrices_cublaslt_large.json --output results/cublaslt_large_fidelity.json
.venv/bin/python -m benchmarks.cublaslt_batching
.venv/bin/python -m benchmarks.profile_graph --batch 128 --seconds .24 --share-rope-tables --attention-mask-backend triton --matrix-tuning results/matrices_cublaslt.json results/matrices_cublaslt_codec8.json results/matrices_cublaslt_large.json --output results/full_cublaslt_large_profile.json
```

The ignored cached matrix inputs can be regenerated using the previous section's capture command. Performance remains hardware/library/shape-specific. These changes stay in the research path, with no supported-runtime default changes and no experimental attention enabled. Next work should target the measured remaining SGEMM cost, investigate exact batched-GEMM acceleration and native-layout tuning, and develop a validated runtime interface for resident weights. Broader quality/corpus validation, request scheduling, multi-GPU work, and the requested 100× whole-model objective remain outstanding.

All GPU jobs completed, including resident/large-sweep session `74770`, diagnostic audit `95360`, and final corrected audit/batching/profile/test session `11333`. The initially rejected batch run `44997` was terminal before follow-up work. No GPU benchmark or download is intentionally left running.


## 2026-09-20, bounded vendor-kernel search and rejected marginal alternatives

Previous goal turn classification: **progress**, verified against clean commit `d040ccd`, its single-copy implementation, 96-test result, and full-model/profile evidence. This turn is also **progress**: new capability-level search and confirmation tools test a much wider hardware configuration space, full-codec validation establishes exactness of its finalists, and matched measurements show that the proposed alternatives do not materially improve the previous prototype. This changes the next action: further repeated tuning of these vendor shapes is a lower priority than a different matrix execution approach. The goal remains active; there is no 100× result and no new supported-runtime speedup in this turn.

New research tooling:

- `cublaslt_search.py`: ABI-backed algorithm/capability enumeration, bounded custom-option sampling, tile/stage/swizzle variation, and split-K/reduction sweeps. Candidates retain pedantic FP32 data/compute and explicit FMA/FP32 capability flags. Descriptor support/workspace checks precede execution. Heuristic seeds and split factors are preserved so a limited manual sweep cannot accidentally discard existing choices.
- `cublaslt_tune.py`: exactness screening on actual checkpoint matrices, warm ranking, and graph/cache-evicted checks for finalists. Reports distinguish every tested configuration from the much smaller finalist set.
- `cublaslt_confirm.py`: fresh baseline/candidate/baseline timing around warm/cold finalists. This detects initial-baseline drift and prevents inflated long-search ratios from becoming performance claims.
- `cublaslt_ablation.py`: previous/new/ new/previous/previous/new whole-codec rounds at three relevant batch/frame geometries, with exact token/hidden/audio gates at every round. It compares against the previous resident matrix prototype, not just the original unoptimized model.

Evidence:

- `results/cublaslt_config_search.json`: **12,422 valid configuration/layout/shape combinations**, **350 exact tuning outputs**, and 108 finalist/baseline records. Eight matrix geometries cover rows 1/3/24/384 and the dominant 1,280↔5,120 FFN dimensions. No runtime failures occurred in the final sweep. Custom ranges above 127 are sampled, so this is not an exhaustive claim. `cublaslt_config_no_split.json` preserves the preliminary unsplit-only experiment; its omissions motivated retaining heuristic split factors.
- The final component rules change three previous choices: one row-one expansion GEMV custom option and two split-K reduction variants. `results/cublaslt_config_fidelity.json` passes **696 exact stress comparisons/graph replays across 58 shapes**.
- `results/full_cublaslt_config_batching.json`: all **18 real-audio cases** remain exact in eager/graph modes and after restoration, including **52,608 tokens and 3,156,480 waveform samples per execution mode**, plus hidden states. These results establish fidelity for the tested inputs, not a universal proof or a new supported default.
- `results/full_cublaslt_config_ablation.json`: previous/new median combined latency **14.795/14.689 ms** at batch one / 80 ms (**1.0072×**); **26.043/26.033 ms** at batch eight / 240 ms (**1.0004×**); **147.561/147.173 ms** at batch 128 / 240 ms (**1.0026×**). Changes remain below 1%. Previous/default tuning is retained; do not attribute the pre-existing 1.19× or 1.16× matrix-prototype gains to this search.
- `results/cublaslt_config_confirmation.json`: nine fresh finalist checks, all exact. The initial row-one expansion baseline was approximately **65.18 μs**, versus **34.12 μs** in fresh bracketing. The apparent 6× warm ratio is therefore not a reliable matched gain; confirmation finds roughly 3.1× warm and only about 1.1× under the eviction protocol for the selected custom option. The selected new configuration still has less than 1% whole-model effect. The physical cause of the initial timing difference was not isolated.
- `results/cublaslt_unknown_capability.json`: six default-layout probes for numerical-flags-zero algorithm ID 76 return status 15 (`NOT_SUPPORTED`); no execution or performance result is claimed for that ID.
- `results/tests.txt`: **97 passed in 12.40 seconds**. The new GPU test exercises capability enumeration, FP32 numerical flags, serialized restoration, closeness to FP64, and exact eager/graph replay for sampled configurations.

Reproduce the configuration experiment after generating the previously documented cached matrix inputs:

```bash
.venv/bin/python -m benchmarks.cublaslt_tune
.venv/bin/python -m benchmarks.cublaslt_confirm
.venv/bin/python -m benchmarks.cublaslt_fidelity --packed-only --tuning results/matrices_cublaslt.json results/matrices_cublaslt_codec8.json results/matrices_cublaslt_large.json results/cublaslt_config_search.json --output results/cublaslt_config_fidelity.json
.venv/bin/python -m benchmarks.cublaslt_batching --tuning results/matrices_cublaslt.json results/matrices_cublaslt_codec8.json results/matrices_cublaslt_large.json results/cublaslt_config_search.json --output results/full_cublaslt_config_batching.json
.venv/bin/python -m benchmarks.cublaslt_ablation
```

All GPU jobs completed, including final capability sweep `19132`, full validation/ablation/tests `15298`, confirmation `72956`, and support probe `76596`. The initial capability enumeration `66540` was deliberately interrupted after discovering an advertised 262,144-value custom range; it was replaced by the documented bounded search, not restarted because of an observation timeout. No benchmark or download is intentionally left running. The main remaining bottleneck is still dense FP32 matrix execution; request scheduling, broader quality validation, multi-GPU support, and the original 100× whole-model objective remain open.


## 2026-09-20, tensor-core correction prototypes and near-tie regression coverage

Previous goal turn classification: **progress**, verified against clean commit `cccd4c7`, its 12,422-configuration search, matched ablations, and 97-test result. This turn is also **progress**: two new hardware kernels were implemented and evaluated, full-model testing exposed a concrete failure of an approximate encoder, its first discrete decision was traced, and the supported fidelity corpus was strengthened. No new runtime acceleration is accepted from this turn. The 100× objective remains active and unachieved.

Changes:

- `benchmarks/tensorcore_mm.py`: Triton TF32x3 and manually decomposed BF16x6 matrix kernels, preserving FP32 learned-weight storage/output while changing component arithmetic. BF16x6 uses a separate correction accumulator and explicit round-to-nearest CUDA addition for high-product block sums. These experiments do not use FP8/FP4 or distillation.
- `tensorcore_shapes.py`: tile/warp/stage exploration, FP64 error diagnostics, eager/graph consistency, register/spill metadata, and separate warm/cache-evicted timing. `tensorcore_select.py` directly compares promising candidates with the existing exact cuBLASLt selection, instead of attributing pre-existing exact-backend gains to the new kernels.
- `tensorcore_model.py`: a limited full-codec probe of the strongest warm QKV candidate, with all other layers on the exact resident path. It records changed tokens, hidden states, round-trip waveforms, fixed-original-code decoding, graph/eager equality, and timing.
- `tensorcore_tie.py`: full-batch distance inspection at the first changed speech token, exposing the narrow quantizer margin rather than assuming small floating-point errors are harmless.
- `benchmarks.fidelity` now includes the failing speech geometry as an exact-backend regression, with checksum, input shape, cyclic offsets, and observed decision coordinates in its metadata.

Evidence:

- `results/tensorcore_shapes.json`: 60 initial comparisons; `tensorcore_compact.json`: 64 compact-tile comparisons; `tensorcore_bf16x6.json`: 48 six-product comparisons. All completed without runtime failure and retain exact eager/graph equality. The candidate arithmetic is not bitwise equal to vendor FP32. Some compact tiles remove register spills, but most larger tested matrices remain slower than vendor SGEMM. These results apply to these implementations/tiles, not all possible tensor-core kernels.
- `results/tensorcore_selection.json`: eight direct candidate-versus-exact comparisons. The strongest plausible short QKV case gives roughly **1.43× warm** but **0.87× cache-evicted** performance against the tuned exact kernel; no candidate wins both timing regimes by 10%.
- `results/full_tensorcore_probe.json`: **14 cases**, **10,752 tokens**, and **645,120 waveform samples** per execution mode. The QKV probe changes **24 tokens** in `data/speech.wav_b8_f3`; other cases keep their codes. Every eager result matches its graph replay. In the failing case, hidden-state maximum error is **8.5831e-6**, but round-trip waveform maximum error reaches **0.1522008**, RMSE **0.0064646**, and signal-to-error ratio **21.82 dB**. With original codes held fixed, the maximum decoder error is only **1.2301e-5**. This candidate cannot be accepted as an exact replacement or represented as proven quality-preserving.
- Matched batch-eight / 240 ms combined latency in that final probe is **26.015 → 25.392 ms**, roughly **1.0245×**, against the previous exact resident prototype. The small timing change is not promoted because fidelity fails; it is not another multiplier for the different supported batch-one benchmark.
- `results/tensorcore_quantizer_tie.json`: the first mismatch is zero-based quantizer **5**, lane **6**, frame **2**. Reference nearest codes **198/1007** differ in squared distance by **1.1920929e-7**. The perturbed latent chooses 1007 instead of 198, followed by further residual-quantizer changes. Full-batch arithmetic is retained during the audit to avoid changing the matrix dispatch itself.
- `results/full_fidelity_near_tie.json` and `full_fidelity_near_tie_cute.json`: **13 cases per supported backend**, including the new eight-lane speech regression. Tokens, hidden states, and audio remain exact in eager and graph modes against original FP32 execution.
- `results/tests.txt`: **104 passed in 15.76 seconds**. New numerical tests cover both component-product modes, odd tiles, widely separated operand scales, incompatible storage, unchanged PyTorch math mode, and graph replay. These unit tests are numerical/structural checks, not codec-quality approval.

Reproduce the experiments sequentially:

```bash
.venv/bin/python -m benchmarks.tensorcore_shapes
.venv/bin/python -m benchmarks.tensorcore_shapes --compact --rows 24 384 3072 --limit 8 --output results/tensorcore_compact.json
.venv/bin/python -m benchmarks.tensorcore_shapes --mode bf16x6 --rows 24 384 3072 --limit 8 --output results/tensorcore_bf16x6.json
.venv/bin/python -m benchmarks.tensorcore_select
.venv/bin/python -m benchmarks.tensorcore_model
.venv/bin/python -m benchmarks.tensorcore_tie
.venv/bin/python -m benchmarks.fidelity --share-rope-tables --attention-mask-backend triton --output results/full_fidelity_near_tie.json
.venv/bin/python -m benchmarks.fidelity --backend cute --share-rope-tables --attention-mask-backend triton --output results/full_fidelity_near_tie_cute.json
```

The row filter lists possible cached shapes; the limit prioritizes dominant weights, so these runs do not claim coverage of every requested row count. Supported runtime arithmetic/defaults remain unchanged. The concrete near-tie case rules out using average latent error as a sufficient acceptance gate. Further work should improve exact matrix execution or provide a justified treatment of discrete quantization, while continuing request scheduling, streaming/parallel execution, and wider quality coverage toward the original goal.

All jobs completed: initial TF32 sweep `22749`, compact sweep `47995`, BF16x6 sweep `22730`, paired selection `66186`, initial full probe/tests `36546`, final fixed-code probe `78086`, decision audit `81660`, and both expanded supported-backend fidelity runs `11938`. A progress update based only on the tail of the probe log initially overstated token equality; the full report exposed the speech failure and the update was promptly corrected. The authoritative results and conclusions above include that failure. No GPU benchmark or download is intentionally left running.


## 2026-09-20, automatic streaming request refill and fused gather

Previous goal turn classification: **progress**, verified against clean commit `f6a17c3`, the rejected tensor-core probe, its first-decision audit, the expanded exact fidelity corpus, and 104 passing tests. This turn is also **progress**: automatic request scheduling now connects the existing lane controls into a usable queue, a new Triton gather packs its inputs, and matched full-model evidence establishes a substantial queue-completion improvement while preserving tested outputs. The 100× whole-model objective remains active and unmet.

Implementation:

- `fast_moss.batching.StreamingBatcher` accepts finite unbatched CUDA audio/code requests and emits owned `BatchChunk` outputs. It assigns FIFO requests to stable lanes, refills finished/cancelled lanes on the next step, resets their state before reuse, handles partial tails and empty requests, and supports submissions between steps. It bounds active plus queued request count, owns contiguous copies of submitted tensors, and releases them on completion/cancellation/exit. A single model/session/stream retains the captured graph throughout.
- A Triton pointer-table gather copies all selected request chunks into the fixed batch in one launch and masks absent inputs. This does not change matrix arithmetic, weights, quantizers, attention reduction, or the supported backend selection.
- `tests/test_batching.py` checks independent original eager timelines in both directions and eager/graph execution; stable lanes; delayed submission; active and queued cancellation; partial tails; empty completion without capture; source/output ownership; capacity and input validation; construction-stream enforcement; and context cleanup. References preserve the batch shape to avoid confounding scheduling with vendor dispatch changes.
- `benchmarks.request_batching` provides full-checkpoint exactness and alternating matched queue timings. `benchmarks.request_gather` isolates the packing component separately. README includes the public API, ownership/length semantics, and numerical/benchmark scope.

Evidence:

- `results/full_request_batching.json`: all exact in three paired rounds per direction. Sixteen heterogeneous real-audio-derived requests contain **34.073 seconds of original samples**; inputs repeat their hashed sources explicitly and two requests exceed ten seconds. The decoder uses their original eager encoded tokens. Each comparison covers **13,632 tokens** or **817,920 waveform samples** against independent original eager timelines at batch eight / three frames. Graph identity remains unchanged across completed queues and different admission policies.
- FIFO refill reduces **86 model steps to 44**, with the same **142 active chunk slots**, improving slot occupancy **20.64% → 40.34%**. Median queue times are **1837.353 → 941.753 ms encode** (**1.951×**) and **1669.982 → 857.258 ms decode** (**1.948×**). The baseline uses the same scheduler/runtime with groups admitted only after the previous group finishes. Timings include input copies, metadata/gather, resets, output copies, and concatenation; model loading, graph capture, and reference construction are excluded from both sides. This is a workload-dependent scheduling gain, not an additional batch-one model multiplier.
- `results/full_request_batching_cute.json`: a separate full-checkpoint CuTe residual run is exact, with **1836.164 → 941.267 ms encode** and **1671.508 → 855.359 ms decode**. The new gather remains Triton in both runs.
- `results/request_gather.json`: all six batch/direction component cases are exact against per-lane PyTorch copies. At batch eight, graph packing is **5.749 → 0.853 μs audio**, **5.114 → 0.859 μs codes**. Larger component ratios reach approximately 29×/25× at batch 128. These resident-metadata microbenchmarks exclude metadata upload and are not model gains.
- `results/tests.txt`: **110 passed in 18.30 seconds**. `git diff --check` is clean.

Reproduction (GPU jobs run sequentially):

```bash
.venv/bin/python -m benchmarks.request_batching
.venv/bin/python -m benchmarks.request_gather
.venv/bin/python -m benchmarks.request_batching --backend cute --repeats 1 --output results/full_request_batching_cute.json
.venv/bin/python -m pytest -q
```

All jobs are terminal: initial scheduler tests `15096`, full Triton queue benchmark `91969`, gather benchmark `72922`, full CuTe queue benchmark `59247`, and full suite `83983`. No GPU benchmark or download is intentionally left running. The scheduler accepts complete input tensors; incremental input fragment queues/network serving and multi-GPU execution remain open. Dense FP32 arithmetic is still the principal model bottleneck, and the supported single-request speed remains approximately 5×. Further work must preserve these fidelity gates while addressing those larger remaining costs.


## 2026-09-20, exact quantizer fusion and encoder-only unused-output removal

Previous goal turn classification: **progress**, verified against clean commit `1a76a2b`, the automatic scheduler and gather implementation, exact full-queue reports, and 110-test result. This turn is also **progress**: new Triton kernels remove repeated quantizer work, the encoder avoids results it never consumes, and matched ablations demonstrate an additional exact encoder improvement. The 100× whole-model goal remains active and unmet.

Implementation:

- `fast_moss.quantizer` fuses distance postprocessing, first-index selection, embedding gather, and the original straight-through subtraction/addition. Original FP32 normalization and vendor GEMM remain unchanged; both distance roundings and first-NaN behavior are retained. A separate fused update preserves masked residual and quantized-vector arithmetic. Noncanonical latent layouts retain original straight-through operations; noncontiguous residual layouts have an arithmetic fallback.
- `optimized(..., quantizer_backend="triton")` enables the reversible option, requiring cached codebooks and the original LFQ model. Public quantizer calls still return vectors, indices, and lengths. The pinned encoder skips accumulated vectors and projections that its return value never uses, including the final quantizer's unused projected vector. All requested code indices are still calculated. Local/global forward hooks retain the complete observable path; the temporary encoder flag is restored after errors and all methods/attributes restore on context exit.
- `benchmarks.quantizer` measures actual full-checkpoint quantizer inputs at five batch/frame geometries, comparing original eager outputs, both optimized variants, graph replay, and repeated timings. `--select-only` reproduces the initial selector/embedding/STE-only stage. Fidelity, queued streaming, and profiling tools now accept the optional quantizer backend; profile summaries also record kernel counts.

Validation and performance:

- `results/full_fidelity_quantizer.json` and `full_fidelity_quantizer_cute.json`: **13 cases each, all exact**, including codes, encoder hidden states, and decoded audio in eager/graph modes. The near-tie speech regression passes.
- `results/full_quantizer.json`: **five geometries, three alternating paired rounds each**, all quantized vectors, indices, lengths, and encoder codes exact. At batch one / 240 ms, public quantizer latency is **1.530 → 1.078 ms** (1.420×) and encoder latency **9.488 → 9.012 ms** (1.053×, 5.01% lower) against the previous optimized graph. At batch eight / 240 ms, encoder latency is **16.406 → 15.805 ms** (3.66% lower). Gains at batch 128 / 240 ms and batch eight / 3.2 s are approximately 0.76%/0.79%; dense matrix execution limits them.
- `results/full_quantizer_select.json` retains the preliminary one-round stage. Its first batch-one / one-frame baseline is unrepresentative of later measurements; the apparent large first-row encoder ratio is not promoted. Final model claims use the repeated ablation instead.
- `results/full_request_batching_quantizer.json`: the 16-request long queue remains exact for **13,632 tokens and 817,920 decoded samples**, including stable lanes, reuse, partial tails, and two requests beyond the ten-second history. Its FIFO/fixed-wave encoder times are **919.119/1792.348 ms**; decoder times **856.165/1671.590 ms**. Those compare scheduling policies with quantizer fusion enabled on both sides, not a fresh quantizer ablation against the prior turn's queue timing.
- `results/full_graph_quantizer_baseline.json` and `full_graph_quantizer_profile.json`: **2,054 → 1,727 encoder kernel launches**, including 32 `_select` and 31 `_update` calls per replay. Decoder count is unchanged at **1,182**. New encoder/decoder profiles attribute **77.10%/80.04%** of GPU kernel time to SGEMM-named kernels. Dense FP32 matrices remain the principal next target.
- `results/tests.txt`: **131 passed in 19.21 seconds**. The 21 new checks cover rounding ties, infinities/NaNs, output strides, graphs, masked lengths, requested quantizer counts, observed public outputs, local/global hooks, restoration, and exception cleanup. The first test pass exposed a gapped singleton-stride mismatch despite equal values; the corrected fallback is covered by the final suite.

Reproduction, with GPU jobs sequential:

```bash
.venv/bin/python -m benchmarks.fidelity --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --output results/full_fidelity_quantizer.json
.venv/bin/python -m benchmarks.fidelity --backend cute --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --output results/full_fidelity_quantizer_cute.json
.venv/bin/python -m benchmarks.quantizer --select-only --rounds 1 --output results/full_quantizer_select.json
.venv/bin/python -m benchmarks.quantizer
.venv/bin/python -m benchmarks.request_batching --quantizer-backend triton --repeats 1 --output results/full_request_batching_quantizer.json
.venv/bin/python -m benchmarks.profile_graph --share-rope-tables --attention-mask-backend triton --output results/full_graph_quantizer_baseline.json
.venv/bin/python -m benchmarks.profile_graph --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --output results/full_graph_quantizer_profile.json
.venv/bin/python -m pytest -q
```

All GPU jobs are terminal: initial selector tests `44819` (one stride failure), corrected selector tests `48215`, initial corpus `66827`, initial component/model ablation `30211`, extended quantizer tests `60710`, full corpus/ablation/stream/profile/test sequence `20073`, and paired baseline profile/final suite `73490`. No benchmark or download is intentionally left running. The new option is exact on the recorded gates and remains opt-in. Wider coverage, incremental request input, multi-GPU execution, and substantially faster dense FP32 arithmetic remain open; this is not a 100× whole-model result.


## 2026-09-20, lossless weight-storage kernels and verified hardware compression

Previous goal turn classification: **progress**, verified against clean commit `9efa1e4`, its exact quantizer kernels and full-model ablation/corpus/streaming evidence, and 131 passing tests. This turn is also **progress**: a new lossless integer storage codec and CUDA VMM compression binding were implemented, their fidelity and lifetime behavior tested, and controlled matrix measurements reject both tested routes for acceleration. No new runtime speedup is accepted; the 100× objective remains active and unmet.

New work:

- `benchmarks.lossless_weights`: GPU block-header generation, exponent-delta bit packing, and parallel FP32 reconstruction. All 24 sign/mantissa bits and the full original exponent are recoverable. Blocks can use 24–32 bits per value plus metadata, so arbitrary bit patterns are supported without claiming that every input compresses. Encoding synchronizes once to size the packed buffer; decoding is graph-compatible. Tests compare raw int32 views, including signed zeros, subnormals, infinities, and quiet/signaling NaN payloads.
- `benchmarks.lossless_shapes`: compares original GEMM, copy-then-GEMM, and reconstruction-then-GEMM using actual checkpoint inputs. It records encoded bytes, exponent-width histograms, exact reconstructed bits, eager/graph output equality, warm timing, cache-evicted timing, and a final fresh baseline.
- `benchmarks.compressible_memory`: CUDA VMM allocation/mapping/access via the driver API, requesting generic data compression and verifying the returned allocation property. PyTorch tensors retain the CUDA-array-interface owner, whose final release synchronizes and unmaps/frees the allocation. This external memory is not counted by PyTorch's caching allocator and no allocation-size reduction is claimed. The module also reproduces the read-only capability query.
- `benchmarks.compressible_shapes`: three alternating rounds over ordinary PyTorch, uncompressed VMM, and compressible VMM allocations. It adds zeros and a BF16-roundtrip matrix solely as controls; neither changes model weights or enters the runtime.

Evidence:

- `results/lossless_shapes.json`: **48 variants** over **three sampled FFN/QKV matrices**, each evaluated at rows 1/3/24/384, with block sizes 128/256/512/1024. All reconstructed bits and GEMM outputs are exact. Encoded storage ratios are **1.128–1.138×**, approximately **11.3–12.1% smaller**, including metadata. Cache-evicted GEMM-chain ratios are only **0.400–0.804×** versus the original operation: **1.24–2.50× slower**. The final baselines confirm that a large baseline drift does not explain this regression. This is not a whole-model memory-saving measurement.
- `results/compression_capability.json`: device attribute 107 reports generic compression support, and the driver reports **50,331,648 bytes (48 MiB) L2** on SM120. Generic compute data compression is distinct from nvCOMP's dedicated decompression engine; current NVIDIA DE documentation lists different Blackwell GPU models.
- `results/compressible_shapes.json`: **126 exact graph matrix comparisons**, across twelve actual-weight geometries plus two controls, three allocation kinds, and three rounds. Requested compression flags are confirmed on each allocation. Actual-matrix cache-evicted gains against uncompressed VMM range **0.955–1.008×**, with no material benefit. The zero control improves **2.2×**, confirming a data-dependent benefit; the BF16-roundtrip control has no cold improvement. A 1.05× comparison against PyTorch is also present with uncompressed VMM and therefore is not evidence of compression benefit.
- `results/tests.txt`: **160 passed in 20.09 seconds**. New tests cover all exponent widths, varied block sizes/tails, exact bit reconstruction, CUDA graphs, allocation flag verification, and ownership across tensor/graph lifetimes. The first decoder compile requested incompatible `.cg`/`.evict_first` modifiers and failed; corrected supported loads pass. No result uses that failed assembly.

Reproduction (GPU work sequential):

```bash
.venv/bin/python -m benchmarks.compressible_memory
.venv/bin/python -m benchmarks.lossless_shapes
.venv/bin/python -m benchmarks.compressible_shapes
.venv/bin/python -m pytest -q
```

All jobs completed: initial codec tests `50175` (assembler rejection), corrected tests `82317`, actual-weight software sweep `25044`, VMM ownership/bit tests `96141`, hardware sweep `26221`, and final suite `76815`. No GPU benchmark or download is intentionally left running. The failed software path motivates avoiding separate reconstruction/materialization in future matrix work; the hardware experiment shows that allocation flags alone do not materially accelerate these sampled dense FP32 weights. Both tools remain research-only. Existing exact runtime performance, streaming, and scheduling are unchanged; wider matrix coverage, incremental input, multi-GPU execution, and the full 100× objective remain open.


## 2026-09-20, incremental input queues with fused fragment packing

Previous goal turn classification: **progress**, verified against clean commit `0de2e05`, its lossless codec and hardware allocation tools, exact matrix reports, and 160 passing tests. This turn is also **progress**: the supported scheduler now accepts live input fragments and preserves exact long-stream outputs; a new packing kernel avoids per-lane concatenation. This expands streaming functionality with small measured fragmentation overhead. It does not establish an additional model speedup, and the 100× goal remains active and unmet.

Implementation:

- `StreamingBatcher.submit(data, final=False)` or `submit(final=False)` opens an incremental request. `append(id, data, final=...)` owns incoming fragments; `append(id, final=True)` permits completion without more data. Existing `submit(data)` behavior remains a complete request. Nonfinal input waits for a full configured chunk, while a final tail is padded and trimmed through the existing session. Late final notification produces an empty final chunk at the accumulated output offset.
- Ready queued requests can bypass queued requests waiting for input. Once active, each request keeps its lane/history through pauses. `can_step` exposes whether work or completion can proceed; an entirely waiting batch returns no chunks without running the model. Reused lanes reset before admitting a new occupant. Cancellation and context exit release retained fragments.
- Optional `max_buffered_bytes` checks retained input payload capacity before mutating request data, independently of `max_pending`. Its counter includes consumed prefixes until their fragment is fully released; model state, output ownership, and allocator overhead are excluded. It is not a total GPU-memory bound or an automatic fragment-compaction policy.
- The new segmented Triton gather reads source descriptors and disjoint tile tasks, directly copying fragments and zero padding into a batch in one launch. The complete-input single-segment case keeps the existing fast path. Segment count is a runtime scalar to avoid recompilation per count. README documents incremental producer use, waiting behavior, ownership, limits, and fixed-stream requirements. Research notes link the NVIDIA ready-control and stateful direct-scheduling design precedents.

Evidence:

- `results/full_incremental_batching.json`: ten real-recording-derived requests, two with 129 codec frames, batch eight / three-frame chunks, three alternating paired rounds in both directions. Every round retains exact **11,072 tokens / 664,320 decoded samples** against independent original eager timelines with the same batch shape. Stable lanes, output offsets, final completion, graph identity, and fully released input counters are checked. The 62 logical ticks include 13 wholly idle calls; encode executes 47 model steps and decode 48 because their partial-tail readiness differs.
- Nine of ten requests emit data before their last input arrival. One-fragment versus three-fragment arrivals use the same logical schedule, with no real-time sleeps or network model. Median encode queue times are **980.417 → 982.021 ms**, and decode **933.580 → 934.928 ms**, approximately **0.16% / 0.14% overhead**. Peak retained input payloads are **184,252 bytes encode / 5,376 bytes decode** for this incremental schedule; these are not whole-model memory figures.
- The preliminary job warmed only a short prefix and exposed compilation in the first fragmented timed round. Its timings were superseded: final code uses a runtime segment count and warms both complete schedules before sampling. `results/full_incremental_batching_cute.json` independently passes the same full-checkpoint gate with CuTe residual fusion; its one-round queue times are **977.097 / 979.662 ms encode** and **933.045 / 936.045 ms decode**.
- `results/fragment_gather.json`: six exact component cases, three alternating rounds each, including output allocation and GPU metadata upload. Against per-lane concatenation/copy into a zero batch, batch-eight packing improves **0.166 → 0.117 ms encode** (1.42×) and **0.163 → 0.091 ms decode** (1.80×). Batch-128 component ratios reach 2.12× / 5.49×. These are packing wall times, not additional codec multipliers.
- `results/full_request_batching_incremental_regression.json`: the prior complete-input workload remains exact for **13,632 codes and 817,920 samples**, with **44 FIFO / 86 fixed-wave steps**. Current one-round times are **916.064 / 1790.204 ms encode** and **856.508 / 1672.007 ms decode**, retaining approximately 1.95× against the same optimized fixed-wave baseline. No new gain versus a prior commit is claimed from these separate runs.
- `results/tests.txt`: **167 passed in 28.74 seconds**. New coverage includes fragmented raw-bit gathering, offsets and padding, eager/graph uneven arrivals in both directions, idle-state preservation, ready admission, late completion/reuse, appended-source ownership, byte accounting, rejection without closing input, validation, stream enforcement, and cleanup. Existing finite request tests also pass. Compile checks and `git diff --check` pass.

Reproduction (GPU jobs sequential):

```bash
.venv/bin/python -m benchmarks.incremental_batching
.venv/bin/python -m benchmarks.incremental_batching --backend cute --repeats 1 --output results/full_incremental_batching_cute.json
.venv/bin/python -m benchmarks.fragment_gather
.venv/bin/python -m benchmarks.request_batching --quantizer-backend triton --repeats 1 --output results/full_request_batching_incremental_regression.json
.venv/bin/python -m pytest -q
```

All GPU jobs are terminal: preliminary full arrivals `84398`, corrected warmed run `76799`, CuTe arrivals `6055`, packing component `67277`, complete-input regression `53221`, and full suite `89983`. The GPU inventory shows only the pre-existing desktop process. Dense FP32 matrix execution remains the dominant bottleneck and supported single-request acceleration remains approximately 5×. Network serving, broader schedules/corpus coverage, multi-GPU execution, faster exact matrix arithmetic, and the requested 100× result remain open.


## 2026-09-20, supported resident FP32 matrix backend

Previous goal turn classification: **progress**, verified against clean commit `fc56113`, its incremental scheduler/fragment kernels and exact full-checkpoint reports, and 167 passing tests. This turn is also **progress**: the measured matrix prototype has become an opt-in supported backend with packaged tuning, graph/storage invalidation, per-stream workspaces, and matched full-model improvements. The 100× objective remains active and unmet.

Implementation:

- `optimized(..., matrix_backend="cublaslt")` owns a `MatrixRuntime` for the pinned checkpoint and recorded RTX 5070 Ti / 70-SM / PyTorch 2.8.0+cu128 / cuBLASLt 12.8.4 environment. The bundled profile retains 57 validated packed choices over 16 weight geometries. Explicit pedantic FP32 compute, original weights, and all quantizers remain unchanged. Unknown input shapes/layouts, bias, gradients, and autocast retain the original contiguous-weight path, including the singleton-stride dispatch guard.
- The shared ctypes binding now lives in `fast_moss.cublaslt`; research tools import compatibility names. Each used CUDA stream owns its plans and a 32 MiB workspace. Calls require one host thread. Supported and experimental contexts reject overlap. Registered parameter/buffer storage aliases and tied names are rejected before mutation; external aliases remain a documented constraint.
- Weight storage is packed lazily only when a validated shape actually reaches it. Device synchronization protects transitions and restoration. Parameters retain identity and all FP32 values; exit closes plans and restores contiguous storage one weight at a time. Exceptions unwind matrix state and the enclosing optimization transformations.
- A device-wide storage epoch invalidates managed graphs on context entry/exit and each newly packed weight. Graph warmup may perform packing before taking its final snapshot, but capture cannot create unwarmed plans. Stale replay fails before GPU submission. Context changes inside a captured function are rejected. All managed graphs on that device are conservatively affected, including unrelated models; reusable graph collections should warm all required shapes first. Raw CUDA graphs must obey the documented lifetime constraint.
- Fidelity, incremental scheduling, and profiling tools accept the new backend. A dedicated paired full-codec benchmark records original eager references, restored outputs, graph timings, setup, and resource/memory accounting. The built wheel includes the profile and all runtime modules.

Evidence and rejected approach:

- `results/full_matrix_runtime_prepack.json`: eager packing of every eligible weight remains exact but regresses small geometries. Batch-one / three-frame time rises from about 17.0 ms to 43.4 ms because unselected operations need repeated contiguous-weight copies. This trial is explicitly superseded; it is not the accepted runtime behavior.
- `results/full_matrix_runtime.json`: final lazy packing, four batch/frame geometries, three real recording sources each, three alternating paired rounds. All codes, hidden states, audio, and restored-model outputs remain exact against original eager references. Both measured variants include supported quantizer fusion. Combined encode/decode medians are **14.291 / 14.293 ms** at batch one / one frame (no meaningful gain, no selected plans), **16.967 / 16.200 ms** at batch one / three frames (**1.047×**), **30.395 / 25.501 ms** at batch eight / three frames (**1.192×**), and **169.851 / 146.823 ms** at batch 128 / three frames (**1.157×**).
- Peak allocated memory in those matrix rounds stays below **7.8 GB**, including per-stream workspaces and restoration. Timings measure warmed graphs with input copies and owned outputs. Packing, plan setup, capture, restoration, and model loading are excluded; separate setup records include the first source's eager/capture checks. The new backend is opt-in and does not promise gains for mixed-shape sequences that trigger fallback copies.
- `results/full_fidelity_matrix_runtime.json` and `full_fidelity_matrix_runtime_cute.json`: **13 cases each, all exact**, including the near-tie speech case, with quantizer fusion and both supported residual backends.
- `results/full_incremental_matrix_runtime.json`: ten uneven requests, including two beyond ten seconds, preserve **11,072 tokens and 664,320 samples** against independent original eager timelines at batch eight / three frames. Graph identity, stable lanes, late final notifications, partial tails, and reuse remain correct. One-fragment/three-fragment times are **862.576/864.235 ms encode** and **812.515/817.501 ms decode**. These are logical-arrival streaming checks, not a newly paired matrix ablation or network latency measurements.
- `results/full_matrix_runtime_profile.json`: batch-eight SGEMM-named kernels still account for **77.78% encode / 79.93% decode** kernel time; attention contributes **6.99% / 8.38%**. This is kernel-event attribution, excluding host time and copies. Dense FP32 arithmetic remains the principal bottleneck despite the accepted gain.
- `results/tests.txt`: **177 passed in 22.46 seconds**. Ten added checks cover lifecycle/restoration, distinct stream workspaces, stale graph rejection, lazy packing after an earlier fallback graph, unchanged storage on new-plan creation, hooks/custom forwards, layout/autocast fallbacks, registered aliases, profile/TF32 rejection, one-thread use, failed-plan cleanup, optimizer unwinding, and capture warmup requirements. Existing research and runtime tests pass. The intentionally empty capture rejection is handled as an expected warning.
- `results/matrix_runtime_package.json`: `uv build --wheel` succeeds and all **19 runtime/profile files** match wheel contents byte-for-byte. The initial `.venv/bin/python -m pip wheel` attempt found no pip module; the available `uv` builder completed the package check. Compile checks and `git diff --check` pass.

Reproduction (GPU jobs sequential):

```bash
.venv/bin/python -m benchmarks.matrix_runtime
.venv/bin/python -m benchmarks.fidelity --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cublaslt --output results/full_fidelity_matrix_runtime.json
.venv/bin/python -m benchmarks.fidelity --backend cute --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cublaslt --output results/full_fidelity_matrix_runtime_cute.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend cublaslt --repeats 1 --output results/full_incremental_matrix_runtime.json
.venv/bin/python -m benchmarks.profile_graph --batch 8 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cublaslt --output results/full_matrix_runtime_profile.json
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-matrix-wheel
```

All jobs are terminal: profile extraction `15492`, initial focused checks `41034`, preliminary corpus `49865`, rejected prepacking ablation `96369`, corrected focused checks `43712`, final ablation `74823`, final Triton/CuTe corpora `27809`/`63879`, incremental gate `27771`, profile `88799`, and final suite `68774`. No GPU benchmark or download is intentionally left running. The historical approximately 5× single-request upstream comparison is not multiplied by a batch-eight ablation. Broader schedules/corpus coverage, network serving, multi-GPU execution, faster exact matrix arithmetic, and the requested matched-workload 100× result remain open.
