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


## 2026-09-20, exact LFQ projection fusion and cached decoder reconstruction

Previous goal turn classification: **progress**, verified against clean commit `2c10a74`, its supported matrix backend, exact full-model/streaming reports, and 177 passing tests. This turn is also **progress**: new Triton kernels remove repeated quantizer projections and decoder reconstruction work, with exact full-model gates and additional matched latency reductions. The 100× whole-model objective remains active and unmet.

Implementation:

- `projection_backend="triton"` enables an eight-channel output projection with the measured forward FMA order and separate bias rounding. With quantizer fusion, the unobserved encoder path combines projection, masking, residual subtraction, and next-step masking. Straight-through latents are retained; decoder lookup values are not substituted into encoder residuals. Input projections remain unchanged.
- The decoder caches all original FP32 projected codebook entries in a **64 MiB** tensor and gathers/adds them in the original codebook order. Its final output projection remains native. Token-oriented gather tiles coalesce table reads for multi-frame input; one-frame input retains the flat layout. Bounds checks preserve asynchronous CUDA assertion behavior, and extra codebooks beyond the model count retain the original ignore behavior.
- The option requires cached convolution weights, the original 32-codebook LFQ geometry, the recorded RTX 5070 Ti / 70-SM / PyTorch 2.8.0+cu128 / cuDNN 9.10.2 environment, and enabled cuDNN with TF32/benchmark disabled. Autocast, observer hooks, and unsupported singleton layouts retain native paths. Cache construction uses functional convolutions and does not emit projection observer calls.
- Guarded cache entry/exit synchronize the device and invalidate managed graphs on it, including unrelated graphs, preventing replay after table cleanup. Raw graphs must remain within the documented context lifetime. Methods/cache attributes restore after normal exit, body exceptions, and partial setup failure. The wheel packages the new runtime module.

Evidence and rejected alternatives:

- `results/pointwise_projection_probe.json`: the first actual output weight is exact with forward FMA at all five geometries. Reverse order, split accumulators, separate multiply/add, and small-shape `linear()` substitutions alter rounding. Component native/fused ratios range **2.13–5.99×**, excluding metadata/model execution. `results/pointwise_input_probe.json`: the 512-channel input projection changes seven of eight outputs at batch one / one frame; the other four tested shapes are exact but run only **0.226–0.412×** as fast as native convolution. This input path is rejected and remains research-only.
- The first projection tests exposed a singleton-time stride mismatch, despite equal values: `(512,1,1)` versus `(512,1,512)`. The accepted implementation falls back to cuDNN for that noncanonical case. Corrected focused tests pass.
- `results/projection_components.json`: **864 exact learned-weight checks**, covering every output projection, five activation geometries, zero/normal/tiny/large inputs and graph replay, plus all entries of all 32 codebooks under different projection layouts. `results/projected_gather.json`: twelve exact mode/geometry checks, with approximately **1.2–9.0×** faster multi-frame component gathers after coalescing reads. This uses a resident random table and excludes validation/allocation/final projection/model work.
- `results/full_projection_runtime.json`: five geometries, three alternating paired rounds, all eager/graph codes, decoded audio, and decoder quantizer outputs exact against original eager references. The baseline already includes supported matrix and quantizer optimizations. At batch one / three frames, encoder time is **8.694 → 8.571 ms**, decoder **7.381 → 7.159 ms**, and decoder quantizer reconstruction **0.253 → 0.059 ms (4.29×)**. Batch-eight encoder/decoder gains are **1.022× / 1.026×**. Across all geometries, reconstruction gains are **3.14–6.16×**; whole-model gains remain approximately 1–3% for smaller shapes and below 1% for larger ones. The preliminary flat-gather ablation is retained separately as `full_projection_runtime_flat.json` and is superseded by the final table-layout measurements.
- Those warmed timings include input copies and owned outputs, excluding loading, cache construction, packing, capture, and restoration. Peak allocated memory stays below **7.83 GB** in the ablation. The explicit table payload is 64 MiB; allocator/model peak differences are not treated as its size.
- `results/full_fidelity_projections.json` and `full_fidelity_projections_cute.json`: **13 cases each, all exact**, including the near-tie speech regression. `results/full_incremental_projections.json`: exact **11,072 tokens and 664,320 samples**, with pauses, partial tails, late completion, stable lanes/reuse, and two requests beyond ten seconds. One-fragment/three-fragment logical schedules take **858.565/862.393 ms encode** and **806.605/807.591 ms decode**. These streaming checks are not an independent paired matrix/projection ablation or network latency claim.
- `results/full_projection_profile.json`: **1856 → 1763 encoder kernels**, **1278 → 1091 decoder kernels** versus the previous matrix profile. The encoder has 31 fused projection/update kernels and no separate quantizer update kernels; the decoder has one projected-table gather. SGEMM-named kernels still consume **77.67% / 80.93%** of encoder/decoder kernel time, excluding host work and copies. Dense FP32 matrices remain the main target.
- `results/tests.txt`: **192 passed in 32.59 seconds**. Fifteen new tests cover arithmetic/layout/bias/graphs, all cached codebook entries, codebook counts including zero and ignored extras, int32 fallback and strided codes, hooks/autocast, model/option validation, exact masked updates, full structural encoder/decoder composition, isolated invalid-index assertions, cache setup/body cleanup, and stale graph rejection. Compile checks and `git diff --check` pass.
- `results/projection_package.json`: `uv build --wheel` succeeds; all **20 runtime/profile files** match packaged contents byte-for-byte.

Reproduction (GPU jobs sequential):

```bash
.venv/bin/python -m benchmarks.pointwise_projection
.venv/bin/python -m benchmarks.pointwise_projection --input-projection
.venv/bin/python -m benchmarks.projection_components
.venv/bin/python -m benchmarks.projected_gather
.venv/bin/python -m benchmarks.projection_runtime
.venv/bin/python -m benchmarks.fidelity --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cublaslt --projection-backend triton --output results/full_fidelity_projections.json
.venv/bin/python -m benchmarks.fidelity --backend cute --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cublaslt --projection-backend triton --output results/full_fidelity_projections_cute.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend cublaslt --projection-backend triton --repeats 1 --output results/full_incremental_projections.json
.venv/bin/python -m benchmarks.profile_graph --batch 8 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cublaslt --projection-backend triton --output results/full_projection_profile.json
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-projections-wheel
```

All jobs are terminal: initial output/input probes `8562`/`64636`, timed probes `33257`/`97206`, initial stride-failing tests `58566`, corrected tests `11474`, preliminary corpus `12728`, fused tests `88274`, intermediate corpus `84525`, flat ablation `44921`, all-weight components `9910`, gather comparison `84679`, final ablation `66393`, final CuTe/Triton corpora `58182`/`45411`, incremental stream `5807`, profile `58541`, and full suite `78156`. No benchmark or download is intentionally left running. The historical approximately 5× whole-model comparison is not multiplied by a component gain. Broader workloads, network serving, multi-GPU execution, faster exact dense arithmetic, and a defensible matched-workload 100× result remain open.


## 2026-09-20, ordered FP32 FFN kernels and a residual-layout fidelity repair

Previous goal turn classification: **progress**, verified at clean commit `83d820a` with its LFQ implementation, saved full-model/profile reports, and 192 passing tests. This turn is **progress**: it adds a supported opt-in Triton path for dominant FFN matrices, obtains new paired performance evidence, and fixes an independently isolated hidden-state fidelity defect. The requested 100× whole-model outcome remains active and unproven.

Work and evidence:

- Added `matrix_backend="triton"`, using consecutive 256-term FP32 FMA partitions and ordered reduction for `(24,5120,1280)` and `(24,1280,5120)`. Other shapes use the prior supported cuBLASLt/native routes. The option inherits model/GPU/library checks, lazy single-copy packing, hooks/fallbacks, one-host-thread restriction, per-stream plans/workspaces, and managed graph epochs. It additionally pins Triton 3.4.0. No reduced-precision operand or weight representation is introduced.
- `ordered_mm.json` records the partition probe, including rejected boundaries and the much slower scalar-load prototype. `ordered_mm_tiled.json` records 24 exact tiled comparisons. The initial invalid sub-minimum dot tile failed compilation and contributes no accepted timing. `ordered_confirm.json` contains 432 exact eager/graph stress checks with 12 inputs per candidate/weight, plus three alternating warm/cold timing rounds. Warm kernel gains of 1.35–1.47× become approximately parity under eviction, so they are not treated as codec speedups.
- `ordered_kernel_resources.json`: both compiled main loops use 166 registers, zero spills, 40,960 bytes of shared memory, FP32 FMA PTX, and no matrix Tensor Core instructions. The resource probe also reconfirms actual-input equality.
- The first full-model run found exact codes/audio but hidden-state differences up to 5.72e-6 at batch 24 / one frame. `ordered_singleton_audit.json` reproduces the same defect without the new matrices and without matrix optimization; disabling residual fusion removes it. `residual_stride_audit.json` isolates equal-valued residual outputs whose singleton strides differ. `empty_like` retained `(1280,1,1)` instead of native `(1280,1280,1)`, changing downstream linear dispatch.
- Both residual backends now allocate canonical contiguous output for the fused contiguous path, matching the pinned TensorIterator setup. Ten new tests check values, strides, downstream matrices and graphs across singleton layouts. The standard real-audio corpus gains batch-24 and batch-128 singleton speech regressions. The initial failure remains in `full_ordered_model_initial.json`; the corrected prototype and integrated runtime have separate reports.
- `full_ordered_runtime.json`: **27 full-checkpoint cases**, all eager/graph/restoration codes, hidden states and audio exact. Three alternating context-by-context rounds show small, variable effects, including a 1.4% decoder regression in one geometry and a 2.6% encoder gain in another. Peak allocated memory is below 7.70 GB. A simplified integrated launch was exact but slightly slower; its reports are retained as `*_specialized.json`, and the final path restores the measured generic tile indexing.
- `full_ordered_paired.json`: **40 interleaved pairs** per direction/geometry within one packed-weight lifetime, all outputs exact against original eager references. At batch eight / three frames, encoder medians are **13.224 → 12.920 ms**, decoder **11.593 → 11.506 ms**. Across three geometries, median ratios are approximately **1.021–1.024× encode** and **1.007–1.008× decode**. The candidate wins only 24–27 of 40 pairs, and distributions overlap. Median paired savings are smaller than differences of independent medians. This report states that two graphs are resident, and includes input copies/owned outputs while excluding loading, packing, capture and restoration. Peak allocated memory is 7.60 GB.
- `full_fidelity_ordered_cute.json`: **15 full-model cases, all exact**, with the integrated matrix and LFQ backends. `full_incremental_ordered.json`: **11,072 tokens and 664,320 samples exact**, with pauses, tails, late completion, lane reuse and two requests longer than ten seconds. One/three-fragment schedules take **850.894/853.428 ms encode** and **796.643/801.382 ms decode**. These stream timings are not an independent backend ablation or network claim.
- `full_ordered_profile.json`: 64 ordered matrix main loops and 64 ordered reductions per direction; total kernels increase by 32 to **1795 encode / 1123 decode**. Vendor SGEMM plus ordered matrices plus vendor split reductions still consume approximately **80.44% / 83.93%** of kernel time. Merely renaming SGEMM work is not counted as removing it. Dense weight traffic and matrix/reduction fusion remain priorities.
- `results/tests.txt`: **206 passed in 32.51 seconds**. Matrix additions cover exact random-weight arithmetic, hooks, fallback layouts after packing, graph replay/lifetime, restoration, compiler gating and helper validation; the existing lifecycle test runs with both backends. Compile and diff checks pass. `ordered_package.json` verifies a built wheel against all **21 runtime/profile files** byte-for-byte.

Reproduction, with GPU jobs sequential:

```bash
.venv/bin/python -m benchmarks.ordered_mm
.venv/bin/python -m benchmarks.ordered_mm --rows 24 --limit 2 --tiled --output results/ordered_mm_tiled.json
.venv/bin/python -m benchmarks.ordered_confirm
.venv/bin/python -m benchmarks.singleton_audit --output results/singleton_audit_current.json
.venv/bin/python -m benchmarks.singleton_audit --residuals --output results/residual_stride_current.json
.venv/bin/python -m benchmarks.ordered_model --supported --output results/full_ordered_runtime.json
.venv/bin/python -m benchmarks.ordered_paired
.venv/bin/python -m benchmarks.fidelity --backend cute --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --output results/full_fidelity_ordered_cute.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend triton --projection-backend triton --repeats 1 --output results/full_incremental_ordered.json
.venv/bin/python -m benchmarks.profile_graph --batch 8 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --output results/full_ordered_profile.json
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-ordered-wheel
```

The two isolation reports preserve pre-fix evidence; rerunning their commands on this commit should observe the corrected behavior. Matrix component tools use the previously captured `results/matrix_inputs.pt`. All observed GPU handles are terminal: partition probe `81389`, rejected tile compilation `36736`, corrected tile sweep `6291`, first stress gate `85992`, initial model failure `9521`, rejected invalid ablation-option combination `23599`, corrected isolation `66924`, residual audit `23361`, residual tests `14247`, corrected prototype `49281`, runtime tests `64539`, specialized runtime `78907`, specialized confirmation `30472`, final confirmation `8808`, final runtime `36562`, sequential CuTe/stream/profile/full-test chain `1564`, interleaved pairs `46525`, and compiled resource check `5646`. No benchmark or download is intentionally left running. The goal remains active: these modest exact improvements do not establish 100×, broader workload guarantees, network serving or multi-GPU execution.


## 2026-09-20, FFN pipeline tuning and exact decoder reduction fusion

Previous goal turn classification: **progress**, verified at clean commit `883cd51`, with the supported ordered matrix path, residual stride repair and 206 passing tests. This turn is **progress**: a new opt-in FFN path passes full-model and stream gates and gives small measured gains. **The 100× whole-model goal remains active and unmet.**

- Added `ffn_backend="triton"`, requiring Triton matrices and either residual backend. Two-stage ordered matrices serve the same two 24-row shapes. Decoder reductions fuse exact GELU/scaled residual arithmetic; encoder retains its existing epilogues after the all-fused comparison showed inconsistent encoder benefit. No reduced precision, retraining, codebook removal or additional persistent weight copy is introduced.
- `ordered_pipeline.json` retains 66 trials: 64 executed exactly, two rejected for excessive shared memory. Aggressive unrolling/spills and the initial cold baseline are not accepted performance claims. Runtime inspection in `ffn_kernel_resources.json` finds **40,960 → 20,480 shared bytes**, unchanged **166 registers**, zero spills, and no matrix Tensor Core instructions. The fused reductions use 18/40 registers and no shared memory.
- The initial fused GELU failed exactness; `ordered_epilogue.json` is explicitly labeled rejected. `gelu_math.json` isolates differences to bundled libdevice on 2,097,152 inputs. The separately validated CUDA 12.8.93 library matches erf/GELU under both tested fusion flags. `gelu_libdevice_source.json` records official wheel provenance and hashes. Added only `nvidia-cuda-nvcc-cu12==12.8.93` to the local environment, optional package `ffn` extra and lockfile. Runtime validates its version/hash before model mutations and passes it only to fused reduction compilation.
- `ffn_components.json`: **192 exact eager/graph comparisons**, including twelve inputs per learned matrix and all four variants; eager bit patterns match throughout. Warm component gains are about 1.095× GELU / 1.140× residual, shrinking under eviction. These are component measurements, not model acceleration factors.
- `full_ffn_runtime.json`: **27 exact full-checkpoint cases**, including codes, hidden states, waveforms, graph replay and context restoration. Forty interleaved pairs at three geometries show variable small gains; batch-eight encoder medians are unchanged, decoder **11.511 → 11.404 ms**. The report records one packed-weight lifetime and two graphs per direction. The final observer-only repair is additionally covered by focused hooks tests and the final CuTe full-model gate.
- `full_ffn_ablation.json`: eight rotating rounds of twenty calls, five graph variants, all original-reference comparisons exact. Previous → selected medians are **13.185 → 13.137 ms encode** and **11.709 → 11.613 ms decode**, about **0.4% / 0.8%** improvements. Distributions overlap. Copies and owned outputs are included; load, packing, capture and restoration are excluded. Initial all-fused reports are explicitly labeled superseded dispatch experiments and retained for audit.
- `full_fidelity_ffn_cute.json`: **15 cases exact**. `full_incremental_ffn.json`: exact **11,072 tokens / 664,320 samples**, pauses, tails, late completion, reuse and two requests beyond ten seconds. One/three-fragment times are **853.404/856.851 ms encode**, **795.444/796.941 ms decode**; these are logical stream fidelity checks, not network or independent speedup claims.
- `full_ffn_profile.json`: **1795 encode / 1059 decode kernels**, 64 fewer decoder launches. Encoder has 64 ordinary ordered reductions; decoder has 64 fused reductions. Matrix-associated groups total **80.74% / 84.77%** of device kernel time; decoder attribution includes fused GELU/residual work. Dense matrix execution and weight traffic remain priorities.
- `results/tests.txt`: **220 passed in 35.16 seconds**. Fourteen new test instances cover exact extreme inputs, special GELU values, both residual backends and encoder/decoder dispatch, hooks including original LayerScale events, custom activations/norms, fallback after packing, graph epochs, restoration and library/option validation. Compile and diff checks pass. `ffn_package.json` verifies the final built wheel's **22 runtime/profile files** byte-for-byte and the pinned optional dependency.

Reproduction, with GPU jobs strictly sequential:

```bash
uv pip install --python .venv/bin/python --no-deps nvidia-cuda-nvcc-cu12==12.8.93
.venv/bin/python -m benchmarks.ordered_pipeline
.venv/bin/python -m benchmarks.ordered_epilogue
.venv/bin/python -m benchmarks.gelu_math
.venv/bin/python -m benchmarks.ffn_components
.venv/bin/python -m benchmarks.ffn_resources
.venv/bin/python -m benchmarks.ffn_model
.venv/bin/python -m benchmarks.ffn_ablation
.venv/bin/python -m benchmarks.fidelity --backend cute --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_fidelity_ffn_cute.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend triton --projection-backend triton --ffn-backend triton --repeats 1 --output results/full_incremental_ffn.json
.venv/bin/python -m benchmarks.profile_graph --batch 8 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_ffn_profile.json
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-ffn-wheel
```

Component tools require the previously captured `results/matrix_inputs.pt`. The initial bundled-library epilogue command intentionally records its failed GELU gate. The pipeline sweep resumes recorded trials; move its output aside for a fresh sweep. All handles are terminal: initial resource-limit failure `66619`, resumed sweep/initial epilogues `17850`, native GELU composition `20048`, isolated wheel download `30034`, math isolation `65106`, install `1183`, components `65343`, initial focused tests `37710`, all-fused corpus `31234`, first ablation `39546`, initial combined validation `20296`, selected corpus `35343`, final sequential ablation/CuTe/stream/profile/math/full-test chain `31304`, and final resource inspection `28916`. No benchmark or download is intentionally left running. Historical approximately 5× whole-model measurements and these component gains are not multiplied together. Broader workloads, exact dense arithmetic improvements, network serving, multi-GPU execution and a defensible matched-workload 100× result remain open.


## 2026-09-20, thirteen more exact ordered attention/FFN shapes

Previous goal turn classification: **progress**, verified at clean commit `bf90313`, with the supported FFN pipeline/decoder epilogue option, saved exactness evidence and 220 passing tests. This turn is **progress**: it expands the hardware kernels beyond those two FFNs and demonstrates corrected whole-model improvements. **The requested 100× whole-model result remains active and unmet.**

- The current profile identified repeated attention projections and 768-channel transformer matrices as remaining dense work. `ordered_shapes.json` probes **18 shapes, 131 accumulation partitions and 65 tiles**. Thirteen shapes match exactly; five remain native because the tested partitions differ. The accepted partitions are 96, 128, 160, 192 and 288 terms, in addition to the prior 256-term FFNs. External primary sources and the arithmetic rationale are recorded in `docs/research.md`.
- `ordered_shapes_confirm.json`: **1,560 exact eager/graph comparisons**, with eager bitwise checks, twelve activation patterns per learned-weight case, two tiles at two pipeline depths, and three alternating warm/cache-evicted rounds. `ordered_shapes_selection.json` records the chosen configurations by evicted median then warm median. Initial cold-start baseline anomalies are not accepted speedup evidence.
- `matrix_backend="triton"` now supports **fifteen attention/FFN shapes** through the existing generic ordered main loop and reduction. The helper validates full shapes and handles partial tails with ceiling division. The previous FFN configurations and explicit stage overrides remain intact. Three new shapes lack vendor algorithms, so the matrix owner records per-stream warmup without a vendor plan; module discovery, cleanup and fallback paths handle them. There is still one persistent copy of every weight, pinned environment/model checks, original FP32 arithmetic, graph invalidation and context restoration.
- `full_ordered_shapes.json`: **27 exact full-checkpoint cases** across original eager codes, hidden states and audio, graph replay, and restored outputs. Its first forty-pair timings are explicitly labeled confounded: sharing newly packed weights forces the old native fallback to make contiguous copies. Those timings are retained for diagnosis and excluded from performance claims.
- `full_ordered_shapes_ablation.json` corrects this by using **independent restored packing lifetimes**. Three rounds alternate backend order; each context times five samples of twenty owned graph calls per direction, with one live graph at a time. All original-reference and restoration checks pass. At batch eight / three frames, encode is **13.167 → 11.520 ms (1.143×)** and decode **11.601 → 9.958 ms (1.165×)**. Batch 24 / one frame gives **1.134× / 1.144×**, batch one / 24 frames **1.142× / 1.167×**, and batch one / three frames **1.009× / 1.009×**. Loading, packing, capture and restoration are excluded; input copies and owned outputs are included. Peak allocation is **7.715 GB**. An additional 283,115,520 bytes of existing weights use packed layout at batch eight; these are not an additional retained copy.
- `full_fidelity_ordered_shapes_cute.json`: **15 exact cases**. `full_incremental_ordered_shapes.json`: exact **11,072 tokens and 664,320 samples** across pauses, tails, late completion, lane reuse and two requests longer than ten seconds. One/three-fragment logical schedules take **780.641/781.489 ms encode** and **716.242/718.831 ms decode**. These are fidelity schedules, not an independent streaming backend ablation or network claim.
- `ordered_shapes_resources.json` verifies all fifteen compiled runtime configurations against actual inputs. Main loops use **65–254 registers, zero spills, and no matrix Tensor Core instructions**. `full_ordered_shapes_profile.json` has **192 ordered main loops per direction**. Encoder has 192 ordinary reductions; decoder has 128 ordinary plus 64 fused reductions. Kernel totals increase by 68 to **1863 encode / 1127 decode**, while matrix-associated work remains **78.58% / 81.88%** of kernel time. The decoder group includes fused GELU/residual work. Renamed vendor work is not treated as removed computation.
- `results/tests.txt`: **249 passed in 44.02 seconds**. Twenty-nine additional test instances extend arithmetic/hooks/fallback/storage/graph checks to all shapes, stress random weights across zero/tiny/subnormal/large inputs and partial tails, and verify capture warmup plus exception cleanup without vendor plans. Focused matrix/FFN tests separately pass **57 cases in 8.83 seconds**. Compile and diff checks pass. `ordered_shapes_package.json` verifies the wheel against all **22 runtime/profile files** byte-for-byte.

Reproduction, with GPU jobs strictly sequential:

```bash
.venv/bin/python -m benchmarks.ordered_shapes
.venv/bin/python -m benchmarks.ordered_shapes_confirm
.venv/bin/python -m benchmarks.ordered_shapes_model
.venv/bin/python -m benchmarks.ordered_shapes_ablation
.venv/bin/python -m benchmarks.fidelity --backend cute --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_fidelity_ordered_shapes_cute.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend triton --projection-backend triton --ffn-backend triton --repeats 1 --output results/full_incremental_ordered_shapes.json
.venv/bin/python -m benchmarks.profile_graph --batch 8 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_ordered_shapes_profile.json
.venv/bin/python -m benchmarks.ordered_shapes_resources
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-ordered-shapes-wheel
```

Component tools use the previously captured `results/matrix_inputs.pt`. The model tool intentionally retains the labeled shared-storage timing diagnostic; accepted performance comes from the separate ablation tool. All GPU handles are terminal: shape/partition search `77052`, stress/confirmation `5915`, full-model/shared-storage diagnostic `80165`, focused tests `55245`, independent-context ablation `2629`, and sequential CuTe/stream/profile/resources/full-suite chain `54771`. No benchmark or download is intentionally left running. The five unmatched native shapes and the now dominant ordered matrix loops remain concrete optimization targets. Historical upstream speedups are not multiplied by this incremental gain; broader workloads, network serving, multi-GPU execution and verified 100× acceleration remain open.


## 2026-09-20, twenty ordered shapes and a fresh whole-codec comparison

Previous goal turn classification: **progress**, verified at clean commit `827e69b` with the 15-shape ordered backend, independent-packing ablation and 249 passing tests. This turn is **progress**: five further exact matrix replacements improve the model, and fresh original-versus-current measurements establish the actual total gain. **The 100× whole-model goal remains active and unmet.**

- The preceding bounded search lacked intermediate partition choices for five shapes without cuBLASLt profile entries. `ordered_remaining.json` records **136 partition probes** and **25 tile trials**. All five have exact candidates. Native grid-Z values 14/8/8/4/5 suggest partitions 96/96/96/192/160 when rounded up to 32; arithmetic probes confirm them. The first one-call profiler traces missed events; final ten-call traces record complete observed main/reduction counts. CUTLASS references and the inference/measurement distinction are documented in `docs/research.md`.
- Added `(24,1280,1280)`, `(48,3072,768)`, `(48,768,768)`, `(192,3072,768)` and `(192,2304,768)` to the existing generic ordered kernel configuration, bringing it to **20 shapes**. The previous fifteen configurations, kernel arithmetic, model weights, precision and dependencies remain unchanged. Existing packing ownership, fallback, stream warmup, absent-vendor-plan and graph lifetime logic handle the additions.
- `ordered_remaining_confirm.json`: **600 exact eager/graph comparisons**, eager bit-pattern checks, twelve inputs per learned-weight case, two tiles at two pipeline depths, and three alternating warm/evicted timing rounds. `ordered_remaining_selection.json` records retained configurations. Warm component ratios range approximately **1.51–2.69×**, evicted ratios **1.33–1.63×**. These are component measurements.
- `full_ordered_remaining.json`: **27 exact full-checkpoint cases**, checking original eager codes, hidden states and audio, replay and restoration. The generalized benchmark has a fidelity-only mode to avoid conflating this gate with the known shared-packing timing diagnostic.
- `full_ordered_remaining_ablation.json`: compares against a recorded **15-shape baseline from `827e69b`**, rejecting changed prior configurations. Three alternating rounds use independent restored contexts, one live graph at a time and five samples of twenty calls per direction. Batch eight / three frames improves **11.530 → 10.448 ms encode (1.104×)** and **9.943 → 8.895 ms decode (1.118×)**. Batch 24 / one frame gives **1.095× / 1.110×**, batch one / 24 frames **1.106× / 1.129×**. Batch one / three frames does not use the new shapes and stays effectively unchanged (**0.9985× / 0.9995×**). Peak allocation is **7.729 GB**. Timings include copies/owned outputs but exclude load, packing, capture and restoration. Another 1,098,907,648 bytes of existing weights use packed layout at batch eight; no second persistent weight copy is retained.
- `full_codec_current.json` refreshes original eager/original graph/current graph measurements directly. Three rotating-order rounds use identical 240 ms speech windows, ten calls per mode/direction and independent restored contexts. All codes/hidden/audio/restoration checks are exact. Batch-one encoder is **47.600 / 11.801 / 8.513 ms**, decoder **37.919 / 10.159 / 7.093 ms**. Total gains versus eager are **5.59× / 5.35×**; versus original graph **1.39× / 1.43×**. Batch-eight encoder is **49.684 / 18.852 / 10.287 ms**, decoder **39.498 / 17.182 / 8.732 ms**: **4.83× / 4.52×** versus eager and **1.83× / 1.97×** versus original graph. Graph input copies and owned outputs are included. Peak allocation is **7.638 GB**. These are fresh matched measurements, not multiplied incremental gains.
- `full_fidelity_ordered_remaining_cute.json`: **15 cases exact**. `full_incremental_ordered_remaining.json`: exact **11,072 tokens and 664,320 samples**, with pauses, tails, late completion, stable lane reuse and two requests beyond ten seconds. One/three-fragment logical schedules take **728.916/729.502 ms encode**, **666.153/667.735 ms decode**. They are fidelity evidence, not an independent streaming/backend ablation or network latency claim.
- `full_ordered_remaining_profile.json`: **272 ordered main loops per direction**. Encoder has 272 ordinary reductions; decoder 208 ordinary plus 64 fused reductions. Two formerly single-kernel native shapes add separate reductions, increasing kernel totals by 24 to **1887 encode / 1151 decode**. Ordered matrix-associated work is **67.27% / 77.00%** of kernel time; including vendor matrix/reduction groups gives **76.30% / 79.20%**. Decoder grouping includes GELU/residual epilogues. The next bottleneck is predominantly ordered kernels and weight traffic; lower SGEMM naming alone is not treated as less computation.
- `ordered_remaining_resources.json` verifies all **20 compiled configurations**, **65–254 registers**, zero spills and no matrix Tensor Core instructions. `results/tests.txt`: **259 passed in 36.56 seconds**. The table-driven arithmetic/hooks/layout/storage/graph and random-weight stress tests add ten test instances for the five new shapes. Compile and diff checks pass. `ordered_remaining_package.json` confirms all **22 runtime/profile files** match the built wheel byte-for-byte.

Reproduction, with GPU work strictly sequential:

```bash
.venv/bin/python -m benchmarks.ordered_remaining
.venv/bin/python -m benchmarks.ordered_shapes_confirm --search results/ordered_remaining.json --output results/ordered_remaining_confirm.json
.venv/bin/python -m benchmarks.ordered_shapes_model --fidelity-only --output results/full_ordered_remaining.json
.venv/bin/python -m benchmarks.ordered_shapes_ablation --previous-configs results/ordered_remaining_baseline.json --output results/full_ordered_remaining_ablation.json
.venv/bin/python -m benchmarks.fidelity --backend cute --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_fidelity_ordered_remaining_cute.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend triton --projection-backend triton --ffn-backend triton --repeats 1 --output results/full_incremental_ordered_remaining.json
.venv/bin/python -m benchmarks.profile_graph --batch 8 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_ordered_remaining_profile.json
.venv/bin/python -m benchmarks.ordered_shapes_resources --output results/ordered_remaining_resources.json
.venv/bin/python -m benchmarks.codec_compare
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-ordered-remaining-wheel
```

Component tools use the previously captured `results/matrix_inputs.pt`. All observed GPU handles are terminal: initial partition/native-launch probe `9130`, complete ten-call launch/tile probe plus stress confirmation `39155`, full-model plus independent-lifetime ablation `65423`, and final sequential CuTe/stream/profile/resources/fresh-comparison/full-test chain `76779`. No benchmark or download is intentionally left running. Exact small-row kernels for short batch-one chunks, more efficient ordered matrix main loops/fusion, broader workloads, network serving and multi-GPU execution remain open. Fresh measured totals still do not establish 100× acceleration.

## 2026-09-20, exact native-layout small-row matrices

Previous goal turn classification: **progress**, verified at clean commit `ba4efdd` with twenty ordered shapes and 259 passing tests. This turn is **progress**: three short-chunk matrix replacements preserve exact output and give a small measured full-model gain. **The 100× whole-model goal remains active and unmet.**

- Native launch profiling records ten calls per sixteen representative shapes. One-row inputs use GEMV; 3/6/12-row inputs use small-N kernels. The first 21 and 336 cyclic/vector/reduction hypotheses fail equality. Local vendor instruction inspection motivates a 256-element boundary: each of sixteen lanes accumulates a tile with ordered FP32 FMA, sums tiles within its lane, and finally participates in a serial lane reduction. `small_tiled_arithmetic.json` tests all 24 captured small-row shapes at three tile sizes. The 256-element version matches all **18** 3/6/12-row cases, including partial K tails; the six one-row GEMVs remain unmatched. Source references and the distinction between inference and numerical evidence are in `docs/research.md`. No vendor assembly is committed.
- The first exact kernel loses all **256** simple tile/warp timing trials against native. Weight-sharing search tests **216** configurations; parallel tile computation tests **108**. All grouped/split search outputs are exact on captured inputs. `small_tiled_confirm.json` adds **480 eager/graph comparisons**, including native controls, twelve activation variants and eager bit-pattern checks. Three alternating warm/evicted rounds retain `(3,5120,1280)` as a grouped kernel and `(3,1280,5120)` / `(12,768,3072)` as split-tile kernels. Warm component gains are approximately **1.18×/1.53×/1.10×**, evicted gains **1.05×/1.15×/1.11×**. Six-row FFN candidates fail the performance gate and remain excluded.
- Added `fast_moss/small_matrices.py`, integrated into the existing opt-in Triton matrix backend. These three shapes preserve native contiguous weight storage, require stream warmup before capture and retain no persistent weight copy/workspace. Later packing by a larger shape invalidates old managed graphs; subsequent small calls use the existing contiguous-weight fallback. Existing hooks, unsupported-layout handling, host-thread ownership, environment checks and context restoration remain enforced. Kernel names distinguish small-row work in traces. No precision, checkpoint or dependency change.
- `full_small_tiled.json` first gates the research wrapper. `full_small_runtime.json` then gates the real runtime: **27 exact checkpoint cases** each, including original eager codes, hidden states, audio, graph replay and restoration. The integrated ablation uses three alternating independent contexts and five samples of ten owned graph calls per direction. Batch one / three frames improves **8.631 → 8.517 ms encode (1.0134×)** and **7.188 → 7.059 ms decode (1.0183×)**. Batch two / three frames gives **1.0048×/1.0049×**, batch one / six frames **1.0073×/1.0028×**. The one-frame control is unchanged. Peak allocation is **7.714 GB**. Copies/owned outputs are timed; load, capture and context setup are excluded. The runtime benchmark disables the new dispatch for its previous-backend control and verifies configurations match the confirmed candidates.
- `full_small_runtime_cute.json`: **27 cases exact**, including the short chunks that exercise new kernels. `full_small_streaming.json`: batch one, three-frame chunks, **12.96 seconds / 54 chunks**, exact **5,184 tokens and 311,040 samples** versus corrected original eager streaming. Both directions exercise the small-row dispatch during warmup/capture. Encoder codes also match offline; original and optimized streaming audio have the same small difference from offline (maximum **1.207e-6**), preserving the documented streaming semantics. The generalized streaming fidelity tool now accepts matrix/quantizer/projection/FFN options.
- `full_incremental_small.json` preserves exact **11,072 tokens and 664,320 samples** through pauses, tails, completion and lane reuse, including two requests beyond ten seconds. This remains a host-driven logical schedule, not network-serving latency evidence.
- `full_small_profile.json`: batch-one / three-frame graphs have **32 grouped small kernels and 44 small tile/reduction pairs per direction**, alongside 48 existing ordered pairs. Small kernels account for **31.01% encode / 38.02% decode** of device kernel time; remaining vendor small matrices account for **29.95% / 34.02%**. All disjoint matrix groups total **77.18% / 81.58%**. The profiler now excludes small-N/GEMV and vendor reduction names from the SGEMM bucket: the type name `cublasGemmSmallNParams` itself contains the substring `sgemm`, which otherwise double-counts these events. The corrected summary was regenerated from the same recorded trace without a second GPU run.
- `small_matrix_resources.json` checks all three runtime configurations: **40–64 registers**, zero spills, no matrix Tensor Core instructions, exact output bits. Split scratch is **4,915,200 / 7,077,888 bytes**; the grouped kernel has no global scratch. Eight focused random-weight/storage/hook/graph/packing tests pass. Full suite: **267 passed in 43.36 seconds**. Compile/diff checks pass. `small_matrix_package.json` verifies all **23 runtime/profile files** against the built wheel byte-for-byte; temporary build output was removed.
- Refreshed `full_codec_current.json` directly compares original eager/original graph/current graph on the same speech input. Batch-one encoder is **47.878 / 11.922 / 8.391 ms**, decoder **38.187 / 10.169 / 6.974 ms**: **5.71× / 5.48×** versus eager and **1.42× / 1.46×** versus original graph. Batch-eight encoder is **49.806 / 19.083 / 10.299 ms**, decoder **39.943 / 17.131 / 8.726 ms**: **4.84× / 4.58×** versus eager, **1.85× / 1.96×** versus original graph. All checks/restorations are exact. These are fresh matched totals, not multiplied incremental gains.

Reproduction, GPU jobs strictly sequential:

```bash
.venv/bin/python -m benchmarks.small_matrix_orders
.venv/bin/python -m benchmarks.small_arithmetic
.venv/bin/python -m benchmarks.small_arithmetic --wide --output results/small_arithmetic_wide.json
.venv/bin/python -m benchmarks.small_tiled_arithmetic
.venv/bin/python -m benchmarks.small_tiled_tune
.venv/bin/python -m benchmarks.small_tiled_grouped
.venv/bin/python -m benchmarks.small_tiled_split
.venv/bin/python -m benchmarks.small_tiled_confirm
.venv/bin/python -m benchmarks.small_tiled_model
.venv/bin/python -m benchmarks.small_tiled_model --runtime --output results/full_small_runtime.json
.venv/bin/python -m benchmarks.small_tiled_model --runtime --fidelity-only --residual-backend cute --output results/full_small_runtime_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 162 --chunk-frames 3 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_small_streaming.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend triton --projection-backend triton --ffn-backend triton --repeats 1 --output results/full_incremental_small.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_small_profile.json
.venv/bin/python -m benchmarks.small_matrix_resources
.venv/bin/python -m benchmarks.codec_compare
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-small-matrices-wheel
```

Component probes reuse captured `results/matrix_inputs.pt`. All GPU handles are terminal: native profiling `32775`, initial compile failure `47188` (unsupported indexing fixed before rerun), narrow/wide probes `92197`/`89161`, tiled arithmetic `16734`, simple tuning `66217`, grouped tuning `89014`, split tuning `51164`, stress/confirmation `27920`, research model `21569`, focused tests `99589`, integrated model `63466`, and sequential CuTe/stream/incremental/profile/resources/fresh-comparison/full-suite chain `83181`. No benchmark or download is intentionally left running. Remaining small-N/GEMV kernels, small-matrix memory traffic and epilogue fusion are concrete next targets. Broader corpora/schedules, network serving, multi-GPU execution and a defensible matched-workload 100× result remain open.

## 2026-09-20, explicit row accumulators and twelve small-matrix shapes

Previous goal turn classification: **progress**, verified at clean commit `3cec6b1` with three native-layout small shapes and 267 passing tests. This turn is **progress**: eleven selected row-accumulator replacements extend native-layout support to twelve shapes and measurably improve the full model. **The 100× objective remains active and unmet.**

- Researched NVIDIA's vectorized-memory-access and matrix-multiplication guidance, then independently tested explicit row reuse and CUDA vector loads while retaining the established FP32 FMA/tile/lane order. `small_fixed_rows.json` contains **198 exact configurations** on four representative shapes. `small_vector_cuda.json` contains **144 exact CUDA configurations**, using installed NVRTC **12.8**, scalar/float2/float4 loads, direct/shared activation paths, block sizes and unrolling. Vectorization competes on one expansion but regresses the long-K projection; no CUDA probe replaces the selected Triton implementation or adds a runtime dependency.
- `small_memory_confirm.json`: **480 eager/graph comparisons**, including bit-pattern checks, twelve activation variants and three warm/evicted timing rounds. A separate **342-configuration cold-only sweep** is retained as a diagnostic in `small_memory_cold.json`; it lacks sustained compute warmup and its absolute times are not mixed with the confirmation protocol. The cold tool now makes this explicit with `--warm-ms`. No unmeasured clock-state explanation is asserted. The first four-shape model experiment passes 27 cases and shows approximately 1.6%/1.8% encode/decode gains at batch one / three frames, supporting further work.
- Expanded to all **18** captured 3/6/12-row shapes and **96 exact configurations** in `small_fixed_shapes.json`. `small_fixed_confirm.json` records **960 eager/graph comparisons**, bit checks, compiled resources and matched timing rounds for sixteen promising shapes. `full_small_fixed.json` compares the sixteen-shape warm selection against an eleven-shape subset with acceptable confirmed evicted behavior, plus the baseline from `3cec6b1`. Both preserve exactness; the eleven-shape set is more consistent across model geometries. The wider set is not enabled merely because it has more component wins.
- `small_fixed_selection.json` records the accepted table: **nine new shapes**, **two existing configurations replaced**, and **one retained split kernel**, for **twelve** native-layout small shapes. Eleven use explicit row accumulators in `small_matrices.py`, mostly three-row tiles; one six-row input benefits from an eight-row tile. Weight vectors are reused across rows without imposing a power-of-two row tensor axis. The existing twenty packed ordered configurations, precision, checkpoint and dependencies are unchanged. Storage ownership, hooks, unsupported-layout fallback, per-stream warmup and graph invalidation continue through the same runtime.
- The real runtime passes **27 original-reference checkpoint cases** in `full_small_fixed_runtime.json`, checking codes, hidden states, audio, graph replay and restoration. Three alternating independent contexts, five samples of ten graph calls, measure batch-one / three-frame **8.505 → 8.239 ms encode (1.0323×)** and **7.063 → 6.794 ms decode (1.0395×)**. Batch two / three frames improves **1.0038×/1.0037×**; batch one / six frames **1.0054×/1.0047×**. The one-frame control is unchanged. Copies/owned outputs are timed; loading/capture/setup are excluded. Peak allocation is **7.714 GB**. Baseline snapshots and the research baseline context preserve old configuration controls when rerunning after integration.
- `full_small_fixed_cute.json`: **27 cases exact**. `full_small_fixed_streaming.json`: all **54 chunks**, **5,184 tokens**, and **311,040 samples** exact versus corrected original eager streaming across **12.96 seconds**, with new dispatch exercised during warmup/capture. Encoder codes also match offline; original and optimized streaming waveforms share the documented maximum **1.207e-6** difference from offline decoding. `full_incremental_small_fixed.json` retains exact **11,072 tokens and 664,320 samples** with pauses, partial completion and lane reuse, including two requests beyond ten seconds. This is not network-serving latency evidence.
- `full_small_fixed_profile.json`: **144 explicit-row kernels and 32 retained small split/reduction pairs per direction**, alongside 48 packed ordered pairs. Twelve fewer kernels per direction give **1666 encode / 1027 decode**. Small kernels consume **51.65% / 61.95%** of kernel time; remaining vendor small matrices **7.13% / 8.94%**. Disjoint matrix groups total roughly **75.92% / 81.30%**. Relabeling vendor work is not treated as acceleration; accepted speedups come from matched independent model timings. The new small kernels and their weight traffic are now the main short-chunk optimization target.
- `small_fixed_resources.json` verifies all **twelve** configurations: **40–128 registers**, zero spills, no matrix Tensor Core instructions, exact actual-input bits. The parameterized storage/hook/graph/packing/random-weight gate grows from eight to **26 focused tests** and passes. Full suite: **285 passed in 47.49 seconds**. Compile/diff checks and baseline-context restoration checks pass. `small_fixed_package.json` verifies all **23 runtime/profile files** against the wheel byte-for-byte; temporary build output was removed. CPU-only CUDA SASS inspection also confirms 128-bit global/shared loads and FP32 FFMA in the vector probe (`small_vector_instructions.json`).
- Refreshed `full_codec_current.json` directly measures original eager/original graph/current graph on the same speech input. Batch-one encoder is **46.738 / 11.918 / 8.120 ms**, decoder **37.283 / 10.168 / 6.710 ms**: **5.76× / 5.56×** versus eager, **1.47× / 1.52×** versus original graph. Batch-eight encoder is **48.836 / 18.848 / 10.288 ms**, decoder **39.112 / 17.239 / 8.723 ms**: **4.75× / 4.48×** versus eager, **1.83× / 1.98×** versus original graph. All checks are exact; peak allocation is **7.638 GB**. Eager baseline variation is retained rather than multiplying incremental ratios or reusing favorable earlier denominators.

Reproduction, GPU work strictly sequential:

```bash
.venv/bin/python -m benchmarks.small_fixed_rows
.venv/bin/python -m benchmarks.small_vector_cuda
.venv/bin/python -m benchmarks.small_memory_confirm
.venv/bin/python -m benchmarks.small_memory_cold --warm-ms 0 --output results/small_memory_cold.json
.venv/bin/python -m benchmarks.small_memory_model
.venv/bin/python -m benchmarks.small_fixed_shapes
.venv/bin/python -m benchmarks.small_fixed_confirm
.venv/bin/python -m benchmarks.small_fixed_model
.venv/bin/python -m benchmarks.small_fixed_model --runtime --output results/full_small_fixed_runtime.json
.venv/bin/python -m benchmarks.small_fixed_model --runtime --fidelity-only --residual-backend cute --output results/full_small_fixed_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 162 --chunk-frames 3 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_small_fixed_streaming.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend triton --projection-backend triton --ffn-backend triton --repeats 1 --output results/full_incremental_small_fixed.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_small_fixed_profile.json
.venv/bin/python -m benchmarks.small_matrix_resources --output results/small_fixed_resources.json
.venv/bin/python -m benchmarks.codec_compare
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-fixed-small-wheel
```

Component probes reuse `results/matrix_inputs.pt`. All GPU handles are terminal: explicit-row search `38095`, CUDA vector search `71050`, first confirmation `24336`, cold-only diagnostic `40439`, four-shape model probe `40999`, eighteen-shape search `12937`, expanded confirmation `84802`, expanded research model `94679`, focused tests `26325`, and integrated-model/CuTe/stream/incremental/profile/resources/fresh-comparison/full-test chain `90773`. No benchmark or download is intentionally left running. One-row GEMV arithmetic, further weight reuse/prefetching in the now dominant small kernels, FFN epilogue fusion, broader corpora/schedules, network serving, multi-GPU execution and verified 100× acceleration remain open.

## Exact one-frame GEMV kernels

Previous goal turn classification: **progress**, verified at clean commit `f846998`, its twelve native-layout small shapes, exact model/stream reports and 285 passing tests. This turn is **progress**: six exact one-row kernels now accelerate the shortest batch-one chunk. **The 100× whole-model objective remains active and unmet.**

- Researched KBLAS and inspected the pinned native GEMV launch geometry and installed SM120 instructions. The inferred arithmetic uses full-K cyclic FMA lanes, an explicit halving reduction and final addition of +0. Lane count is shape-specific: 32, 16 or 8. Generic `tl.sum` fails the 32-lane probe, and ordinary numeric equality hides negative-zero discrepancies after subnormal underflow. No vendor assembly is redistributed.
- Saved 336 initial arithmetic hypotheses, 48 full-width arithmetic checks, 60 signed-zero comparisons and **195 exact tuning configurations**. `gemv_confirm.json` records **672 eager/graph comparisons**, all bit-exact, with fourteen activation variants and three alternating warm/evicted timing rounds. Selected warm component gains are roughly **1.69–3.51×**, evicted gains **1.00–2.09×**. The first 68.8 µs native K=5120 timing did not persist: repeated confirmation gives 35.7 µs, so the initial apparent 6.7× ratio is rejected.
- `small_matrices.py` adds six GEMV shapes, for **eighteen** native-layout configurations. All use the existing opt-in version-gated matrix runtime, stream warmup, hooks, native storage and graph lifetime rules. The twelve previous small and twenty packed ordered configurations remain the ablation baseline. No persistent scratch, weight copy, reduced precision or new dependency is added.
- `exact_gemv_model.py` preserves the older research-only `gemv_model.py` and freezes the preceding configuration table in `gemv_baseline.json`. Both research wrapper and integrated runtime pass **27 full-model cases** against original eager codes, hidden states, audio, graph replay and restored execution. The integrated one-frame comparison gives **7.520 → 7.417 ms encode (1.0139×)** and **6.686 → 6.598 ms decode (1.0134×)**. Unaffected controls vary about −0.5% to +0.3%. Three alternating independent contexts use five samples of ten graph calls; input copies and owned outputs are timed, setup/capture/restoration excluded.
- **44 focused tests** pass, extending random-weight, storage, hook, warmup, fallback and graph checks to all eighteen shapes and adding vector/rank-three inputs plus forced signed-zero underflow. Compile and diff checks pass; `gemv_package.json` verifies all **23 runtime/profile files** against the built wheel byte-for-byte, and temporary build output is removed.
- `full_gemv_cute.json` passes **27 CuTe cases**. `full_gemv_streaming.json` passes all **162 one-frame chunks** across **12.96 seconds**, with exact **5,184 tokens and 311,040 samples** versus corrected original eager streaming. Original and optimized streaming share a maximum **1.349e-6** waveform difference from offline. `full_incremental_gemv.json` retains exact **11,072 tokens / 664,320 samples** through pauses, partial tails, lane reuse and long requests; one/three-fragment schedules take **729.165/731.337 ms encode** and **666.378/668.599 ms decode**, without claiming a new streaming speedup.
- `full_gemv_profile.json` records **129 / 128 GEMV calls** and **1548 / 944 total kernels** per one-frame encode/decode. GEMV takes **44.27% / 50.61%**, vendor small matrices **26.43% / 28.82%**, and disjoint matrix groups together **71.56% / 80.44%** of kernel time. The next targets are these remaining vendor small shapes and GEMV weight traffic. `gemv_resources.json` checks all eighteen configurations bit-for-bit; new GEMVs use **39–80 registers**, **16–640 shared bytes**, zero spills and no matrix Tensor Core instructions.
- The fresh **one-frame** upstream comparison in `full_codec_single_frame.json` measures batch-one encoder **47.008 / 10.344 / 7.346 ms** and decoder **37.312 / 9.107 / 6.523 ms** for original eager / original graph / current optimized graph. Total ratios are **6.40× / 5.72×** versus eager and **1.41× / 1.40×** versus original graphs. Batch-eight encoder is **46.869 / 13.899 / 10.802 ms**, decoder **37.990 / 12.492 / 9.595 ms**, or **4.34× / 3.96×** versus eager and **1.29× / 1.30×** versus original graphs. All outputs and restored executions are exact, with peak allocation **7.517 GB**. The three-frame headline remains the preceding direct measurement of that unchanged dispatch geometry; these results do not multiply historical incremental ratios.
- Full suite: **303 passed in 48.02 seconds**. All GPU jobs completed successfully; compile/diff checks and wheel verification pass.

Reproduction, GPU jobs strictly sequential:

```bash
.venv/bin/python -m benchmarks.small_arithmetic --rows 1 --wide --output results/gemv_arithmetic.json
.venv/bin/python -m benchmarks.small_matrix_orders --rows 1 --all-captured --output results/gemv_native_orders.json
.venv/bin/python -m benchmarks.gemv_order
.venv/bin/python -m benchmarks.gemv_zero
.venv/bin/python -m benchmarks.gemv_tune
.venv/bin/python -m benchmarks.gemv_confirm
.venv/bin/python -m benchmarks.exact_gemv_model
.venv/bin/python -m benchmarks.exact_gemv_model --runtime --output results/full_gemv_runtime.json
.venv/bin/python -m benchmarks.exact_gemv_model --runtime --fidelity-only --residual-backend cute --output results/full_gemv_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_gemv_streaming.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend triton --projection-backend triton --ffn-backend triton --repeats 1 --output results/full_incremental_gemv.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .08 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_gemv_profile.json
.venv/bin/python -m benchmarks.small_matrix_resources --output results/gemv_resources.json
.venv/bin/python -m benchmarks.codec_compare --frames 1 --output results/full_codec_single_frame.json
.venv/bin/python -m pytest tests/test_small_matrices.py -q
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-exact-gemv-wheel
```

All GPU handles are terminal: initial arithmetic `4122`, full-width arithmetic `97988`, signed-zero probe `59682`, native profiling/tuning chain `20430`, component confirmation `31214`, research model `35266`, focused tests `88160`, and runtime/CuTe/stream/incremental/profile/resources/original-comparison/full-tests chain `45275`. No benchmark is intentionally left running. Remaining vendor small matrices, further GEMV weight reuse/prefetching, FFN epilogue fusion, broader corpora/schedules, network serving, multi-GPU execution and verified 100× whole-model acceleration remain open.

## Remaining short-row matrix work

Previous goal turn classification: **progress**, verified at clean commit `e5c974a` with six exact GEMV kernels, full-model/streaming evidence and **303 passing tests**. This turn is **progress**: twenty further exact matrix configurations improve short-chunk codec latency and pass expanded batched and streaming gates. **The 100× whole-model goal remains active and unmet.**

- Native profiling and **78 full-output arithmetic hypotheses** cover all **26** captured two/four/eight-row shapes. Every shape matches the 256-term tiled, sixteen-lane serial-reduction arithmetic bit-for-bit. The search extends existing exact hardware kernels to the remaining one-frame matrix costs, rather than inferring correctness from similar dimensions.
- `short_rows.json` records **558 bit-exact, spill-free configurations**. `short_confirm.json` adds **2,380 eager/graph bit comparisons** on fourteen activation variants across 25 promising shapes, all passing, and three alternating warm/evicted timing rounds. Large initial native cold outliers do not persist and are excluded from retained gain claims. The twenty conservative candidates have confirmed warm ratios **1.077–1.704×** and evicted ratios **0.964–1.545×**.
- `short_model.py` passes **39 original-reference model cases**, adding one-frame batches two/four/eight and two-frame batch-one examples for each real audio source. Independent restored-context comparisons favor the **20-shape conservative set** over the 25-shape warm-only set. Research-wrapper one-frame gains are **1.0394×/1.0497×** encode/decode at batch one, **1.0291×/1.0302×** at batch two, and **1.0460×/1.0540×** at batch eight. Copies and owned outputs are timed; loading, setup and capture are excluded. The three-frame control retains its existing dispatch.
- `short_selection.json` selects **sixteen fixed-row and four split-tile additions**, bringing native-layout support to **38 shapes**. The existing eighteen configurations, twenty packed ordered configurations, FP32 arithmetic, checkpoint and dependencies are unchanged. The supported runtime retains its storage, hook, stream warmup and capture invalidation rules.
- `full_short_runtime.json` passes all **39 cases** through the actual runtime. One-frame encode/decode improves **7.421 → 7.146 ms / 6.603 → 6.322 ms** at batch one (**1.0385× / 1.0445×**), **8.408 → 8.208 ms / 7.175 → 6.970 ms** at batch two (**1.0243× / 1.0294×**), and **10.962 → 10.449 ms / 9.739 → 9.242 ms** at batch eight (**1.0491× / 1.0537×**). Three-frame control ratios are **1.0021× / 0.9990×**. Three alternating independent contexts and five samples of ten calls time copies/owned outputs and exclude setup/capture. Peak allocation is **7.712 GB**.
- **84 focused tests** pass in **37.50 seconds**. Compile/diff checks pass; `short_package.json` verifies all **23 runtime/profile files** in the built wheel byte-for-byte, and temporary build output is removed.
- `full_short_cute.json` passes **39 cases**. `full_short_streaming_b1.json` and `full_short_streaming_b8.json` each pass all **162 one-frame chunks** across **12.96 seconds**. Exact original-streaming totals are **5,184 tokens / 311,040 samples** at batch one and **41,472 tokens / 2,488,320 samples** at batch eight. Encoder codes also match offline; both original and optimized streaming waveforms share maximum offline differences of **1.349e-6 / 2.703e-5**, respectively. The eight-lane gate peaks at **8.791 GB** allocation.
- The fresh one-frame original/current comparison (`full_codec_short_rows.json`) measures batch-one encoder **46.539 / 10.348 / 7.070 ms** and decoder **36.810 / 9.097 / 6.247 ms** for original eager / original graph / current graph. Total speedups are **6.58× / 5.89×** versus eager and **1.46× / 1.46×** versus original graphs. At batch eight the corresponding encoder times are **46.210 / 14.003 / 10.296 ms**, decoder **37.346 / 12.508 / 9.088 ms**: **4.49× / 4.11×** versus eager and **1.36× / 1.38×** versus original graphs. All original-reference and restored-output checks pass. Peak allocation is **7.517 GB**. These totals use freshly measured denominators rather than multiplying earlier incremental gains; the three-frame dispatch is unchanged.
- The one-frame profile now includes a separate per-kernel breakdown within the small-matrix group, explicitly marked as already included in that group to prevent double counting. Batch one runs **129/128 GEMVs, 61 fixed-row kernels and 36 split/reduction pairs**, among **1584 / 980 total kernels** for encode/decode. The 36 additional launches come from split reductions, despite lower measured codec latency. GEMV still accounts for **46.47% / 52.01%** of kernel time. Remaining vendor small matrices fall to **8.28% / 8.12%**; all disjoint matrix groups together consume **70.74% / 79.48%**. At batch eight there are **128 fixed-row kernels per direction**, plus one encoder split pair, and **1581 / 1038 total kernels**. Vendor SGEMM takes **28.49% / 24.98%**, vendor small matrices **10.18% / 10.70%**, and all matrix groups **78.58% / 80.56%**. This points to still-native 16/32/64-row work in one-frame batches, alongside GEMV memory traffic, as subsequent research targets. Profile names moving from vendor to Triton categories are not themselves evidence of speedup.
- `short_resources.json` verifies all **38** runtime configurations against actual checkpoint operands: **36–128 registers**, zero spills and no matrix Tensor Core instructions. The incremental request schedule remains exact for **11,072 tokens and 664,320 samples**, including pauses, tails, lane reuse and long requests (`full_incremental_short.json`). One/three-fragment schedules measure **727.365/729.620 ms encode** and **668.148/668.943 ms decode**; these are regression measurements, not a new paired streaming speedup or network-serving claim.
- Full suite: **343 passed in 56.27 seconds**, including forty newly exercised shape/behavior cases through the parameterized matrix tests. Final report, compiler-resource, subgroup-accounting, wheel and diff audits pass.

Reproduction, GPU jobs strictly sequential:

```bash
.venv/bin/python -m benchmarks.small_matrix_orders --rows 2 4 8 --all-captured --output results/short_native_orders.json
.venv/bin/python -m benchmarks.small_tiled_arithmetic --rows 2 4 8 --output results/short_tiled_arithmetic.json
.venv/bin/python -m benchmarks.short_rows
.venv/bin/python -m benchmarks.short_confirm
.venv/bin/python -m benchmarks.short_model
.venv/bin/python -m benchmarks.short_model --runtime --output results/full_short_runtime.json
.venv/bin/python -m benchmarks.short_model --runtime --fidelity-only --residual-backend cute --output results/full_short_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_short_streaming_b1.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 8 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_short_streaming_b8.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend triton --projection-backend triton --ffn-backend triton --repeats 1 --output results/full_incremental_short.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .08 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_short_profile_b1.json
.venv/bin/python -m benchmarks.profile_graph --batch 8 --seconds .08 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_short_profile_b8.json
.venv/bin/python -m benchmarks.small_matrix_resources --output results/short_resources.json
.venv/bin/python -m benchmarks.codec_compare --frames 1 --output results/full_codec_short_rows.json
.venv/bin/python -m pytest tests/test_small_matrices.py -q
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-short-rows-wheel
```

All GPU handles are terminal: native/arithmetic chain `3359`, 558-configuration sweep `76259`, stress/research-model chain `35078`, focused tests `96510`, and runtime/CuTe/two long streams/incremental/two profiles/resources/upstream comparison/full-tests chain `6623`. No benchmark is intentionally left running. Further native 16/32/64-row replacements, GEMV memory traffic, matrix/FFN fusion, broader fidelity and scheduling corpora, network serving, multi-GPU execution and verified 100× whole-model acceleration remain open.

## Medium-row matrix work and underflow correction

Previous goal turn classification: **progress**, verified at clean commit `31b95eb`, its 38 native-layout shapes, exact model/stream evidence and **343 passing tests**. The 100× whole-model goal remains active and unmet.

- Native launch and full-output arithmetic probes cover all **twenty** 16/32/64-row shapes. The large-K `(32,3072,768)` kernel requires a **512 + 256** partition; `(64,768,240)` needs an **80-term** boundary, where a rounded 96-term guess fails. Six 16-row shapes retain 256-term small-N arithmetic. The initial sweep records **288 exact actual-input configurations**.
- A stronger signed-zero audit finds **42 failing case pairs**, including **23 on previously supported ordered configurations**, despite ordinary numerical equality. Retained failing evidence is in `mid_zero.json` and `mid_confirm_unfixed.json`. Runtime ordered kernels now reproduce the native shape-specific final addition and avoid zero FMAs beyond the two affected preserving tails; the fused FFN residual epilogue also canonicalizes its second projection before scaling.
- The corrected audit passes **288 eager/graph bit comparisons** with actual and forced-positive weights across 72 non-GEMV configurations. New ordered and fused-FFN underflow regressions pass in the **89-test focused suite**. Subsequent table ablations share the correction between both arms and state that baseline scope explicitly.
- Corrected confirmation passes **2,072 eager/graph comparisons** on fourteen activation variants. Twenty candidates have warm component ratios **1.073–2.683×** and evicted ratios **0.650–1.900×**; the latter explicitly includes regressions. A seventeen-shape conservative subset is also compared at model level.
- `full_mid_rows.json` passes **39 model cases** and favors the full twenty-shape selection over that subset in all affected measured geometries. Research-mode encode/decode ratios are **1.1338× / 1.1563×** at batch-eight / one-frame, **1.1338× / 1.1645×** at batch-four / two-frame, and **1.1398× / 1.1682×** at batch-one / eight-frame. Independent packing/restoration contexts prevent shared-storage contamination. Both arms share the signed-zero correction, so these are configuration-table increments rather than an unqualified comparison to the legacy arithmetic implementation.
- `mid_selection.json` selects **six native-layout and fourteen packed additions**, bringing support to **44 small and 34 ordered shapes**. Existing ownership, layout fallback, stream warmup and graph lifetime gates remain active. **227 focused tests pass in 72.08 seconds**; wheel verification matches all **23 runtime/profile files** and temporary build output is removed.
- `full_mid_runtime.json` passes **39 cases** through the default tables, as does `full_mid_cute.json`. Integrated encode/decode improves **10.444 → 9.240 ms / 9.241 → 8.021 ms** at batch-eight / one-frame (**1.1304× / 1.1520×**), **10.370 → 9.142 ms / 8.953 → 7.700 ms** at batch-four / two-frame (**1.1343× / 1.1627×**), and **10.273 → 9.030 ms / 8.769 → 7.503 ms** at batch-one / eight-frame (**1.1377× / 1.1687×**). Control ratios stay within 0.4% of one. Three alternating contexts use five samples of ten graph calls; copies/owned outputs are included, setup/packing/capture excluded. Peak allocation is **7.740 GB**.

Long-stream gates pass chunk-by-chunk equality against corrected original eager streaming: **41,472 tokens / 2,488,320 samples** over 162 one-frame chunks in eight lanes (`full_mid_streaming_b8.json`), and **5,120 tokens / 307,200 samples** over twenty eight-frame chunks in one lane (`full_mid_streaming_f8.json`). Both cross the ten-second cache boundary. Encoder codes also match offline. Original and optimized streaming share the same maximum waveform differences from offline: **2.703e-5** and **1.125e-6**, respectively. Peak allocations are **8.798 GB / 7.597 GB**. The incremental scheduler retains exact **11,072 tokens / 664,320 samples** through arrivals, pauses, tails and lane reuse. One/three-fragment schedules take **721.725/723.854 ms encode** and **660.962/661.612 ms decode** (`full_incremental_mid.json`); these single-repeat regression measurements are not new paired speedup or network-latency claims.

The batch-eight / one-frame profile (`full_mid_profile.json`) runs **1,581 / 1,038 kernels** for encode/decode, including **98 / 97 ordered main/reduction pairs** and **176 fixed-row kernels** per direction. Vendor SGEMM falls to **8.56% / 1.31%** of kernel time and vendor small matrices to **0.71% / 0%**. Ordered kernels take **14.89% / 16.56%**, native-layout small kernels **52.19% / 59.93%**, and all disjoint matrix groups **76.34% / 77.81%**. The nested small-kernel breakdown is already included in its parent group. Scheduling and weight reuse in our small kernels are now the largest target at this geometry; the remaining encoder vendor SGEMM still needs layer attribution. Kernel relabeling alone is not a performance gain.

Resource audits cover all **44 small and 34 ordered configurations**, with actual-input bit equality, zero spills and no matrix Tensor Core instructions. Small-kernel register counts span **36–128**, ordered main kernels **56–254** (`mid_small_resources.json`, `mid_ordered_resources.json`). The latter records the corrected signed-zero/tail flags.

Fresh original/current comparisons use three rotating rounds of ten single-call samples, including graph input copies and owned outputs, excluding load/packing/capture/restoration. In `full_codec_mid_one.json`, one-frame batch-one encode is **46.758 / 10.343 / 7.064 ms**, decode **36.976 / 9.093 / 6.241 ms**, for original eager / original graph / optimized graph: **6.62× / 5.92×** versus eager, **1.46× / 1.46×** versus original graphs. At batch eight, encode is **46.483 / 13.896 / 9.086 ms**, decode **37.642 / 12.300 / 7.870 ms**: **5.12× / 4.78×** versus eager and **1.53× / 1.56×** versus original graphs.

In `full_codec_mid_three.json`, three-frame batch-one encode is **47.867 / 11.811 / 8.115 ms**, decode **38.253 / 10.169 / 6.707 ms**: **5.90× / 5.70×** versus eager and **1.46× / 1.52×** versus original graphs. Batch-eight encode is **49.835 / 19.070 / 10.164 ms**, decode **39.870 / 17.362 / 8.613 ms**: **4.90× / 4.63×** versus eager and **1.88× / 2.02×** versus original graphs. Every comparison and restored execution is exact. These are direct totals with fresh denominators, not products of historical incremental ratios.

The complete suite passes **419 tests in 64.26 seconds**. Wheel verification matches all **23 runtime/profile files**. This turn is **progress**: twenty measured configurations and a signed-zero correctness repair, with exact corpus and streaming gates. The requested **100× whole-model acceleration remains unmet**.

Reproduction (run GPU commands sequentially):

```bash
.venv/bin/python -m benchmarks.small_matrix_orders --rows 16 32 64 --all-captured --output results/mid_native_orders.json
.venv/bin/python -m benchmarks.mid_arithmetic
.venv/bin/python -m benchmarks.mid_rows --legacy
.venv/bin/python -m benchmarks.mid_zero --legacy --output results/mid_zero.json
.venv/bin/python -m benchmarks.mid_confirm --legacy --output results/mid_confirm_unfixed.json
.venv/bin/python -m benchmarks.mid_zero
.venv/bin/python -m benchmarks.mid_confirm
.venv/bin/python -m benchmarks.mid_model
.venv/bin/python -m benchmarks.mid_model --runtime --selection warm --output results/full_mid_runtime.json
.venv/bin/python -m benchmarks.mid_model --runtime --selection warm --fidelity-only --residual-backend cute --output results/full_mid_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 8 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_mid_streaming_b8.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 160 --chunk-frames 8 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_mid_streaming_f8.json
.venv/bin/python -m benchmarks.incremental_batching --matrix-backend triton --projection-backend triton --ffn-backend triton --repeats 1 --output results/full_incremental_mid.json
.venv/bin/python -m benchmarks.profile_graph --batch 8 --seconds .08 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_mid_profile.json
.venv/bin/python -m benchmarks.small_matrix_resources --output results/mid_small_resources.json
.venv/bin/python -m benchmarks.ordered_shapes_resources --output results/mid_ordered_resources.json
.venv/bin/python -m benchmarks.codec_compare --frames 1 --output results/full_codec_mid_one.json
.venv/bin/python -m benchmarks.codec_compare --frames 3 --output results/full_codec_mid_three.json
.venv/bin/python -m pytest -q
uv build --wheel --out-dir /tmp/moss-mid-wheel
```

All GPU handles are terminal: native profiling `82785`, arithmetic `76834`, tuning `74392`, legacy zero/confirmation `3529`, corrected zero `69055`, corrected tests/confirmation/model `77341`, integrated focused tests `29928`, and runtime/CuTe/streams/incremental/profile/resources/upstream comparisons/full-suite chain `37173`. No benchmark is left running. Small-matrix scheduling and weight reuse, remaining vendor attribution, matrix/FFN fusion, broader fidelity and scheduling corpora, network serving, multi-GPU execution and verified 100× acceleration remain open.

Final audit: all new passing reports, retained legacy failure counts, compiler resources, profile subgroup accounting and wheel/source hashes agree with the documentation; Python compilation and `git diff --check` pass. Only the pre-existing desktop GPU process remains.

## Narrow grids and explicit row layouts

Previous goal turn classification: **progress**, verified at clean commit `fd559c3`, its 44 native-layout and 34 ordered configurations, corrected signed-zero behavior and 419 passing tests. The 100× whole-model goal remains active and unmet.

The latest batch-eight one-frame profile identifies native-layout small matrices as the largest kernel-time group (52.19% encode / 59.93% decode). This turn tests one/two output columns per block and more row reuse, preserving the established arithmetic. `narrow_baseline.json` snapshots the current tables before any replacement. `narrow_rows.py` measures the actual runtime kernels with alternative launch configurations against those tables. A separate research-only Gluon kernel specifies the lane/column/row layout explicitly and extends register-resident row reuse to sixteen rows.

- The narrow-grid sweep completes **564 bit-exact, spill-free configurations**. Repeated confirmation passes **3,528 eager/graph comparisons** on 31 promising shapes. Thirty retain warm improvements; a 25-shape set also passes the evicted gate.
- The full 48-case corpus passes for the broad narrow selection, but model timing rejects both broad sets: improvements on batch-one one-frame inputs coexist with regressions on batch-eight one-frame and batch-one eight-frame inputs. Component timing alone is insufficient for adoption.
- The Gluon probe passes 144 actual-input configurations, **750 stress comparisons**, and **48 model cases**. Its component gains do not establish a consistent whole-model advantage; it remains research-only.
- Grouped model ablation selects **ten one-/two-row configurations**: four GEMV and six two-row tiles. Batch-one / one-frame gains are **1.0105× encode / 1.0088× decode**, with other measured geometries within 0.3% of parity. Replacing one two-row split path also removes a kernel launch. Runtime shape counts stay 44 small / 34 ordered, with four split shapes. The default-runtime, CuTe, long-stream and full-suite gates follow before commit.
- CPU inspection of six captured checkpoint weights rejects exact BF16/FP16 storage round-trips. A lossless block exponent representation has an estimated **1.135–1.138×** size ratio, including the stated metadata allowance; no codec or GPU speedup is implemented. This is a potential research direction, not a runtime precision change.


The integrated default-table gate (`full_narrow_runtime.json`) passes **48 cases**. Batch-one / one-frame encode improves **7.170 → 7.097 ms (1.0102×)** and decode **6.323 → 6.261 ms (1.0098×)**. Other measured cases stay within 0.3% of parity, including batch-two / one-frame; no additional gain is claimed for them. Three alternating restored contexts time five samples of ten graph calls, including copies and owned outputs, excluding setup/packing/capture/restoration. Peak allocation is **7.747 GB**. Wheel verification matches all **23 runtime/profile files** byte-for-byte.


The CuTe residual combination passes the same **48-case corpus** (`full_narrow_cute.json`). Two streams span **162 one-frame chunks over 12.96 seconds**, with one and two lanes. All **5,184 / 10,368 tokens** and **311,040 / 622,080 samples** match corrected original eager streaming chunk by chunk. Encoder codes also match offline. Original and optimized streaming share the same maximum waveform differences from offline decoding: **1.349e-6 / 1.952e-6**. Peak allocations are **7.526 / 7.696 GB** (`full_narrow_streaming_b1.json`, `full_narrow_streaming_b2.json`).

The batch-one / one-frame profile (`full_narrow_profile.json`) has **1,572 / 968 kernels** for encode/decode, down twelve per direction from the previous same-geometry profile. Fixed-row calls rise from 61 to **73**, while split/reduction pairs fall from 36 to **24**. There are still **129 / 128 GEMV calls**, consuming **45.26% / 51.24%** of kernel time. All disjoint matrix groups sum to **69.72% / 77.77%**; the nested small-kernel breakdown is already included. The compiler audit covers all **44 configurations**, with **40–128 registers**, zero spills and no matrix Tensor Core instructions (`narrow_resources.json`).

The fresh original/current comparison (`full_codec_narrow.json`) measures one-frame batch-one encode **46.434 / 10.339 / 7.010 ms** and decode **36.813 / 9.088 / 6.188 ms**, for original eager / original graph / current optimized graph. Direct total ratios are **6.62× / 5.95×** versus eager and **1.47× / 1.47×** versus original graphs. Batch-eight encode is **46.231 / 14.057 / 9.160 ms**, decode **37.406 / 12.509 / 7.874 ms**: **5.05× / 4.75×** versus eager and **1.53× / 1.59×** versus original graphs. All outputs and restored executions are exact. Three rotating rounds use ten single-call samples; graph copies and owned outputs are included, loading/packing/capture/restoration excluded. These totals have fresh denominators; historical incremental gains are not multiplied together.

The full suite passes **419 tests in 68.70 seconds**. The existing shape-driven tests exercise all ten changed configurations, random weights, arithmetic edge cases, native storage, hooks and graph lifetime without adding implementation-mirroring tests. This turn is **progress**: ten validated launch configurations, rejection of broader sets that regress codec latency, and new explicit-layout/storage evidence. **100× whole-model acceleration remains unmet.** The next investigations should quantify weight traffic and consider exact storage/decode or scheduling changes beyond warm-cache tile tuning; the block-format estimate is not an implemented optimization.

Reproduction (GPU commands sequential):

```bash
.venv/bin/python -m benchmarks.narrow_rows
.venv/bin/python -m benchmarks.narrow_confirm
.venv/bin/python -m benchmarks.small_layout
.venv/bin/python -m benchmarks.small_layout_confirm
.venv/bin/python -m benchmarks.narrow_model
.venv/bin/python -m benchmarks.narrow_model --groups --output results/full_narrow_groups.json
.venv/bin/python -m benchmarks.narrow_model --runtime --only-rows 1 2 --selection warm --output results/full_narrow_runtime.json
.venv/bin/python -m benchmarks.narrow_model --runtime --only-rows 1 2 --selection warm --fidelity-only --residual-backend cute --output results/full_narrow_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_narrow_streaming_b1.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 2 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_narrow_streaming_b2.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .08 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_narrow_profile.json
.venv/bin/python -m benchmarks.small_matrix_resources --output results/narrow_resources.json
.venv/bin/python -m benchmarks.codec_compare --frames 1 --output results/full_codec_narrow.json
.venv/bin/python -m pytest -q
.venv/bin/python -m benchmarks.weight_storage
uv build --wheel --out-dir /tmp/moss-narrow-wheel
```

All handles are terminal: narrow sweep `83835`, confirmation/initial Gluon compilation `4823` (confirmation completed; prototype compilation failed), corrected prototype attempt `2851` (compiler error), successful Gluon sweep `75929`, broad model comparison `82618`, Gluon stress `65087`, grouped model comparison `45108`, and runtime/CuTe/streams/profile/resources/upstream comparison/full-suite chain `64806`. CPU storage audit handles `66721` and `59614` are also complete. Only the pre-existing desktop GPU process remains; no benchmark is left running. Broader exact optimizations, serving/multi-GPU work and verified 100× whole-model acceleration remain open.

Final audit verifies the baseline snapshot against `fd559c3`, all ten runtime table replacements, unchanged ordered tables, exact report counts and outputs, compiler resource claims, profile subgroup accounting and wheel/source hashes. Python compilation and `git diff --check` pass. No GPU benchmark remains active.

## Logical weight traffic and exact packed-GEMV experiments

Previous goal turn classification: **progress**, verified at clean commit `e87f54d`, with ten tuned configurations, exact model/stream evidence and 419 passing tests. The 100× whole-model goal remains active and unmet.

- Added full-checkpoint call attribution. One-frame logical module-weight demand is **3.549 GB encode / 3.547 GB decode**, with **2.521 / 2.517 GB** in the one-row GEMVs. Hooks are removed before timing, and all six diagnostic cases remain exact. These are logical demand counts, not DRAM counters. The initial 11.641 ms encoder timing does not recur with longer warmup (7.006 ms); both artifacts are retained and no speedup claim uses the initial denominator.
- Implemented two lossless FP32 encodings and direct reconstruction kernels: variable-width exponent deltas and compact 28-bit blocks with verbatim FP32 fallback. Actual storage ratios are approximately **1.13×**, including metadata and padding. A cooperative CUDA decoder shares seven packed-word loads across eight reconstructed values. A preliminary Gluon attempt failed and was discarded; its incomplete evidence is explicitly marked.
- All **84 completed configurations** preserve captured outputs. **2,352 component stress comparisons** pass, along with exact reconstruction of checkpoint words and IEEE special-value payloads. Nevertheless, every format loses against the current kernels in both warm and evicted component timing, so none is integrated into runtime dispatch. Full-model compressed-memory savings or speedups are not claimed.
- Added thirteen round-trip, mixed-block and fused-arithmetic regressions. The focused suite passes **13 tests in 7.26 seconds**; the full suite passes **432 tests in 67.87 seconds**. Production runtime files remain unchanged. This turn is **research progress**, with exact implementations and evidence rejecting a proposed acceleration route; there is no additional runtime speedup.

Reproduction (GPU commands sequential):

```bash
.venv/bin/python -m benchmarks.weight_traffic --warmup 3 --output results/weight_traffic_initial.json
.venv/bin/python -m benchmarks.weight_traffic
.venv/bin/python -m benchmarks.packed_fp32_probe
.venv/bin/python -m benchmarks.packed_fp32_probe --format fixed28 --output results/packed_fp32_fixed28.json
.venv/bin/python -m benchmarks.packed_fp32_probe --format fixed28 --cooperative --output results/packed_fp32_cooperative.json
.venv/bin/python -m pytest tests/test_packed_fp32_research.py -q
.venv/bin/python -m pytest -q
```

All handles are terminal: initial traffic `49362`; variable-width compile attempt `48424` and successful probe `27349`; fixed28 indexing attempt `99834` and corrected probe `82504`; discarded Gluon attempt `2964` and diagnostics `93195`, `50887`, `95743`; successful CUDA cooperative probe `16027`; focused tests/refreshed traffic/full-suite chain `1500`. No benchmark remains running. Actual memory-system counters, further exact kernel schedules, broader serving/multi-GPU work and verified 100× acceleration remain open.

Final audit verifies all 23 production runtime/profile hashes against the preceding wheel, 2,352 exact component comparisons, rejection of every tested packed variant on both timing regimes, actual encoded sizes, repeated logical call counts and the 432-test report. Python compilation and diff checks pass. Only the pre-existing desktop GPU process remains.

## CUDA vector loads and ordered GEMV prefetch

Previous goal turn classification: **progress**, verified at clean commit `936b0e4`, with exact lossless packing probes, logical weight attribution and 432 passing tests. The 100× whole-model goal remains active and unmet.

- Added research CUDA/NVRTC kernels that vectorize consecutive native FP32 lanes and prefetch ordered windows without changing the arithmetic. The **336-configuration** sweep is captured-input bit-exact and spill-free. Twenty-two finalists pass **660 eager/graph stress comparisons**, with another 360 passing control comparisons.
- Generated-instruction inspection covers **24 kernels**. All eleven vectorized examples emit actual 64-/128-bit global loads; the scalar controls remain scalar. No inspected kernel spills or uses matrix Tensor Core instructions. Corrected a PTX counter that missed qualified loads; removed its incomplete sweep fields and retained the corrected finalist/SASS evidence. The first inspection invocation failed for lack of a materialized CUDA context and was rerun successfully after fixing initialization.
- Only the one-row 1280→5120 FFN projection retains a warm gain: **1.0499×**, versus **0.9091×** after eviction. Its full-model ablation passes **48 cases**, including exact token, hidden-state, waveform and restored-output bits. It regresses one-frame batch-one encode/decode by **0.47% / 0.49%**. Other geometries do not call the candidate and show only timing variation. The candidate is rejected for runtime adoption.
- Added thirteen arithmetic, graph and input-contract regressions. Focused tests pass **13 in 5.19 seconds**; the complete suite passes **445 in 71.43 seconds**. All **23 runtime/profile hashes** still match `narrow_package.json`. No new codec speedup or runtime memory reduction is claimed. This turn is **research progress**, with a measured rejection of wider loads/prefetch as a default GEMV replacement.

Reproduction (GPU commands sequential):

```bash
.venv/bin/python -m benchmarks.gemv_vector_tune
.venv/bin/python -m benchmarks.gemv_vector_confirm
.venv/bin/python -m benchmarks.gemv_vector_instructions
.venv/bin/python -m pytest tests/test_gemv_vector_research.py -q
.venv/bin/python -m benchmarks.gemv_vector_model
.venv/bin/python -m pytest -q
```

All handles are terminal: sweep `11540`, confirmation `91168`, initial instruction inspection `11530` (context initialization error, no tests run), corrected instruction inspection/focused tests/model chain `30038`, full suite `74126`. The final audit checks every comparison, candidate dispatch counts, emitted load widths, unchanged production hashes, compilation and diff hygiene. Actual memory-system counters, more effective exact data-access schedules, broader serving/multi-GPU work and the 100× whole-model goal remain open.

## Exact interleaved GEMV storage and weight-ring diagnostics

Previous goal turn classification: **progress**, verified at clean commit `dafaa46`, with 336 CUDA configurations, exact model gates, measured rejection of the candidate and 445 passing tests. The 100× goal remains active and unmet.

- Implemented an FP32 output-group layout with integer-bit packing/unpacking and an unchanged native reduction. All **324 configurations** preserve captured outputs, with zero spills or matrix Tensor Core instructions. Twenty finalists pass **600 candidate plus 360 control** eager/graph stress comparisons.
- Repeated component gains reach **17–18%** on QKV and the expanding FFN, but codec gains are much smaller. Both warm and cold selections pass **48 model cases** each. One-frame batch-one encode improves **0.57% / 1.02%**, decode **0.72% / 0.72%**, respectively. These prototypes add **4.618 / 4.614 GB** of packed weights, with peak allocation **12.370 / 12.366 GB**; packing setup is separately recorded. They remain outside production dispatch because the small codec gain does not justify that default memory cost.
- Added distinct-allocation rings with 1/4/16/32 copies of a captured weight. All twelve cases pass. QKV's **1.176×** one-allocation gain becomes **1.0033×** at 32 allocations; expanding-FFN alternatives become **1.0137× / 1.0165×**. This changes the next tuning procedure: test distinct weight allocations early instead of selecting primarily on repeated access to one buffer. Rings are synthetic component diagnostics, not physical DRAM counters or full-model speedups.
- Added **28** storage, arithmetic, graph and invalid-contract tests. Focused tests pass in **4.25 seconds**; the full suite passes **473 in 81.06 seconds**. All **23 production runtime/profile hashes** remain unchanged. This turn is **research progress**, with a new exact layout, bounded full-model evidence and a diagnostic explaining why much of the component gain disappears. No additional supported-runtime speedup is claimed.

Reproduction (GPU commands sequential):

```bash
.venv/bin/python -m benchmarks.gemv_interleaved_tune
.venv/bin/python -m benchmarks.gemv_interleaved_confirm
.venv/bin/python -m pytest tests/test_gemv_interleaved_research.py -q
.venv/bin/python -m benchmarks.gemv_interleaved_model
.venv/bin/python -m benchmarks.gemv_interleaved_model --selection cold --output results/full_gemv_interleaved_cold.json
.venv/bin/python -m benchmarks.gemv_weight_ring
.venv/bin/python -m pytest -q
```

All handles are terminal: sweep `29275`, confirmation/focused tests `11924`, warm model `66852`, cold model `65896`, rings/full-suite chain `7162`. The warm model report predates the selection CLI; its subsequently added selection label is explicitly documented metadata, with measured samples unchanged. Final audit verifies every comparison, dispatch and packing count, all twelve rings, unchanged production hashes, Python compilation and diff hygiene. Further exact scheduling, physical memory-system attribution, broader serving/multi-GPU work and verified 100× acceleration remain open.

## Integrated one-row FFN epilogues

Previous goal turn classification: **progress**, verified at clean commit `21cb936`, with exact interleaved layouts, full-model packing-cost evidence, distinct-allocation diagnostics and 473 passing tests. The 100× goal remains active and unmet.

- Attempted actual CUPTI collection using the installed headers/library. Metadata setup succeeds, but fetching the availability image and later starting collection return **`CUPTI_ERROR_HARDWARE_BUSY`**. Both attempts are retained, with **no counters collected** and no driver/permission changes. The owning client remains unidentified; this is not treated as a whole-goal blocker.
- Implemented exact native GEMV/GELU and GEMV/scaled-residual fusion. All **48 configurations** pass **1,440 stress comparisons**. Using distinct-allocation rings selects the existing native launch geometries, with **1.0229× / 1.0206×** component gains at 32 allocations. No weight packing or additional persistent workspace is required.
- Research and integrated runtime ablations each pass **48 cases**. The integrated one-frame batch-one gate improves encode **7.097 → 7.036 ms (1.0088×)** and decode **6.266 → 6.205 ms (1.0098×)**. Other measured geometries stay within 0.18% of parity. The implementation is included under the existing `ffn_backend="triton"` option, retaining the 24-row path and hook/custom-forward/graph safeguards.
- The CuTe combination passes **48 cases**. Both **162-chunk / 12.96-second** streams match corrected original eager streaming exactly: **5,184 / 10,368 tokens** and **311,040 / 622,080 waveform samples**. Integrated peak allocation is **7.723 GB**, stream peaks **7.526 / 7.696 GB**.
- The profile confirms **64 fewer launches per direction**, leaving **1,508 encode / 904 decode kernels**. Fused GEMVs are counted inside the matrix group, including their epilogues. Fresh original/current ratios are **6.86× encode / 6.17× decode** versus original eager at batch one / one frame, and **1.49× / 1.48×** versus original graphs. These are direct fresh-denominator totals; the approximately 1% isolated improvement is not multiplied into historical ratios.
- The compiler audit passes for both integrated kernels, with zero spills and no matrix Tensor Core instructions. The expanded suite passes **501 tests in 80.75 seconds**. The wheel matches all **23 runtime/profile files**. This turn is **progress**: a verified runtime optimization plus concrete evidence about unavailable hardware-counter collection. The broader 100× objective remains unmet.

Reproduction (GPU commands sequential):

```bash
.venv/bin/python -m benchmarks.counter_access
.venv/bin/python -m benchmarks.gemv_counters
.venv/bin/python -m benchmarks.gemv_epilogue_probe
.venv/bin/python -m benchmarks.gemv_epilogue_model
.venv/bin/python -m pytest tests/test_ffn.py tests/test_gemv_epilogue_research.py -q
.venv/bin/python -m benchmarks.gemv_epilogue_model --runtime --output results/full_gemv_epilogue_runtime.json
.venv/bin/python -m benchmarks.gemv_epilogue_model --runtime --fidelity-only --residual-backend cute --output results/full_gemv_epilogue_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_gemv_epilogue_streaming_b1.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 2 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_gemv_epilogue_streaming_b2.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .08 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --output results/full_gemv_epilogue_profile.json
.venv/bin/python -m benchmarks.codec_compare --frames 1 --output results/full_codec_gemv_epilogue.json
.venv/bin/python -m pytest -q
.venv/bin/python -m benchmarks.gemv_epilogue_resources
uv build --wheel --out-dir /tmp/moss-gemv-epilogue-wheel
```

All handles are terminal: initial/extended metadata probes `37476` / `67384`, component search `41486`, first collection attempt `13719`, revised collection/research-model chain `73703`, initial focused tests `7436`, integrated/CuTe/streams/profile/upstream/full-suite chain `4232`, compiler resources `75719`. Wheel building completes successfully and its temporary repository build output is removed. Final audit verifies every corpus/stream/component check, kernel-count reduction, matrix-group accounting, direct performance ratios and wheel/source hashes. Python compilation and diff checks pass. Physical counter attribution, further exact matrix/launch optimizations, broader serving/multi-GPU execution and verified 100× acceleration remain open.

## Integrated exact CUDA LayerNorm

Previous goal turn classification: **progress**, verified at clean commit `52a2c91`, with integrated native FFN epilogues, 501 passing tests, exact codec/streaming gates and measured launch reductions. The 100× goal remains active and unmet.

- Reproduced the native PyTorch 2.8 Welford tree in CUDA and captured 20 actual LayerNorm geometries. All **200** dynamic-count configurations pass. The first constant-count variant passes only **30/200** because compiler-selected FMA orientation changes mean bits. Explicitly preserving the native rounded-product/FMA order fixes all **200** configurations. The failed experiment is retained.
- Corrected finalists pass **4,320** native/candidate output, mean and reciprocal-standard-deviation comparisons. Component gains are **1.49–1.71×**. Tests additionally cover zero epsilon, partial row grids, signed zeros, subnormals, large inputs, infinities and NaNs. These results support the pinned arithmetic implementation, not arbitrary reordering of LayerNorm.
- Added optional `norm_backend="cuda"`, requiring the `normalization` extra and recorded GPU/PyTorch/checkpoint profile. Reversible per-module wrappers retain existing FFN fusion, hooks, custom-forward fallbacks, gradient/autocast/layout/epsilon fallbacks, stream warmup and graph-lifetime guards. No parameter packing or persistent activation workspace is added.
- Research, supported-runtime and CuTe combinations each pass **48 full-checkpoint cases**. The integrated batch-one / one-frame ablation improves encode **7.035 → 6.940 ms (1.0137×)** and decode **6.210 → 6.078 ms (1.0218×)**. Batch-eight gains are **1.0093× / 1.0164×**. At batch one / three frames, encode is effectively unchanged (**1.0009×**), while decode improves **1.0206×**. Peak allocation is **7.721 GB**.
- Both **162-chunk / 12.96-second** streams remain exact against corrected eager streaming: **5,184 / 10,368 tokens** and **311,040 / 622,080 samples** for one/two lanes. Their preexisting decoder discrepancy from offline execution is unchanged. Peak stream allocations are **7.526 / 7.696 GB**.
- The profile retains **1,508 encoder / 904 decoder kernels**, including 64 fused FFN GEMVs per direction. CUDA LayerNorm replaces **88 encoder / 136 decoder** calls, leaving 48 native encoder calls. LayerNorm uses approximately **4.34% / 4.28%** of kernel time; matrix groups remain **72.03% / 81.37%**.
- A fresh original/current comparison measures **6.870 / 6.001 ms** for batch-one / 80 ms encode/decode: **6.89× / 6.25×** versus original eager, **1.51× / 1.52×** versus original graphs. Ratios use fresh denominators; isolated gains are not multiplied into historical results.
- The full suite passes **554 tests in 84.14 seconds**. Four integrated compiler variants use **30–38 registers**, **0–48 bytes shared memory**, no local-memory spills and no matrix Tensor Core instructions. The wheel matches all **24 runtime/profile files** and passes an isolated import. This turn is **progress**, with a verified runtime optimization; the broader 100× goal remains unmet.

Reproduction (GPU commands sequential):

```bash
.venv/bin/python -m benchmarks.native_layer_norm_probe
.venv/bin/python -m benchmarks.layer_norm_inputs
.venv/bin/python -m benchmarks.native_layer_norm_tune
.venv/bin/python -m benchmarks.native_layer_norm_confirm
.venv/bin/python -m benchmarks.native_layer_norm_tune --fixed --output results/native_layer_norm_fixed.json
.venv/bin/python -m benchmarks.native_layer_norm_tune --fixed --fixed-mode 3 --output results/native_layer_norm_fixed_corrected.json
.venv/bin/python -m benchmarks.native_layer_norm_confirm --input results/native_layer_norm_fixed_corrected.json --output results/native_layer_norm_fixed_confirm.json
.venv/bin/python -m pytest tests/test_native_layer_norm_research.py -q
.venv/bin/python -m benchmarks.native_layer_norm_model
.venv/bin/python -m pytest tests/test_normalization.py tests/test_ffn.py -q
.venv/bin/python -m benchmarks.native_layer_norm_model --runtime --output results/full_native_layer_norm_runtime.json
.venv/bin/python -m benchmarks.native_layer_norm_model --runtime --fidelity-only --residual-backend cute --output results/full_native_layer_norm_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_native_layer_norm_streaming_b1.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 2 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_native_layer_norm_streaming_b2.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .08 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_native_layer_norm_profile.json
.venv/bin/python -m benchmarks.codec_compare --frames 1 --norm-backend cuda --output results/full_codec_native_layer_norm.json
.venv/bin/python -m pytest -q
.venv/bin/python -m benchmarks.native_layer_norm_resources
uv build --wheel --out-dir /tmp/moss-native-layer-norm-wheel
```

All handles are terminal: initial arithmetic/capture `51809` / `8204`, dynamic sweep/confirmation `87929` / `94242`, first constants/order diagnosis `76476` / `9811`, corrected sweep/confirmation `15640`, research model/tests `56678` / `17637`, initial/corrected focused fixtures `52180` / `38424`, supported model `59923`, CuTe `53631`, streams `83767` / `84666`, profile `54734`, original comparison `49283`, full suite `37916`, resource-audit initial/context-corrected calls `70005` / `62711`, wheel import `94706`. The initial focused failures were training-mode parent test fixtures; the initial resource audit needed an allocated tensor to establish its CUDA context. Both are corrected, and final gates pass. Temporary repository wheel-build output is removed. Final audit verifies the retained arithmetic failures, corrected comparisons, full-model/streaming exactness, profile dispatch, direct speed ratios and package hashes. Further exact matrix work, noncontiguous encoder normalization, physical hardware-counter attribution, broader serving/multi-GPU execution and verified 100× acceleration remain open.

## Integrated normalization for transposed encoder inputs

Previous goal turn classification: **progress**, verified at clean commit `6bdd709`, with exact CUDA LayerNorm integration, 554 passing tests, exact corpus/streaming gates and an audited package. The GPU had only the preexisting 29 MiB client, with no task job still running. The 100× goal remains active and unmet.

- Captured actual optimized-runtime LayerNorm inputs without replacing its owned dispatch semantics. Eighteen dense channel/time transposes explain the remaining **48 encoder calls at 80 ms / 112 at 240 ms**. Tested direct strided loads and cooperative shared-memory staging, including aligned padding variants, while retaining the native Welford reduction and affine arithmetic.
- All **396 configurations** pass captured output/statistic gates. Finalists pass **3,888 eager/graph comparisons**. Confirmed component gains are **1.23–2.18×** versus native copy plus normalization. Initial single-sweep ratios were wider; repeated measurements determine promotion.
- Extended `norm_backend="cuda"` to the validated layouts, with original fallbacks for other shapes/strides and separate graph warmup keys for contiguous/transposed inputs. The contiguous arithmetic source remains unchanged. No parameter packing, precision conversion or persistent activation workspace is added. The standard fidelity CLI also accepts the normalization option.
- Research and integrated runtime gates each pass **48 cases**, including the quantizer near-tie regression and 128-lane input. The integrated encoder ablation improves **6.940 → 6.864 ms / 9.143 → 9.038 ms** at batch one/eight for 80 ms inputs, and **8.231 → 8.022 ms / 10.347 → 10.151 ms** at 240 ms. Gains are **1.1–2.6%**; decoder medians stay within **0.09%** of parity. Integrated peak allocation is **7.741 GB** under the expanded four-geometry timing workload.
- The CuTe combination passes another **48 cases**. Both **162-chunk / 12.96-second** streams match corrected eager streaming exactly: **5,184 / 10,368 tokens** and **311,040 / 622,080 samples**, with peaks **7.526 / 7.696 GB**. Their preexisting decoder difference from offline execution is unchanged.
- The 80 ms profile removes **48 encoder launches**, leaving **1,460 / 904** encode/decode kernels. Both 80/240 ms profiles show **136 CUDA / zero native LayerNorm calls** per direction at the measured batch-one geometries. Matrix groups remain **73.40% / 81.43%** of 80 ms kernel time, identifying the larger remaining cost.
- Fresh original/current comparisons measure **6.787 / 6.002 ms** at batch one / 80 ms, or **6.90× / 6.18×** versus original eager and **1.52× / 1.52×** versus original graphs. At 240 ms the totals are **7.897 / 6.582 ms**, or **6.09× / 5.82×** versus eager. These are matched fresh denominators; isolated gains are not multiplied into historical ratios.
- Focused tests pass **98 in 20.83 seconds**; the full suite passes **598 in 102.03 seconds**. Ten compiler variants use **30–39 registers**, **0–12,288 bytes shared memory**, zero local-memory spills and no matrix Tensor Core instructions. The wheel matches all **25 runtime/profile files** and passes an isolated import. This turn is **progress**, with a verified runtime optimization. The broader 100× goal remains unmet.

Reproduction (GPU commands sequential):

```bash
.venv/bin/python -m benchmarks.strided_layer_norm_inputs
.venv/bin/python -m benchmarks.strided_layer_norm_tune
.venv/bin/python -m benchmarks.strided_layer_norm_confirm
.venv/bin/python -m benchmarks.strided_layer_norm_model
.venv/bin/python -m pytest tests/test_strided_layer_norm_research.py tests/test_strided_normalization.py tests/test_normalization.py tests/test_ffn.py -q
.venv/bin/python -m benchmarks.strided_layer_norm_model --runtime --output results/full_strided_layer_norm_runtime.json
.venv/bin/python -m benchmarks.strided_layer_norm_model --runtime --fidelity-only --residual-backend cute --output results/full_strided_layer_norm_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_strided_layer_norm_streaming_b1.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 2 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_strided_layer_norm_streaming_b2.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .08 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_strided_layer_norm_profile_f1.json
.venv/bin/python -m benchmarks.codec_compare --frames 1 --norm-backend cuda --output results/full_codec_strided_layer_norm_f1.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .24 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend triton --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_strided_layer_norm_profile_f3.json
.venv/bin/python -m benchmarks.codec_compare --frames 3 --norm-backend cuda --output results/full_codec_strided_layer_norm_f3.json
.venv/bin/python -m pytest -q
.venv/bin/python -m benchmarks.strided_layer_norm_resources
uv build --wheel --out-dir /tmp/moss-strided-layer-norm-wheel
```

All handles are terminal: capture `40433`, sweep `57237`, confirmation `20179`, research model `5658`, focused tests `1642`, integrated model `63892`, sequential CuTe/streams/profiles/original-comparisons/full-suite/resources chain `33314`, isolated wheel import `84105`. The chain checks every subprocess exit before starting the next GPU job. Temporary repository wheel-build output is removed. Final cross-report audit verifies exactness, dispatch and launch counts, direct ratios, compiler resources and package hashes. Further exact matrix scheduling/fusion, physical memory-system counters, broader serving/multi-GPU execution and verified 100× acceleration remain open.

## Integrated single-block matrix partitions

Previous goal turn classification: **progress**, verified at commit `c2e4722`, with exact strided CUDA normalization, 598 passing tests and audited corpus, streams and package. No previous task GPU job remained active. The 100× goal remains active and unmet.

- Profiled remaining native matrices without observer hooks or replacing the optimized forward path. Actual ATen operator/shape entries identify eight small-row transformer geometries; custom-launch runtime API associations are retained with an explicit attribution limitation. Eager attribution timings are not whole-codec graph measurements.
- Implemented single-block CUDA K partitions with shared-memory exchange, one barrier, explicit FP32 FMA/add ordering and original contiguous weights. Corrected an initial shuffle-membership assumption before promotion; initial empirical reports/source are retained as superseded evidence. The corrected CUDA sweep passes **432 configurations**, and the alternative Triton layout passes **288**.
- Each implementation passes **480 eager/graph stress comparisons** plus exact output checks on **24 allocation-ring cases**. Synthetic distinct-allocation rings inform selection; no physical memory-counter claim is made. Five CUDA shapes show **1.06–1.43×** component gains. The less consistent warm-selected set and the Triton port remain research evidence, not additional runtime optimizations.
- Added optional `matrix_backend="cuda"` and pinned `cuda` extra. Existing matrix ownership, fallbacks, packing for other shapes, FFN fusion, stream warmup and graph lifetime rules remain active. The five new kernels add neither persistent weight storage nor a global partial-results buffer.
- Research ring/warm selections and integrated runtime each pass **48 full-model cases**. The integrated paired batch-one / 80 ms encode/decode improves **6.860 → 6.833 ms / 6.072 → 6.047 ms**, about **0.4%**. At 240 ms it improves **8.027 → 7.971 ms / 6.664 → 6.620 ms**, about **0.7%**. Batch-eight controls make no new calls and stay within **0.11%** of parity. Peak allocation is **7.723 GB**. These paired increments are separate from fresh original/current ratios.
- Focused research/runtime tests pass **53 in 20.02 seconds**. The wheel matches all **26 runtime/profile files** byte-for-byte and passes an isolated CUDA-module import. Compute Sanitizer was not found in the searched tool locations; no sanitizer result is claimed.
- The CuTe combination passes **48 cases**. Both **162-chunk / 12.96-second** streams match corrected eager streaming, with **5,184 / 10,368 tokens** and **311,040 / 622,080 samples**. Peak allocations are **7.526 / 7.696 GB**; their existing decoder differences from offline execution are unchanged.
- The full suite passes **651 tests in 117.71 seconds**. The five production kernels use **40 registers**, **2,304–9,216 bytes shared memory**, zero local-memory spills and no matrix Tensor Core instructions. Corrected research/production source hashes match. This turn is **progress**, with a verified runtime improvement; the 100× objective remains unmet.
- Fresh matched original/current comparisons give **6.758 / 5.970 ms** encode/decode at batch one / 80 ms, or **7.01× / 6.28×** versus eager and **1.53× / 1.52×** versus original graphs. At 240 ms, totals are **7.866 / 6.541 ms**, or **6.10× / 5.85×** versus eager. Historical ratios are not multiplied.

Reproduction (GPU commands sequential):

```bash
.venv/bin/python -m benchmarks.native_matrix_attribution
.venv/bin/python -m benchmarks.cta_tiled_tune
.venv/bin/python -m benchmarks.cta_tiled_confirm
.venv/bin/python -m benchmarks.cta_tiled_triton_tune
.venv/bin/python -m benchmarks.cta_tiled_triton_confirm
.venv/bin/python -m benchmarks.cta_tiled_model --output results/full_cta_tiled_ring.json
.venv/bin/python -m benchmarks.cta_tiled_model --selection warm --output results/full_cta_tiled_warm.json
.venv/bin/python -m pytest tests/test_cta_tiled_research.py tests/test_cuda_matrix_runtime.py -q
.venv/bin/python -m benchmarks.cta_tiled_model --runtime --output results/full_cta_tiled_runtime.json
.venv/bin/python -m benchmarks.cta_tiled_model --runtime --fidelity-only --residual-backend cute --output results/full_cta_tiled_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cuda --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_cta_tiled_streaming_b1.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 2 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cuda --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_cta_tiled_streaming_b2.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .08 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cuda --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_cta_tiled_profile_f1.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .24 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cuda --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_cta_tiled_profile_f3.json
.venv/bin/python -m benchmarks.codec_compare --frames 1 --matrix-backend cuda --norm-backend cuda --output results/full_codec_cta_tiled_f1.json
.venv/bin/python -m benchmarks.codec_compare --frames 3 --matrix-backend cuda --norm-backend cuda --output results/full_codec_cta_tiled_f3.json
.venv/bin/python -m pytest -q
.venv/bin/python -m benchmarks.cta_tiled_resources
uv build --wheel --out-dir /tmp/moss-cta-matrix-wheel
```

The profiles verify **24 / 36 new CUDA calls per direction** for one-/three-frame inputs, unchanged total launch counts and 136 CUDA normalization calls per direction. Two initial short-chunk encoder profile measurements remain slow despite identical dispatch; they are preserved. Added an explicit, reported `--warmup-replays` option. Repeating the one-frame command with `--warmup-replays 200 --output results/full_cta_tiled_profile_f1_warm.json` yields **6.753 ms** encoder wall time, consistent with the independently rotated codec comparison. Its matrix groups still occupy **72.49% / 81.95%** of kernel time. The device-state cause of the earlier transient is not established.

All handles are terminal: eager attribution `5249`; superseded initial CUDA sweep/confirmation/model `68878` / `51223` / `58834`; Triton sweep `50931`; corrected sequential confirmation/sweep/model chain `95200`; focused tests `41949`; integrated runtime `17340`; sequential CuTe/streams/profiles/original-comparisons/full-suite/resource chain `62435`; isolated wheel import `85375`; repeated/warmed short profile `18020` / `67885`. GPU jobs ran sequentially. The final audit checks source/configuration provenance, exactness, profile dispatch, directly computed speed ratios, compiler resources and package bytes. Temporary repository wheel-build output is removed. Further exact matrix work, physical counter attribution, broader serving/multi-GPU execution and verified 100× acceleration remain open.

## Long-K shared-partial investigation

Previous goal turn classification: **progress**, verified at clean commit `397db43`, with five exact CUDA matrix shapes, 651 passing tests, bit-identical corpus/streaming gates and an audited package. Only the preexisting 29 MiB GPU client remained; no previous task GPU job was active. The 100× goal remains active and unmet.

The next investigation targets the four remaining global partial/reduction pairs and related long-K fixed-row matrices. A CUDA block computes one or several independent 256-term partitions per thread group, stores them in shared memory, synchronizes, then retains the native ordered FP32 reduction. Selection must pass repeated distinct-allocation and full-codec gates before any runtime change.

- All **1,080 configurations** across nine long-K shapes reproduce captured outputs bit for bit, with zero local-memory bytes. Repeated finalists pass **720 eager/graph stress comparisons** including native/current controls, plus exact outputs across **27 allocation-ring cases**. Six choices retain **1.04–1.23×** component gains; three larger-row alternatives are rejected.
- The 48-case research gate is exact. A first batch-eight decoder timing difference triggers a five-round repeat with direction-specific capture counts. That decoder makes no candidate calls, and its timing difference reverses sign. The repeated evidence retains batch-one benefits while treating the batch-eight decoder as a control.
- Added six native-storage long-K kernels to `matrix_backend="cuda"`, covering all four small-matrix split/reduction shapes and two fixed-row replacements. The five previous CUDA shapes and all other existing matrix paths remain available. Each new kernel exchanges independent 256-term partitions in shared memory and performs the native ordered FP32 reduction, without a global partial buffer or persistent weight copy.
- All **48 integrated runtime cases** are bit-identical. Five rotating timing rounds measure batch-one / 80 ms encode/decode **6.831 → 6.780 ms / 6.046 → 5.997 ms**, about **0.8% lower latency**. At 240 ms, **7.963 → 7.768 ms / 6.625 → 6.435 ms** gives **2.4% / 2.9% lower latency**. Batch-eight results stay within **0.09%** of parity. Peak allocation is **7.723 GB**.
- The **35 focused tests pass in 11.82 seconds**, covering research arithmetic/tails and runtime storage, fallback, graph and fusion behavior. The wheel matches all **27 runtime/profile files** and passes isolated import. No dependency changes are needed beyond the existing pinned CUDA extra.

Reproduction (GPU commands sequential):

```bash
.venv/bin/python -m benchmarks.wide_cta_tune
.venv/bin/python -m benchmarks.wide_cta_confirm
.venv/bin/python -m pytest tests/test_wide_cta_research.py -q
.venv/bin/python -m benchmarks.wide_cta_model --output results/full_wide_cta_ring.json
.venv/bin/python -m benchmarks.wide_cta_model --timing-only --rounds 5 --output results/full_wide_cta_repeat.json
.venv/bin/python -m pytest tests/test_wide_matrix_runtime.py tests/test_wide_cta_research.py -q
.venv/bin/python -m benchmarks.wide_cta_model --runtime --rounds 5 --output results/full_wide_cta_runtime.json
.venv/bin/python -m benchmarks.wide_cta_model --runtime --fidelity-only --residual-backend cute --output results/full_wide_cta_cute.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 1 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cuda --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_wide_cta_streaming_b1.json
.venv/bin/python -m benchmarks.streaming_fidelity --batch 2 --frames 162 --chunk-frames 1 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cuda --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_wide_cta_streaming_b2.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .08 --warmup-replays 200 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cuda --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_wide_cta_profile_f1.json
.venv/bin/python -m benchmarks.profile_graph --batch 1 --seconds .24 --warmup-replays 200 --share-rope-tables --attention-mask-backend triton --quantizer-backend triton --matrix-backend cuda --projection-backend triton --ffn-backend triton --norm-backend cuda --output results/full_wide_cta_profile_f3.json
.venv/bin/python -m benchmarks.codec_compare --frames 1 --matrix-backend cuda --norm-backend cuda --output results/full_codec_wide_cta_f1.json
.venv/bin/python -m benchmarks.codec_compare --frames 3 --matrix-backend cuda --norm-backend cuda --output results/full_codec_wide_cta_f3.json
.venv/bin/python -m pytest -q
.venv/bin/python -m benchmarks.wide_cta_resources
uv build --wheel --out-dir /tmp/moss-wide-cta-wheel
```

The CuTe combination passes **48 cases**. Both **162-chunk / 12.96-second** streams remain exact against corrected eager streaming: **5,184 / 10,368 tokens**, **311,040 / 622,080 samples**, and peak allocations **7.522 / 7.695 GB**. Their existing decoder difference from offline execution is unchanged.

Warmed profiles show **36 / 44 long-K CUDA calls per direction** at one/three frames. They remove **24 / 32 launches per direction**, leaving **1,436 / 880** and **1,522 / 995** encoder/decoder kernels, respectively. Both profiles retain the prior narrow-K CUDA dispatch and all 136 CUDA LayerNorm calls per direction, with zero small-matrix global partial/reduction pairs. Matrix groups still occupy **73.05% / 81.73%** of one-frame kernel time.

Fresh matched comparisons give **6.711 / 5.924 ms** batch-one encode/decode at 80 ms, or **7.08× / 6.34×** versus original eager and **1.54× / 1.54×** versus original graphs. At 240 ms, totals are **7.677 / 6.347 ms**, or **6.27× / 6.06×** versus eager. These direct ratios use fresh denominators; historical increments are not multiplied.

The full suite passes **686 tests in 128.37 seconds**. Six compiler variants use **38–40 registers**, **3,072–24,576 shared bytes**, zero local-memory bytes and no matrix Tensor Core instructions. Source hashes and selected configurations match the research/runtime reports, and the 27-file package audit passes. This turn is **progress**, with a verified integrated optimization; the 100× goal remains active and unmet.

All handles are terminal: sweep `45296`; confirmation/research-test chain `84459`; research model `49341`; five-round timing repeat `53615`; focused integration tests `52046`; integrated model `89065`; sequential CuTe/streams/profiles/original-comparisons/full-suite/resource chain `46076`; isolated wheel import `75565`. GPU jobs ran sequentially and only the preexisting 29 MiB client remains. Temporary repository wheel-build output is removed. Final audit verifies arithmetic/source provenance, selection, output equality, negative-control counts, launch removal, direct ratios, compiler resources and package bytes. Further exact matrix/FFN fusion, physical memory-system counters, broader serving/multi-GPU execution and verified 100× acceleration remain open.
