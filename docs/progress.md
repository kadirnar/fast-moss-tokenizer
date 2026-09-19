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
