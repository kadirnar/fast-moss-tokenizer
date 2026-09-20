# Fast MOSS Audio Tokenizer

Ongoing GPU optimization of the **original 1.6B MOSS Audio Tokenizer**, retaining FP32 weights, all 32 quantizers, and its learned architecture. No distillation, FP8, or FP4. **100× whole-model acceleration has not been demonstrated.**

Implemented: reversible inference caches for normalized codebooks and convolution weights; bitwise FP32 residual fusion in Triton and CuTe DSL (explicit CUDA PTX rounding); Triton RoPE with optional stage-shared tables, fused/shared attention masks, and ring-cache kernels; CUDA graphs; incremental encoder/decoder sessions with independently pausable, finishable, and reusable batch lanes; fused lane reset; incremental request scheduling with fused fragment gather and optional input byte limits; optional exact quantizer fusion; version-gated resident FP32 cuBLASLt, ordered Triton matrices and optional single-block CUDA partitions; fused LFQ output projections and cached decoder reconstruction; optional FFN pipeline tuning and exact short-row/24-row GELU/residual fusion; optional CUDA LayerNorm preserving the native Welford reduction, including fused one-row QKV/GELU projections; exact attention output projection and scaled residual fusion for nine native matrix shapes.

Fresh full-checkpoint measurements on RTX 5070 Ti, one 240 ms speech input, FP32 and all 32 codebooks:

| Operation | Original eager | Original graph | Optimized graph | Speedup vs eager |
| --- | ---: | ---: | ---: | ---: |
| Encode | 47.417 ms | 11.940 ms | 7.356 ms | 6.45× |
| Decode | 38.007 ms | 10.183 ms | 6.186 ms | 6.14× |

Three rotating-order rounds use independent restored contexts, 200 extra graph warmups per context, and ten samples per mode/direction. Graph timings include input copies and owned outputs; loading, packing and capture are excluded. Tokens, hidden states, waveforms and restored outputs are exact. At batch eight, the measured graph latency is **10.024 ms encode / 8.679 ms decode**, or **4.94× / 4.57×** versus original eager at the same batch. Relative to the original graph, gains are **1.62× / 1.65×** at batch one and **1.89× / 1.98×** at batch eight. [240 ms matched comparison](results/full_codec_attention_residual_f3.json). The [earlier seeded-input comparison](results/full_compare_lane_reset_240ms.json) remains historical evidence. **100× whole-model acceleration has not been demonstrated.**

For the shortest **80 ms, batch-one chunk**, the exact kernels give **6.488 ms encode / 5.779 ms decode**. A fresh matched comparison measures **7.26× / 6.46×** versus original eager and **1.60× / 1.58×** versus original CUDA graphs, with the same extra warmups as the 240 ms table. [Warmed one-frame comparison](results/full_codec_attention_residual_f1.json). These overall ratios are directly measured, not products of historical incremental gains.

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

## Resident FP32 matrix backend

`optimized(..., matrix_backend="cublaslt")` enables the measured matrix choices as an opt-in runtime option. The bundled profile requires the pinned checkpoint revision, RTX 5070 Ti with 70 SMs, PyTorch 2.8.0+cu128, cuBLASLt 12.8.4, and TF32 disabled. Mismatched environments are rejected; unprofiled shapes and unsupported input layouts use the original contiguous-weight arithmetic.

```python
import torch
from fast_moss import load_model
from fast_moss.graphs import GraphedCallable
from fast_moss.optimize import optimized

model = load_model()
x = torch.zeros(8, 1, 5760, device="cuda")
with torch.inference_mode(), optimized(
    model, residual_backend="triton", rope_backend="triton", kv_backend="triton",
    share_rope_tables=True, attention_mask_backend="triton",
    quantizer_backend="triton", matrix_backend="cublaslt",
):
    encode = GraphedCallable(lambda audio: (model._encode_frame(audio).audio_codes,), x)
    codes = encode(x)[0]
    del encode
```

Weights are transposed into resident FP32 storage only when reached by a validated matrix shape. Parameter identity and values are preserved; storage and strides change until restoration on exit. Registered aliases and tied parameters are rejected before mutation. Do not retain external weight-storage aliases, mutate weights, or use raw CUDA graphs across storage transitions. Use one host thread. Each CUDA stream owns separate plans and a 32 MiB workspace; packing and restoration synchronize the device.

Keep sessions and graphs inside the context. **Entering/exiting this backend or packing another weight invalidates all managed `GraphedCallable` objects on that device**, including unrelated graphs. Stale replay raises before submitting GPU work. Warm all needed shapes before constructing a collection of reusable graphs; fixed-shape sessions warm their own path automatically. A later unsupported shape may need a contiguous-weight copy for already packed parameters, so mixed-shape workloads should be measured separately. Create a new graph after a storage transition.

Three alternating paired rounds compare combined encode/decode against the preceding optimized runtime, with quantizer fusion on both sides:

| Batch / input per lane | Previous optimized | With matrices | Additional speedup |
| --- | ---: | ---: | ---: |
| 1 / 80 ms | 14.291 ms | 14.293 ms | 1.00× |
| 1 / 240 ms | 16.967 ms | 16.200 ms | 1.05× |
| 8 / 240 ms | 30.395 ms | 25.501 ms | 1.19× |
| 128 / 240 ms | 169.851 ms | 146.823 ms | 1.16× |

All tested codes, hidden states, waveforms, and restored-model results match original eager references. Peak allocated memory stays below 7.8 GB in this ablation. Packing, plan creation, graph capture, and restoration are excluded from steady-state timings; setup and memory are recorded separately. These are additional gains over the previous optimized runtime, not 100× results. [Paired ablation](results/full_matrix_runtime.json), [Triton corpus](results/full_fidelity_matrix_runtime.json), [CuTe corpus](results/full_fidelity_matrix_runtime_cute.json).

## Ordered Triton matrices

`matrix_backend="triton"` uses explicit FP32 SIMT kernels for 34 attention/FFN matrix shapes with 24, 32, 48, 64, 96 or 192 rows. The exact geometries, accumulation partitions and tiles are listed in [the runtime configuration](fast_moss/ordered_matrices.py). Other matrix shapes retain the supported cuBLASLt/native paths. It requires the same pinned model, GPU and library profile as `"cublaslt"`, plus Triton 3.4.0. Packing, fallbacks, hooks, workspace ownership and graph lifetime rules remain the same; no additional persistent weight copy is kept.

Each kernel accumulates consecutive groups of 64–512 terms, then adds partial results in order. The original two FFN shapes use 256 terms. Changing those boundaries changes FP32 results. Compiled main loops use FP32 FMA instructions and no matrix Tensor Core instructions. [Component stress and timing](results/ordered_confirm.json), [compiled kernels](results/ordered_kernel_resources.json).

Forty interleaved graph pairs sharing one packed-weight lifetime give these medians against the previous matrix/projection runtime:

| Batch / frames | Encoder, cuBLASLt → Triton | Decoder, cuBLASLt → Triton |
| --- | ---: | ---: |
| 8 / 3 | 13.224 → 12.920 ms | 11.593 → 11.506 ms |
| 24 / 1 | 14.079 → 13.783 ms | 12.651 → 12.546 ms |
| 1 / 24 | 13.078 → 12.779 ms | 11.413 → 11.338 ms |

These historical measurements describe the initial two-shape version. Its gains are modest and variable: the candidate wins 24–27 of 40 pairs, depending on direction/geometry. A separate three-round context-by-context comparison ranges from a 1.4% decoder regression to a 2.6% encoder gain. Warm component improvements of roughly 1.35–1.47× shrink to approximately parity after cache eviction. Input copies and owned outputs are timed; loading, packing and capture are excluded. [Interleaved pairs](results/full_ordered_paired.json), [context comparison and 27-case fidelity](results/full_ordered_runtime.json).

Expanded singleton-frame gates also exposed an older residual-layout bug. Both residual backends now allocate the canonical contiguous output used by native PyTorch, preserving downstream matrix dispatch. The regression previously changed encoder hidden states at batch 24 / one frame despite equal residual values, tokens and audio. The full-model corpus now includes batch 24 and 128 singleton inputs. [Isolation](results/ordered_singleton_audit.json), [CuTe corpus](results/full_fidelity_ordered_cute.json), [long incremental streams](results/full_incremental_ordered.json).

## Expanded attention and FFN matrices

The first expansion added thirteen exact matrix shapes, including the repeated attention input projection and several 768-channel transformer matrices. Compared with the preceding two-shape backend, with the same FFN/LFQ optimizations enabled:

| Batch / frames | Encoder, previous → expanded | Decoder, previous → expanded |
| --- | ---: | ---: |
| 8 / 3 | 13.167 → 11.520 ms (1.143×) | 11.601 → 9.958 ms (1.165×) |
| 24 / 1 | 13.992 → 12.336 ms (1.134×) | 12.663 → 11.073 ms (1.144×) |
| 1 / 24 | 12.973 → 11.364 ms (1.142×) | 11.454 → 9.819 ms (1.167×) |
| 1 / 3 | 8.698 → 8.622 ms (1.009×) | 7.269 → 7.207 ms (1.009×) |

These are three alternating rounds with independent, restored weight-packing lifetimes and five samples of twenty graph calls per backend/round. Copies and owned outputs are timed; loading, packing, capture and restoration are excluded. Peak allocation is below 7.72 GB. An initial shared-storage comparison penalized baseline fallbacks with new weight copies and is explicitly excluded from performance claims. [Corrected ablation](results/full_ordered_shapes_ablation.json), [27-case exactness and diagnostic paired run](results/full_ordered_shapes.json).

All **1,560 component comparisons**, **27 full-model cases**, the **15-case CuTe corpus**, and long streams with **11,072 tokens / 664,320 samples** are exact. The runtime retains the pinned FP32 environment, one persistent copy of each weight, hooks/fallbacks, per-stream ownership and graph invalidation. Three new shapes need no vendor algorithm; their ordered kernels use the same managed lifetime. [Components](results/ordered_shapes_confirm.json), [CuTe](results/full_fidelity_ordered_shapes_cute.json), [streams](results/full_incremental_ordered_shapes.json).

The profile now contains 192 ordered main loops per direction. Total kernels increase by 68 per direction because some replacements need a separate ordered reduction. Matrix-associated work still occupies approximately **78.58% encoder / 81.88% decoder** device kernel time, including fused decoder epilogues. The requested 100× whole-model result remains unproven. [Profile](results/full_ordered_shapes_profile.json), [compiled resources](results/ordered_shapes_resources.json).

## Five further native-matrix replacements

The backend now also covers `(M,N,K)=(24,1280,1280)`, `(48,3072,768)`, `(48,768,768)`, `(192,3072,768)` and `(192,2304,768)`. A wider partition search found exact 96-, 160- and 192-term accumulation groups that the earlier search omitted for native-only shapes. All **600 component comparisons** and **27 full-model cases** pass. [Native launches and search](results/ordered_remaining.json), [stress confirmation](results/ordered_remaining_confirm.json), [full-model fidelity](results/full_ordered_remaining.json).

Against the preceding fifteen-shape backend, independent restored contexts give batch-eight / three-frame encoder time **11.530 → 10.448 ms (1.104×)** and decoder time **9.943 → 8.895 ms (1.118×)**. Batch 24 / one frame improves **1.095× / 1.110×**; batch one / 24 frames improves **1.106× / 1.129×**. Batch one / three frames does not activate the new shapes and is effectively unchanged. These are three alternating rounds, each with five samples of twenty graph calls; copies and owned outputs are included, while loading, packing, capture and restoration are excluded. [Matched fifteen-versus-twenty-shape ablation](results/full_ordered_remaining_ablation.json).

The twenty-shape runtime also passes the **15-case CuTe corpus**, exact long streams with **11,072 tokens / 664,320 samples**, and **259 tests**. Compiled kernels retain FP32 FMA, zero spills and no matrix Tensor Core instructions. The latest profile attributes roughly **76.30% encode / 79.20% decode** device kernel time to matrix-associated work, now mainly the ordered kernels themselves. [CuTe fidelity](results/full_fidelity_ordered_remaining_cute.json), [streams](results/full_incremental_ordered_remaining.json), [compiled resources](results/ordered_remaining_resources.json), [profile](results/full_ordered_remaining_profile.json).

## Small-row native-layout matrices

The Triton matrix backend supports **44 native-layout shapes**: six one-row GEMVs and 38 multi-row matrices, including attention, FFN and inter-stage projections. For multi-row shapes, explicit row accumulators share weights across rows while preserving the same ordered FP32 arithmetic: reset sixteen lanes every 256 K elements, sum tiles within each lane, then reduce lanes serially. Four shapes use split-tile computation. If a larger call packs a weight, small calls retain the existing contiguous-weight fallback. [Current configurations](fast_moss/small_matrices.py).

The preceding twenty additions covered two-, four- and eight-row shapes. Against the preceding runtime, one-frame encode/decode improves **7.421 → 7.146 ms / 6.603 → 6.322 ms** at batch one (**1.039× / 1.044×**) and **10.962 → 10.449 ms / 9.739 → 9.242 ms** at batch eight (**1.049× / 1.054×**). Both Triton and CuTe combinations pass **39 full-model cases**. One- and eight-lane streams across 12.96 seconds preserve every output versus original eager streaming, including **41,472 tokens and 2,488,320 samples** in the eight-lane case. [Selection](results/short_selection.json), [integrated comparison](results/full_short_runtime.json), [CuTe](results/full_short_cute.json), [eight-lane stream](results/full_short_streaming_b8.json).

The earlier six-shape GEMV addition preserves full-K cyclic accumulation over 8, 16 or 32 lanes, an explicit halving reduction, and signed-zero rounding. Against the preceding runtime, one-frame encode improves **7.520 → 7.417 ms (1.014×)** and decode **6.686 → 6.598 ms (1.013×)** in independently restored contexts. All **672 component comparisons**, **27 full-model cases**, and **27 CuTe cases** pass; a 162-chunk stream preserves exact **5,184 tokens and 311,040 samples**. [Integrated GEMV comparison](results/full_gemv_runtime.json), [bit checks](results/gemv_confirm.json), [CuTe](results/full_gemv_cute.json), [one-frame long stream](results/full_gemv_streaming.json).

Against the preceding three-shape runtime, three rotating-order rounds with independently restored contexts give batch-one / three-frame encode **8.505 → 8.239 ms (1.032×)** and decode **7.063 → 6.794 ms (1.040×)**. Batch two / three frames improves roughly 0.4%; batch one / six frames roughly 0.5%. The one-frame control is unchanged. Timings include copies and owned outputs, with five samples of ten calls per round. [Integrated ablation and 27 exact cases](results/full_small_fixed_runtime.json), [960 component comparisons](results/small_fixed_confirm.json).

Selection includes both component and full-model evidence. A larger sixteen-shape candidate set offers no consistent advantage over the retained eleven replacements; the existing split kernel supplied the twelfth shape at that stage. Explicit CUDA vector-load probes also preserve exact arithmetic but do not displace the selected Triton implementation. [Broader model ablation](results/full_small_fixed.json), [CUDA probes](results/small_vector_cuda.json).

All **27 CuTe cases** also match exactly. A **12.96-second batch-one stream** exercises the new shapes across the cache boundary: all **5,184 tokens and 311,040 samples** match corrected original eager streaming, chunk by chunk. As with the original streaming implementation, its waveform can differ slightly from offline decoding. [CuTe](results/full_small_fixed_cute.json), [long stream](results/full_small_fixed_streaming.json). The current suite passes **432 tests**; all 44 compiled native-layout configurations have zero spills and no matrix Tensor Core instructions. [Resources](results/narrow_resources.json).

## Medium-row matrices and signed-zero fidelity

Twenty further configurations cover 16/32/64-row matrices. Against the preceding configuration tables, batch-eight / one-frame encode improves **10.444 → 9.240 ms (1.130×)** and decode **9.241 → 8.021 ms (1.152×)**. Batch-four / two-frame gains are **1.134× / 1.163×**; batch-one / eight-frame gains are **1.138× / 1.169×**. Both ablation arms share a newly corrected signed-zero policy. Full-model measurements favor all twenty additions even though some component timings regress after cache eviction. [Matched runtime ablation](results/full_mid_runtime.json), [selection](results/mid_selection.json).

Underflow tests exposed bit differences hidden by ordinary numerical equality. Ordered kernels now reproduce each native shape's final addition of zero and omit extra padded FMAs in two partial tails; the fused FFN residual projection applies the same correction before scaling. All **288 signed-zero checks**, **2,072 component stress comparisons**, and **39 full-model cases with each residual backend** pass. [Zero audit](results/mid_zero_fixed.json), [components](results/mid_confirm.json), [CuTe corpus](results/full_mid_cute.json).

Long-stream gates retain exact **41,472 tokens / 2,488,320 samples** in eight one-frame lanes and **5,120 tokens / 307,200 samples** in a single eight-frame lane, across the ten-second cache boundary. Incremental arrivals, pauses, tails and lane reuse also remain exact. These comparisons use original eager streaming at the same batch/chunk geometry. [Eight lanes](results/full_mid_streaming_b8.json), [eight-frame chunks](results/full_mid_streaming_f8.json), [incremental scheduler](results/full_incremental_mid.json).

At batch eight / one frame, small native-layout kernels now consume **52.19% / 59.93%** of encode/decode kernel time; all disjoint matrix groups together consume **76.34% / 77.81%**. Their scheduling and weight reuse are the next optimization target. All **34 ordered configurations** compile without spills or matrix Tensor Core instructions. [Profile](results/full_mid_profile.json), [ordered resources](results/mid_ordered_resources.json).

## Smallest matrix grids

Ten one-/two-row configurations now use narrower output tiles; one two-row matrix also moves from split computation to a single kernel. The integrated comparison measures batch-one / 80 ms encode **7.170 → 7.097 ms (1.0102×)** and decode **6.323 → 6.261 ms (1.0098×)** against the preceding runtime. Other measured geometries stay within 0.3% of parity. All **48 full-model cases** pass. [Runtime comparison](results/full_narrow_runtime.json), [selected configurations](results/narrow_selection.json).

The full 30-shape narrow-grid selection and its 25-shape cache-evicted subset regress some codec geometries despite promising component timings. An explicit Gluon row-layout probe also preserves exact outputs but provides no consistent codec advantage. Those experiments remain outside runtime dispatch. [Broad comparison](results/full_narrow_rows.json), [group isolation](results/full_narrow_groups.json), [3,528 narrow-grid checks](results/narrow_confirm.json), [750 explicit-layout checks](results/small_layout_confirm.json).

Both supported residual backends pass the **48-case** corpus. Two 12.96-second streams, with one and two lanes, match original eager streaming chunk by chunk. The complete suite passes **432 tests**; all 44 small-matrix configurations compile without spills or matrix Tensor Core instructions. The batch-one profile has **12 fewer launches per direction**, while GEMV still consumes **45.26% / 51.24%** of encode/decode kernel time. [CuTe](results/full_narrow_cute.json), [one lane](results/full_narrow_streaming_b1.json), [two lanes](results/full_narrow_streaming_b2.json), [profile](results/full_narrow_profile.json).

## FFN pipeline and epilogues

Add `ffn_backend="triton"` to `optimized(...)` with `matrix_backend="triton"` or `matrix_backend="cuda"` and either residual backend enabled. At the two supported 24-row FFN shapes, it uses two pipeline stages, halving main-loop shared memory to 20,480 bytes. Decoder reductions also compute GELU or scaled residual addition with explicit FP32 rounding; 24-row encoder FFNs retain their existing epilogues.

For one-row 1280↔5120 FFNs, both encoder and decoder fuse GELU and scaled residual addition into the native GEMV kernels. This keeps native FP32 weight storage and adds no weight copy or persistent workspace. Unsupported shapes, custom activations/norms, and observer hooks retain the appropriate existing module paths; packing and graph lifetime rules still apply.

With `matrix_backend="cuda"`, the same FFN option now also fuses sixteen matrix shapes: 768-wide FFNs with 2, 4, 6, 8, 12 or 16 rows, and 1280-wide FFNs with 3 or 8 rows. It retains native FP32 weights and the existing accumulation order. Paired batch-one measurements show about **0.4% lower encode / 1.0% lower decode latency at 80 ms**, and **1.8% lower decode latency at 240 ms**; the 240 ms encoder makes no new fused calls, so its 0.2% timing difference is a control, not an optimization gain. All 48 integrated cases are bit-identical. [Short-row ablation](results/full_short_ffn_runtime.json), [math and schedule investigation](docs/research.md#short-row-ffn-epilogues-on-native-matrix-schedules).

The same option also preserves dense transposed BTC residual/output layouts in eight contraction shapes, extending fusion to previously skipped encoder blocks. Paired batch-one measurements reduce encoder latency by **1.1% at 80 ms / 2.7% at 240 ms**, removing **72 / 168 launches**. All 48 integrated cases match bit for bit; the decoder makes no new calls. [Strided encoder ablation](results/full_strided_ffn_runtime.json), [layout investigation](docs/research.md#preserving-encoder-layouts-through-ffn-fusion).

With `norm_backend="cuda"`, `matrix_backend="cuda"` and the FFN option, one-row LayerNorm now fuses into GELU projections; `attention_mask_backend="triton"` also enables QKV fusion. Native FP32 weights and normalization/projection rounding stay unchanged. Seven warmed paired rounds measure **0.36% / 0.44% lower one-frame encode/decode latency** versus the preceding runtime, with no-call controls within 0.22% of parity. This removes **64 launches per direction**. Both 48-case residual-backend gates and two 12.96-second streams pass exact checks. [Integrated ablation](results/full_norm_projection_runtime.json), [CuTe](results/full_norm_projection_cute.json), [streaming](results/full_norm_projection_streaming_b1.json), [implementation and research](docs/research.md#supported-one-row-normalizationprojection-fusion).

With CUDA matrices, the FFN option and Triton attention masks, attention output projections now fuse scaled residual addition for **nine native matrix shapes**, preserving the existing normalization/QKV path. Seven warmed paired rounds measure **1.39% lower encode / 0.84% lower decode latency at 80 ms**, and **1.75% / 0.49% at 240 ms**, at batch one. One-frame batch-eight decode is **0.50% lower**; the batch-eight three-frame no-call control is within 0.11% of parity. Both 48-case backend gates and two long streams remain bit-identical, and all **1,010 tests** pass. This removes **92 / 68 encoder/decoder launches** at 80 ms. [Integrated ablation](results/full_attention_residual_runtime.json), [implementation and validation](docs/research.md#supported-attention-projectionresidual-fusion).

A research shared-memory weight pipeline preserves native FP32 arithmetic and shows about **0.39% lower encode / 0.47% lower decode latency** in forty interleaved one-frame graph pairs. Both 48-case backend gates and long streams are exact; runtime integration remains pending. [Asynchronous staging evidence](docs/research.md#asynchronous-shared-weight-staging-for-one-row-projections).

The one-row addition removes **64 launches per encode/decode** and improves latency **1.0088× / 1.0098×** against the preceding optimized runtime. Both 48-case residual-backend gates and two 12.96-second streams remain exact against their original references. These incremental gains are separate from the fresh upstream comparison above. [Runtime ablation](results/full_gemv_epilogue_runtime.json), [CuTe gate](results/full_gemv_epilogue_cute.json), [profile](results/full_gemv_epilogue_profile.json), [one-lane stream](results/full_gemv_epilogue_streaming_b1.json), [two-lane stream](results/full_gemv_epilogue_streaming_b2.json).

This option requires `nvidia-cuda-nvcc-cu12==12.8.93`, included in `requirements.lock` and the package's `ffn` extra. Its CUDA math library is checked by version and SHA-256 before model mutation and passed only to these fused kernels. Triton's bundled math library produces different GELU values on the pinned environment; replacing it globally is unnecessary. [Math isolation](results/gelu_math.json), [compiled runtime kernels](results/ffn_kernel_resources.json).

At batch eight / three frames, eight rotating rounds of 20 graph calls give encoder medians **13.185 → 13.137 ms** and decoder **11.709 → 11.613 ms** versus the preceding ordered-matrix runtime. A separate 40-pair single-call comparison finds no encoder advantage at this geometry. Effects are small and variable; decoder fusion removes 64 kernel launches, but does not establish 100× acceleration. [Five-way ablation](results/full_ffn_ablation.json), [paired samples and 27 exact full-model cases](results/full_ffn_runtime.json), [profile](results/full_ffn_profile.json).

The CuTe residual combination passes all 15 corpus cases, and long incremental streams retain exact **11,072 tokens and 664,320 samples**. The full suite now passes **432 tests**. [CuTe fidelity](results/full_fidelity_ffn_cute.json), [streaming fidelity](results/full_incremental_ffn.json), [tests](results/tests.txt).

## LFQ projection and decoder reconstruction

Add `projection_backend="triton"` to `optimized()` to accelerate the 32 LFQ output projections and decoder codebook reconstruction. It requires cached convolution weights, the original LFQ geometry, RTX 5070 Ti with 70 SMs, PyTorch 2.8.0+cu128, and cuDNN 9.10.2. cuDNN must be enabled with TF32 and benchmark mode disabled. The option is independent of the matrix backend.

The decoder caches every codebook entry after its original FP32 output convolution, using **64 MiB** of additional resident storage. One Triton gather adds the selected vectors in the original codebook order; the final output projection remains unchanged. With `quantizer_backend="triton"`, the encoder also fuses each eight-channel projection with its masked residual update. It retains the actual straight-through values and does not substitute decoder lookup values. Input projections and nearest-code decisions keep their existing arithmetic.

Local/global hooks retain the observable module path. Autocast, unsupported singleton strides, and changed cuDNN execution settings retain native paths. Cache construction bypasses projection hooks, and context exit restores methods and removes cached tensors. Entry/exit synchronize the device and invalidate all managed graphs on it, including unrelated graphs, to prevent replay through freed cache pointers. Keep weights unchanged and graphs/sessions inside the optimization context; raw CUDA graphs must not outlive it. Invalid used code indices retain asynchronous CUDA assertion behavior; unused extra codebooks retain the original ignore behavior.

In three paired rounds against the supported matrix/quantizer runtime, batch-one / 240 ms encoder time improves **8.694 → 8.571 ms**, decoder **7.381 → 7.159 ms**, and decoder quantizer reconstruction **0.253 → 0.059 ms (4.29×)**. At batch eight / 240 ms, encode/decode gains are **1.022× / 1.026×**. Across five geometries, reconstruction improves **3.14–6.16×**, while whole-model improvements remain much smaller. Cache construction and graph setup are outside the timing loop; input copies and owned outputs are included. [Paired evidence](results/full_projection_runtime.json).

All **864 learned-projection component checks**, both **13-case full-model corpora**, and the long incremental stream gate remain exact on recorded inputs. The component checks include all entries of all 32 codebooks; they do not establish numerical equivalence for untested library/device configurations. [Components](results/projection_components.json), [Triton corpus](results/full_fidelity_projections.json), [CuTe corpus](results/full_fidelity_projections_cute.json), [incremental streams](results/full_incremental_projections.json).

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

## Lossless weight-storage experiments

Logical call attribution finds about **3.55 GB of module weights per encode/decode**, including **2.52 GB in GEMV** for batch-one 80 ms input. These are logical bytes, not measured DRAM traffic. [Attribution](results/weight_traffic.json).

Research-only FP32 block encodings save **11–12% of weight storage** while preserving every original bit. Direct Triton reconstruction and cooperative CUDA decoding pass **2,352 component comparisons**, but all tested variants are slower than the current GEMV kernels. They remain outside runtime dispatch; no additional codec speedup or model-memory saving is claimed. [Variable-width probe](results/packed_fp32_probe.json), [fixed-block probe](results/packed_fp32_fixed28.json), [cooperative CUDA probe](results/packed_fp32_cooperative.json), [13 regression tests](results/packed_fp32_tests.txt).

A separate CUDA vector-load/prefetch search passes **336 captured configurations**, **660 finalist stress comparisons**, and **48 full-model cases**. Its best projection gains 5% in warm component timing but makes one-frame codec latency about 0.5% slower. It also remains research-only, with no runtime speedup claimed. [Component confirmation](results/gemv_vector_confirm.json), [generated instructions](results/gemv_vector_instructions.json), [codec ablation](results/full_gemv_vector.json).

Interleaving exact FP32 weights gives **17–18% warm component gains**, but only **0.6–1.0% encode / 0.7% decode** in full-model ablations, while adding **4.6 GB** of packed storage. It remains research-only. Distinct-allocation rings show why repeated access to one weight overstates these gains. Both 48-case model gates pass; that checkpoint passed **473 tests**. [Warm selection](results/full_gemv_interleaved.json), [cold selection](results/full_gemv_interleaved_cold.json), [weight rings](results/gemv_weight_ring.json).

## Fidelity and benchmark scope

`benchmarks.compare` records exact equality, mismatching element counts, maximum error, and RMSE, and exits with failure on any mismatch. It compares equal batches, audio lengths, FP32 arithmetic, and quantizer counts. Wall-clock and CUDA-event samples include graph input copies/output ownership; amortized kernel-only measurements are labeled separately. Synthetic inputs and small structural tests are development gates, not proof of quality across real speech, music, or sound effects.

A separate split-cache FP32 attention prototype improves filled-ring streaming by 1.24–1.26× encode and about 1.29× decode relative to the already optimized runtime. All 30,720 tokens match across three 12.8-second cases, but waveform rounding changes (maximum 8.51e-5). It remains under `benchmarks/` and is not part of the supported runtime or the speedup table above. [Experimental evidence](results/full_attention_experiment.json).

A cuBLASLt pedantic-FP32 prototype selects measured algorithms and transposed FP32 weight layouts. Its new single-copy mode retains the packed storage and restores the original layout on exit. At batch eight / 240 ms, combined encode/decode falls from **30.916 to 26.019 ms (1.19×)** against the already optimized graph, with peak allocated memory around **7.46 GB** instead of the earlier duplicated-weight prototype's 13.25 GB. Batch 128 improves **169.831 → 146.979 ms (1.16×)** with peak allocation **7.63 GB**. All 18 cases in the batch sweep retain exact tokens, hidden states, audio, and restored-model outputs. [Batch evidence](results/full_cublaslt_batching.json).

The eight-lane 12.96-second stream with pauses, completion, and reuse also remains exact in single-copy mode, with peak allocated memory **8.74 GB**. [Streaming evidence](results/full_cublaslt_resident_streaming.json). The broader tuning passes [684 matrix stress checks](results/cublaslt_large_fidelity.json). These historical research tools remain under `benchmarks/`. The supported `matrix_backend="cublaslt"` option described above packages the retained tuning with stream-specific workspaces, graph invalidation, and stronger lifetime checks. Its current measurements include quantizer fusion and are recorded separately; the earlier upstream-comparison reports remain historical matched measurements.

A later search screened 12,422 vendor configurations. Its selected alternatives preserve tested outputs but change whole-codec latency by less than 1% against the previous prototype, so they have not replaced the default tuning. [Matched comparison](results/full_cublaslt_config_ablation.json), [fresh component confirmation](results/cublaslt_config_confirmation.json).

Multi-pass tensor-core experiments keep FP32 weight storage but change arithmetic. A TF32x3 QKV probe changes 24 tokens in the near-tie speech case and produces up to 0.152 waveform error, despite small hidden-state errors. It is rejected as an exact replacement and remains outside the runtime. [Full probe](results/full_tensorcore_probe.json), [first changed decision](results/tensorcore_quantizer_tie.json).

Lossless weight-storage experiments preserve every FP32 bit but have not produced a runtime improvement. Block exponent packing reduces stored bytes by roughly 12%, while separate reconstruction plus GEMM is 1.24–2.50× slower under the eviction protocol. Verified CUDA compressible allocations help the zero-matrix control but give no material benefit on the tested checkpoint matrices. Both remain research-only. [Software codec measurements](results/lossless_shapes.json), [hardware allocation comparison](results/compressible_shapes.json).

Outstanding: broader audio corpus coverage, more streaming durations/batch schedules, network serving, multi-GPU execution, matrix-kernel optimization, and a defensible matched-workload 100× result. See [research notes](docs/research.md) and [work log](docs/progress.md).

## Exact CUDA LayerNorm

Add `norm_backend="cuda"` to `optimized(...)` to use the validated CUDA kernels for 20 contiguous FP32 LayerNorm geometries and 18 dense channel/time transposes. Install the pinned `normalization` extra (`uv sync --extra normalization`), or use `requirements.lock`. This backend requires the recorded RTX 5070 Ti, PyTorch and checkpoint environment. It combines with the existing FFN and either residual backend.

The runtime retains native execution for unsupported shapes/layouts, changed epsilon, gradients and autocast. It restores module methods on exit and invalidates managed graphs on entry/exit. Construct graphs and streaming sessions inside the optimization context; warm each shape on its capture stream first (`GraphedCallable` does this automatically).

The 48-case runtime ablation is bit-identical. At batch one / 80 ms, it improves encode **7.035 → 6.940 ms** and decode **6.210 → 6.078 ms**, about **1.4% / 2.2%**. At 240 ms, encoder timing is effectively unchanged and decode improves about 2%. These are isolated comparisons against the preceding optimized runtime, not whole-model speedup claims. [Runtime results](results/full_native_layer_norm_runtime.json), [arithmetic and compiler investigation](docs/research.md#native-welford-layernorm-with-constant-counts).

The CUDA backend now also reads validated transposed encoder inputs directly, eliminating their separate contiguous-copy launch. The isolated integrated comparison improves encoding by **1.1–2.6%** across batch-one/eight and 80/240 ms inputs; decoder timing stays within 0.09% of parity. All 48 full-model cases remain bit-identical. [Strided normalization ablation](results/full_strided_layer_norm_runtime.json), [memory-layout investigation](docs/research.md#eliminate-encoder-normalization-copies).

## Exact CUDA small-row matrices

Add `matrix_backend="cuda"` to `optimized(...)` to enable eleven measured small-row matrix replacements alongside the existing Triton matrix paths. Install the pinned `cuda` extra (`uv sync --extra cuda`), or use `requirements.lock`. The same checkpoint, GPU and library profile is required. This option combines with `ffn_backend="triton"`, `norm_backend="cuda"` and either residual backend.

The new kernels keep the native FP32 accumulation order and contiguous weights. K partitions communicate through shared memory within one launch; no global partial-results buffer or persistent weight copy is added by these kernels. Other inherited matrix paths retain their existing packing and lifetime rules.

The initial five-shape extension passes its 48-case corpus bit for bit. Its paired batch-one graph measurements improve encode/decode latency by about **0.4% at 80 ms** and **0.7% at 240 ms**; the batch-eight control stays within 0.11% of parity. These modest whole-codec gains are separate from the larger component timings. [Integrated ablation](results/full_cta_tiled_runtime.json), [implementation and selection evidence](docs/research.md#single-block-native-order-matrix-partitions).

Six additional long-K shapes now use shared partials, covering all four small-matrix shapes that previously needed a separate reduction launch. The integrated 48-case comparison preserves all output bits and lowers batch-one graph latency by about **0.8% at 80 ms**, and **2.4% encode / 2.9% decode at 240 ms**. Batch-eight measurements remain within 0.09% of parity. The existing Triton backend remains available. [Integrated long-K ablation](results/full_wide_cta_runtime.json), [search and repeated selection](docs/research.md#long-k-matrices-without-global-partial-buffers).
