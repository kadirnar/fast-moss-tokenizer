# Research and optimization decisions

Sources inspected on 2026-09-20:

- [Original MOSS model card](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer): identifies the requested 24 kHz checkpoint and streaming API constraints. Do not substitute the newer v2 model. Local pinned source is authoritative for exact arithmetic and cache behavior.
- [MOSS paper](https://arxiv.org/abs/2602.10934): architecture/training background; its claims do not establish speedups for this implementation.
- [PyTorch CUDA graphs article](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) and [CUDA semantics documentation](https://docs.pytorch.org/docs/main/notes/cuda.html): motivate eliminating repeated host launch overhead while retaining stable graph tensor addresses. Capture setup is measured separately; output clones prevent replay aliasing.
- [Triton layer normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html): reduction and fusion reference. A new reduction order can change codec decisions; normalization fusion is not yet promoted.
- [CuTe DSL documentation](https://docs.nvidia.com/cutlass/4.3.2/media/docs/pythonDSL/cute_dsl.html) and [NVIDIA elementwise tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/ampere/tutorial/elementwise_add_autotune.py): launch/framework integration reference. Implementation uses installed CuTe 4.7.1 API. SM100-only tensor instructions are inappropriate for this GeForce SM120 GPU.

The residual kernel uses two rounded FP32 operations, rather than FMA, to match upstream multiply then add. Triton sets `enable_fp_fusion=False`; CuTe emits `mul.rn.f32` followed by `add.rn.f32`. Cancellation tests exercise this difference.

Initial reduced-model profiling identifies 66 weight-normalization calls per encode, accounting for 18.3% of recorded device operator time in that fixture. Its quantizer dimensions/depth are retained, but transformer depth/width are reduced: percentages must not be extrapolated to the full checkpoint. Frozen weight materialization and codebook normalization are now cached without changing operations or parameter tensors.

Future investigations: repeated RoPE tables/masks across transformer layers; fused ring-cache updates; strict-FP32 quantizer nearest-neighbor kernels preserving near-tie decisions; the many small 1x1 quantizer convolutions; GEMM shapes and bandwidth limits of 6+ GB FP32 parameters. Measure each change in isolation, reject fidelity failures, and retain regressions in the evidence log.

Confirmed streaming defect: upstream `RingKVCache.complete` writes a whole chunk before attention into a ring of `context` entries. With multiple local tokens per chunk, this overwrites history needed by early queries once the ring fills. A uniform-attention regression demonstrates an incorrect average of 2.5 instead of 2.0. Sessions now allocate `context + chunk_tokens - 1` entries at each transformer resolution, retaining the unchanged context mask. Tests verify both the analytic counterexample and capacity throughout encoder/decoder stages. Full-checkpoint long-duration comparisons against offline causal attention remain required; matching corrected eager streaming alone is insufficient.
