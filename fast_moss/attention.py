# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted self-attention forward: fused/shared causal bias; original SDPA arithmetic.
"""Fused causal-mask construction; attention arithmetic stays in upstream SDPA."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mask(OFFSET, POS, OUT, T: tl.constexpr, K: tl.constexpr,
          PS0: tl.constexpr, PS1: tl.constexpr, CONTEXT: tl.constexpr,
          PAD: tl.constexpr, ADDITIVE: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (i < T * PAD) & (i % PAD < K)
    pos = tl.load(POS + b * PS0 + (i % PAD) * PS1, valid, -1)
    offset = tl.load(OFFSET + b)
    delta = offset + i // PAD - pos
    allowed = (pos >= 0) & (delta >= 0)
    if CONTEXT is not None:
        allowed = allowed & (delta < CONTEXT)
    if ADDITIVE:
        # Matches PyTorch's bool-to-additive conversion and its zero padding.
        result = tl.where(allowed | ~valid, 0.0, -float("inf"))
    else:
        result = allowed
    tl.store(OUT + b * T * PAD + i, result, i < T * PAD)


def causal_mask(offset, positions, length, context, additive=False):
    b, k = positions.shape
    if (offset.shape != (b,) or offset.dtype != torch.int64
            or positions.dtype != torch.int64 or not offset.is_contiguous()
            or not offset.is_cuda or offset.device != positions.device or length < 1 or k < 1):
        raise ValueError("Expected CUDA int64 lane offsets and key positions")
    pad = triton.cdiv(k, 8) * 8 if additive else k
    out = torch.empty((b, 1, length, pad), device=offset.device,
                      dtype=torch.float32 if additive else torch.bool)
    _mask[(b, triton.cdiv(length * pad, 256))](
        offset, positions, out, length, k, *positions.stride(), context, pad, additive, 256)
    return out[..., :k]


def forward(self, query, key, value, *, _fast_projected=None, _fast_epilogue=None):
    # The upstream self-attention also projects query for Q, K, and V.
    if not query.is_cuda or query.dtype != torch.float32:
        return self._fast_original_attention(query, key, value)
    state = self._streaming_state
    b, t = query.shape[:2]
    offset = (torch.zeros(b, device=query.device, dtype=torch.long)
              if state is None else state.offset)
    projected = self.in_projs[0](query) if _fast_projected is None else _fast_projected
    projected = projected.reshape(b, t, 3, self.num_heads, self.embed_dim // self.num_heads)
    projected = projected.permute(2, 0, 3, 1, 4)
    q, k, v = projected[0], projected[1], projected[2]
    if self.rope:
        q, k = self.rope(q, k, offset, time_before_heads=False)
    k, v, positions = self._complete_kv(k, v)
    bias = None
    if self.causal:
        pool = self._fast_mask_pool
        # Pool lifetime is exactly one synchronized stage invocation. Context
        # and key length are included in case a layer has distinct geometry.
        pool_key = (b, t, positions.shape[1], self.context, query.device)
        if pool is not None and pool_key in pool:
            bias = pool[pool_key]
        else:
            bias = causal_mask(offset, positions, t, self.context, additive=True)
            if pool is not None:
                pool[pool_key] = bias
    x = F.scaled_dot_product_attention(q, k, v, bias, dropout_p=0.0)
    x = x.transpose(1, 2).reshape(b, t, self.embed_dim)
    x = (self.out_projs[0](x) if _fast_epilogue is None
         else self.out_projs[0](x,_fast_epilogue=_fast_epilogue))
    if state is not None:
        state.offset[:] = torch.where(state.exec_mask, state.offset + t, state.offset)
        state.offset_cpu += t
    return x
