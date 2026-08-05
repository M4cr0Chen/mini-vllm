"""Paged KV cache + paged attention for MLX, with an incremental decode buffer.

Storage stays paged: K/V for every token live in fixed-size blocks in a shared
pool, written by scatter. That keeps the block structure for prefix sharing.

Decode: re-gathering the whole history from the pool every step is ~10x more
expensive than reading contiguous memory, so for the running batch we keep a
contiguous, right-aligned decode buffer per layer that we append one column to
each step; attention reads a slice of it. The buffer is rebuilt from the pool
(one gather) only when the batch composition changes — see `ModelRunner._rebuild`.

Prefill: normally a fresh sequence attends causally to itself. With prefix
caching, a sequence reuses cached prefix blocks and prefills only its suffix;
that suffix attends to the cached prefix gathered from the pool (`gather_prefix`)
plus itself (offset-causal mask) — "prefill with history".

`PagedAttention` reuses mlx-lm's projection/norm/rope submodules unchanged, so
weights (incl. quantized) load through mlx-lm; only the attention math is ours.
"""
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class BatchMeta:
    """Per-forward-step metadata, shared across all layers."""

    is_prefill: bool
    rope_offset: Any                     # int (prefill) or (B,) array (decode)
    pool_slots: mx.array                 # (num_new_tokens,) pool slots to write
    prefix_slots: Optional[mx.array] = None   # prefill-with-history: cached prefix slots
    write_col: int = 0                   # decode: buffer column for the new token
    attn_len: int = 0                    # decode: columns to attend after writing
    mask: Any = None                     # "causal", or additive mask


class LayerKVCache:
    """One layer's handle. Reads its pool + decode buffer from the owner by index
    so the runner can swap buffers on a rebuild without stale references."""

    def __init__(self, owner: "PagedKVCache", idx: int):
        self.owner = owner
        self.idx = idx

    def write(self, keys: mx.array, values: mx.array) -> None:
        m = self.owner.meta
        o, i = self.owner, self.idx
        if m.is_prefill:
            # (B, n_kv, T, D) -> (B*T, n_kv, D) scattered into the pool
            B, H, T, D = keys.shape
            o.pools_k[i][m.pool_slots] = keys.transpose(0, 2, 1, 3).reshape(B * T, H, D)
            o.pools_v[i][m.pool_slots] = values.transpose(0, 2, 1, 3).reshape(B * T, H, D)
        else:
            B, H, _, D = keys.shape       # T == 1
            # pool: source of truth for rebuilds (and prefix sharing)
            o.pools_k[i][m.pool_slots] = keys.reshape(B, H, D)
            o.pools_v[i][m.pool_slots] = values.reshape(B, H, D)
            # decode buffer: append the new token as one contiguous column
            c = m.write_col
            o.buf_k[i][:, :, c:c + 1, :] = keys
            o.buf_v[i][:, :, c:c + 1, :] = values

    def attn_kv(self) -> tuple[mx.array, mx.array]:
        m, o, i = self.owner.meta, self.owner, self.idx
        return o.buf_k[i][:, :, :m.attn_len, :], o.buf_v[i][:, :, :m.attn_len, :]

    def gather_prefix(self) -> tuple[mx.array, mx.array]:
        """Gather cached-prefix K/V from the pool for prefill-with-history (B=1)."""
        slots, o, i = self.owner.meta.prefix_slots, self.owner, self.idx
        n, H, D = slots.shape[0], o.pools_k[i].shape[1], o.pools_k[i].shape[2]
        K = o.pools_k[i][slots].reshape(1, n, H, D).transpose(0, 2, 1, 3)
        V = o.pools_v[i][slots].reshape(1, n, H, D).transpose(0, 2, 1, 3)
        return K, V


class PagedKVCache:
    """Owns the paged pools for every layer, the contiguous decode buffers (set
    by the runner on rebuild), and the current BatchMeta."""

    def __init__(self, n_layers: int, num_slots: int, n_kv: int, head_dim: int, dtype):
        self.meta: Optional[BatchMeta] = None
        self.pools_k = [mx.zeros((num_slots, n_kv, head_dim), dtype=dtype) for _ in range(n_layers)]
        self.pools_v = [mx.zeros((num_slots, n_kv, head_dim), dtype=dtype) for _ in range(n_layers)]
        self.buf_k: list[Optional[mx.array]] = [None] * n_layers
        self.buf_v: list[Optional[mx.array]] = [None] * n_layers
        self.layers = [LayerKVCache(self, i) for i in range(n_layers)]

    def set_meta(self, meta: BatchMeta) -> None:
        self.meta = meta


class PagedAttention(nn.Module):
    """Drop-in replacement for mlx-lm's Attention using the paged cache."""

    def __init__(self, orig: nn.Module, n_heads: int, n_kv: int, head_dim: int):
        super().__init__()
        self.q_proj, self.k_proj = orig.q_proj, orig.k_proj
        self.v_proj, self.o_proj = orig.v_proj, orig.o_proj
        self.q_norm, self.k_norm = orig.q_norm, orig.k_norm
        self.rope = orig.rope
        self.scale = orig.scale
        self.n_heads, self.n_kv, self.head_dim = n_heads, n_kv, head_dim

    def __call__(self, x: mx.array, mask, cache: LayerKVCache) -> mx.array:
        B, L, _ = x.shape
        meta = cache.owner.meta

        q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q, offset=meta.rope_offset)
        k = self.rope(k, offset=meta.rope_offset)
        cache.write(k, v)

        if meta.is_prefill:
            if meta.prefix_slots is not None:
                Kp, Vp = cache.gather_prefix()               # cached prefix from the pool
                K = mx.concatenate([Kp, k], axis=2)
                V = mx.concatenate([Vp, v], axis=2)
                out = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.scale, mask=meta.mask)
            else:
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        else:
            K, V = cache.attn_kv()
            out = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.scale, mask=meta.mask)

        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


def patch_attention(model: nn.Module, n_heads: int, n_kv: int, head_dim: int) -> None:
    """Replace every layer's self-attention with PagedAttention in place."""
    for layer in model.layers:
        layer.self_attn = PagedAttention(layer.self_attn, n_heads, n_kv, head_dim)
