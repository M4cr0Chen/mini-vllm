"""Paged KV cache + paged attention for MLX.

We don't write a custom Metal kernel. Instead:
  * K/V live in fixed-size blocks in a shared pool (num_slots, n_kv, head_dim).
  * Each step, new K/V are *scattered* into their slots (pool[slots] = kv).
  * For decode, each sequence's history is *gathered* from its blocks into a
    padded (B, n_kv, L_max, head_dim) tensor, then `mx.fast.scaled_dot_product_
    attention` runs with an additive padding mask.
  * Per-sequence positions are handled by `mx.fast.rope`'s array-valued offset.

`PagedAttention` reuses mlx-lm's projection/norm/rope submodules unchanged, so
weights (including quantized ones) load through mlx-lm; only the attention math
and cache interaction are ours.
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
    slot_mapping: mx.array               # (num_new_tokens,) pool slots to write
    gather_index: Optional[mx.array] = None   # (B, L_max) decode only
    mask: Any = None                     # "causal" (prefill) or additive (B,1,1,L_max)


class LayerKVCache:
    """One layer's view into the paged pools + the shared BatchMeta."""

    def __init__(self, pool_k: mx.array, pool_v: mx.array, owner: "PagedKVCache"):
        self.pool_k = pool_k
        self.pool_v = pool_v
        self.owner = owner

    def write(self, keys: mx.array, values: mx.array) -> None:
        # keys/values: (B, n_kv, T, head_dim) -> flatten to (B*T, n_kv, head_dim)
        B, H, T, D = keys.shape
        slots = self.owner.meta.slot_mapping
        self.pool_k[slots] = keys.transpose(0, 2, 1, 3).reshape(B * T, H, D)
        self.pool_v[slots] = values.transpose(0, 2, 1, 3).reshape(B * T, H, D)

    def gather(self) -> tuple[mx.array, mx.array]:
        idx = self.owner.meta.gather_index          # (B, L_max)
        B, Lmax = idx.shape
        H, D = self.pool_k.shape[1], self.pool_k.shape[2]
        flat = idx.reshape(-1)
        K = self.pool_k[flat].reshape(B, Lmax, H, D).transpose(0, 2, 1, 3)
        V = self.pool_v[flat].reshape(B, Lmax, H, D).transpose(0, 2, 1, 3)
        return K, V


class PagedKVCache:
    """Owns the K/V block pools for every layer and the current BatchMeta."""

    def __init__(self, n_layers: int, num_slots: int, n_kv: int, head_dim: int, dtype):
        self.meta: Optional[BatchMeta] = None
        self.pools_k = [mx.zeros((num_slots, n_kv, head_dim), dtype=dtype) for _ in range(n_layers)]
        self.pools_v = [mx.zeros((num_slots, n_kv, head_dim), dtype=dtype) for _ in range(n_layers)]
        self.layers = [LayerKVCache(self.pools_k[i], self.pools_v[i], self) for i in range(n_layers)]

    def set_meta(self, meta: BatchMeta) -> None:
        self.meta = meta


class PagedAttention(nn.Module):
    """Drop-in replacement for mlx-lm's Attention that reads/writes the paged
    cache and supports ragged batches via gather + masked SDPA."""

    def __init__(self, orig: nn.Module, n_heads: int, n_kv: int, head_dim: int):
        super().__init__()
        self.q_proj = orig.q_proj
        self.k_proj = orig.k_proj
        self.v_proj = orig.v_proj
        self.o_proj = orig.o_proj
        self.q_norm = orig.q_norm
        self.k_norm = orig.k_norm
        self.rope = orig.rope
        self.scale = orig.scale
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim

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
            # Fresh single sequence: its own K/V are the full context.
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        else:
            K, V = cache.gather()
            out = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.scale, mask=meta.mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


def patch_attention(model: nn.Module, n_heads: int, n_kv: int, head_dim: int) -> None:
    """Replace every layer's self-attention with PagedAttention in place."""
    for layer in model.layers:
        layer.self_attn = PagedAttention(layer.self_attn, n_heads, n_kv, head_dim)
