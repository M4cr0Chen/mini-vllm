import mlx.core as mx
import numpy as np
from mlx_lm import load

from .config import Config
from .paged_attention import BatchMeta, PagedKVCache, patch_attention


class ModelRunner:
    """Owns the MLX model + tokenizer, the paged KV cache, and the forward
    passes. Reuses mlx-lm's model (weights, quantized projections, RoPE) but
    swaps in our PagedAttention.

    Decode keeps a contiguous per-batch buffer that we append to each step; it's
    rebuilt from the pool only when the running set changes (see `_rebuild`).
    """

    def __init__(self, config: Config):
        self.config = config
        self.model, self.tokenizer = load(config.model)

        args = self.model.args
        self.n_heads = args.num_attention_heads
        self.n_kv = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.n_layers = len(self.model.layers)
        self.block_size = config.block_size
        self.dtype = self.model.model.norm.weight.dtype

        eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if not eos_ids:
            eos_ids = [self.tokenizer.eos_token_id]
        self.eos_ids = {int(t) for t in eos_ids if t is not None}

        patch_attention(self.model, self.n_heads, self.n_kv, self.head_dim)
        self.num_blocks = self._num_blocks_for_budget()
        self.cache = PagedKVCache(
            self.n_layers, self.num_blocks * self.block_size,
            self.n_kv, self.head_dim, self.dtype,
        )
        # decode-buffer state
        self._last_ids: tuple = ()
        self.col = 0
        self.left_pad = np.zeros(0, dtype=np.int64)

    def _num_blocks_for_budget(self) -> int:
        itemsize = mx.zeros(1, dtype=self.dtype).nbytes
        per_token = 2 * self.n_layers * self.n_kv * self.head_dim * itemsize  # K + V
        budget = int(self.config.kv_cache_memory_gb * (1024 ** 3))
        return max(1, budget // (per_token * self.block_size))

    def _slots(self, block_table, positions: np.ndarray) -> np.ndarray:
        bt = np.asarray(block_table, dtype=np.int64)
        return bt[positions // self.block_size] * self.block_size + positions % self.block_size

    def prefill(self, seq) -> mx.array:
        """Prefill a sequence's uncached suffix. With a reused prefix it attends
        to the cached prefix (gathered from the pool) plus itself. Returns (1, V)."""
        ids = seq.token_ids
        L = len(ids)
        start = seq.prefill_start                              # tokens reused from the cache
        suffix_slots = self._slots(seq.block_table, np.arange(start, L))
        if start > 0:
            prefix_slots = self._slots(seq.block_table, np.arange(start))
            qpos = np.arange(start, L)[:, None]               # suffix query positions
            kpos = np.arange(L)[None, :]                      # all key positions
            mask = np.where(kpos <= qpos, 0.0, -1e9).reshape(1, 1, L - start, L)
            meta = BatchMeta(
                is_prefill=True, rope_offset=start,
                pool_slots=mx.array(suffix_slots.astype(np.int32)),
                prefix_slots=mx.array(prefix_slots.astype(np.int32)),
                mask=mx.array(mask).astype(self.dtype),
            )
        else:
            meta = BatchMeta(
                is_prefill=True, rope_offset=0,
                pool_slots=mx.array(suffix_slots.astype(np.int32)), mask="causal",
            )
        self.cache.set_meta(meta)
        x = mx.array([ids[start:]])                           # (1, L-start) suffix tokens
        logits = self.model(x, cache=self.cache.layers)[:, -1, :]
        self._last_ids = ()                                   # running set changed -> rebuild
        return logits

    def _rebuild(self, seqs) -> None:
        """Rebuild the contiguous decode buffers from the pool for the current
        running set, right-aligned (newest token shares the last column)."""
        B, bs = len(seqs), self.block_size
        lengths = np.array([s.num_cached for s in seqs], dtype=np.int64)   # history in pool
        end0 = int(lengths.max())
        remaining = max(1, max(s.sampling_params.max_tokens - s.num_output_tokens for s in seqs))
        cap = end0 + remaining + 1
        left_pad = end0 - lengths                                          # (B,)

        maxblk = max(len(s.block_table) for s in seqs)
        bt = np.zeros((B, maxblk), dtype=np.int64)
        for i, s in enumerate(seqs):
            bt[i, :len(s.block_table)] = s.block_table

        cols = np.arange(end0)[None, :]
        pos = cols - left_pad[:, None]                                     # (B, end0)
        valid = pos >= 0
        gslot = bt[np.arange(B)[:, None], np.clip(pos, 0, None) // bs] * bs + np.clip(pos, 0, None) % bs
        flat = mx.array(np.where(valid, gslot, 0).reshape(-1).astype(np.int32))

        H, D = self.n_kv, self.head_dim
        for i in range(self.n_layers):
            bk = mx.zeros((B, H, cap, D), dtype=self.dtype)
            bv = mx.zeros((B, H, cap, D), dtype=self.dtype)
            if end0 > 0:
                bk[:, :, :end0, :] = self.cache.pools_k[i][flat].reshape(B, end0, H, D).transpose(0, 2, 1, 3)
                bv[:, :, :end0, :] = self.cache.pools_v[i][flat].reshape(B, end0, H, D).transpose(0, 2, 1, 3)
            self.cache.buf_k[i] = bk
            self.cache.buf_v[i] = bv

        self.left_pad = left_pad
        self.col = end0
        self._last_ids = tuple(s.seq_id for s in seqs)

    def decode(self, seqs) -> mx.array:
        """One batched decode step over running sequences. Returns logits (B, V)."""
        ids = tuple(s.seq_id for s in seqs)
        cap = self.cache.buf_k[0].shape[2] if self.cache.buf_k[0] is not None else 0
        if ids != self._last_ids or self.col >= cap:
            self._rebuild(seqs)

        B, bs = len(seqs), self.block_size
        ncached = np.array([s.num_cached for s in seqs], dtype=np.int64)   # write positions
        maxblk = max(len(s.block_table) for s in seqs)
        bt = np.zeros((B, maxblk), dtype=np.int64)
        for i, s in enumerate(seqs):
            bt[i, :len(s.block_table)] = s.block_table
        wslot = bt[np.arange(B), ncached // bs] * bs + ncached % bs

        col = self.col
        attn_len = col + 1
        valid = np.arange(attn_len)[None, :] >= self.left_pad[:, None]     # (B, attn_len)
        mask = np.where(valid, 0.0, -1e9).reshape(B, 1, 1, attn_len)

        self.cache.set_meta(BatchMeta(
            is_prefill=False,
            rope_offset=mx.array(ncached.astype(np.int32)),
            pool_slots=mx.array(wslot.astype(np.int32)),
            write_col=col, attn_len=attn_len,
            mask=mx.array(mask).astype(self.dtype),
        ))
        x = mx.array([[s.token_ids[s.num_cached]] for s in seqs])          # (B, 1)
        logits = self.model(x, cache=self.cache.layers)[:, -1, :]
        self.col += 1
        return logits
