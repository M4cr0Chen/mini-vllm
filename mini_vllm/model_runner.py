import mlx.core as mx
import numpy as np
from mlx_lm import load

from .config import Config
from .paged_attention import BatchMeta, PagedKVCache, patch_attention


class ModelRunner:
    """Owns the MLX model + tokenizer, the paged KV cache, and the forward
    passes. Reuses mlx-lm's model (weights, quantized projections, RoPE) but
    swaps in our PagedAttention.
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

    def _num_blocks_for_budget(self) -> int:
        itemsize = mx.zeros(1, dtype=self.dtype).nbytes
        per_token = 2 * self.n_layers * self.n_kv * self.head_dim * itemsize  # K + V
        budget = int(self.config.kv_cache_memory_gb * (1024 ** 3))
        return max(1, budget // (per_token * self.block_size))

    def _slots(self, block_table, positions: np.ndarray) -> np.ndarray:
        bt = np.asarray(block_table, dtype=np.int64)
        return bt[positions // self.block_size] * self.block_size + positions % self.block_size

    def prefill(self, seq) -> mx.array:
        """Process the full prompt for one sequence. Returns last logits (1, V)."""
        ids = seq.token_ids
        slots = self._slots(seq.block_table, np.arange(len(ids)))
        self.cache.set_meta(BatchMeta(
            is_prefill=True, rope_offset=0,
            slot_mapping=mx.array(slots.astype(np.int32)), mask="causal",
        ))
        x = mx.array([ids])                                   # (1, L)
        return self.model(x, cache=self.cache.layers)[:, -1, :]

    def decode(self, seqs) -> mx.array:
        """One batched decode step over running sequences. Returns logits (B, V)."""
        bs = self.block_size
        B = len(seqs)
        ncached = np.array([s.num_cached for s in seqs], dtype=np.int64)   # write positions
        lengths = ncached + 1                                              # valid len after write
        Lmax = int(lengths.max())

        maxblk = max(len(s.block_table) for s in seqs)
        bt = np.zeros((B, maxblk), dtype=np.int64)
        for b, s in enumerate(seqs):
            bt[b, :len(s.block_table)] = s.block_table

        rows = np.arange(B)[:, None]
        wslot = bt[np.arange(B), ncached // bs] * bs + ncached % bs        # (B,) new-token slots
        pos = np.arange(Lmax)[None, :]                                     # (1, Lmax)
        gslot = bt[rows, pos // bs] * bs + pos % bs                        # (B, Lmax)
        valid = pos < lengths[:, None]
        gslot = np.where(valid, gslot, 0)
        add_mask = np.where(valid, 0.0, -1e9).reshape(B, 1, 1, Lmax)

        self.cache.set_meta(BatchMeta(
            is_prefill=False,
            rope_offset=mx.array(ncached.astype(np.int32)),
            slot_mapping=mx.array(wslot.astype(np.int32)),
            gather_index=mx.array(gslot.astype(np.int32)),
            mask=mx.array(add_mask).astype(self.dtype),
        ))
        x = mx.array([[s.token_ids[s.num_cached]] for s in seqs])          # (B, 1)
        return self.model(x, cache=self.cache.layers)[:, -1, :]
