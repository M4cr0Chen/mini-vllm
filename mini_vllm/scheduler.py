from collections import deque

from .block_manager import BlockManager, hash_blocks
from .config import Config
from .sequence import Sequence, SequenceStatus


class Scheduler:
    """Continuous-batching scheduler: admit waiting sequences as capacity frees
    up (reusing cached prompt prefixes), decode all running sequences together."""

    def __init__(self, config: Config, block_manager: BlockManager):
        self.config = config
        self.bm = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def try_admit(self) -> Sequence | None:
        """Admit one waiting sequence if there's a batch slot and block budget.
        Reuses cached prefix blocks; reserves the rest of its potential length."""
        if not self.waiting or len(self.running) >= self.config.max_num_seqs:
            return None
        seq = self.waiting[0]
        bs = self.config.block_size
        L = seq.num_prompt_tokens

        hashes, tuples = hash_blocks(seq.prompt_token_ids, bs)
        matched = self.bm.match_prefix(hashes, tuples)         # ref++'d
        reuse = min(len(matched), (L - 1) // bs)               # keep >=1 token to prefill
        if len(matched) > reuse:
            self.bm.unref(matched[reuse:])
        reused = matched[:reuse]

        potential = min(L + seq.sampling_params.max_tokens, self.config.max_model_len)
        n_new = self.bm.blocks_for(potential) - reuse
        if self.bm.available() < n_new:
            self.bm.unref(reused)                              # release, retry later
            if not self.running:
                raise RuntimeError(
                    f"KV cache too small for a sequence of {potential} tokens; "
                    f"increase kv_cache_memory_gb or lower max_tokens."
                )
            return None

        self.waiting.popleft()
        seq.block_table = reused + self.bm.alloc(n_new)
        seq.block_hashes = hashes
        seq.prefill_start = reuse * bs
        seq.num_cached = reuse * bs
        seq.status = SequenceStatus.RUNNING
        return seq

    def drop_finished(self) -> None:
        still = []
        for seq in self.running:
            if seq.is_finished:
                self.bm.free_seq(seq)
            else:
                still.append(seq)
        self.running = still
