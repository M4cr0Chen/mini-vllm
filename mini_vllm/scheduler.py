from collections import deque

from .block_manager import BlockManager
from .config import Config
from .sequence import Sequence, SequenceStatus


class Scheduler:
    """Continuous-batching scheduler: admit waiting sequences as capacity frees
    up, decode all running sequences together each step."""

    def __init__(self, config: Config, block_manager: BlockManager):
        self.config = config
        self.bm = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _potential_len(self, seq: Sequence) -> int:
        return min(seq.num_prompt_tokens + seq.sampling_params.max_tokens,
                   self.config.max_model_len)

    def try_admit(self) -> Sequence | None:
        """Admit one waiting sequence if there's batch slot + block budget for
        its full potential length. Returns the admitted (block-allocated) seq."""
        if not self.waiting or len(self.running) >= self.config.max_num_seqs:
            return None
        seq = self.waiting[0]
        need = self._potential_len(seq)
        if not self.bm.can_allocate(need):
            if not self.running:  # can't even fit one sequence
                raise RuntimeError(
                    f"KV cache too small for a sequence of {need} tokens; "
                    f"increase kv_cache_memory_gb or lower max_tokens."
                )
            return None
        self.waiting.popleft()
        self.bm.allocate(seq, need)
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
