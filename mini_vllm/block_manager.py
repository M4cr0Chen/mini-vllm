import math
from collections import deque

from .sequence import Sequence


class BlockManager:
    """Allocates fixed-size KV blocks from a shared pool.

    We reserve a sequence's *full* potential (prompt + max output) at admission
    time, so a running sequence can never run out of blocks mid-decode. That
    trades some concurrency for simplicity — no preemption/eviction needed.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free: deque[int] = deque(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self.free)

    def blocks_for(self, num_tokens: int) -> int:
        return max(1, math.ceil(num_tokens / self.block_size))

    def can_allocate(self, num_tokens: int) -> bool:
        return self.num_free >= self.blocks_for(num_tokens)

    def allocate(self, seq: Sequence, num_tokens: int) -> None:
        for _ in range(self.blocks_for(num_tokens)):
            seq.block_table.append(self.free.popleft())

    def free_seq(self, seq: Sequence) -> None:
        self.free.extend(seq.block_table)
        seq.block_table = []
