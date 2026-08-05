import math
from collections import deque

from .sequence import Sequence


def hash_blocks(token_ids: list[int], block_size: int):
    """Chained per-block content hashes for the full blocks of `token_ids`.

    Block i folds in block i-1's hash, so its hash identifies the entire prefix
    up to block i — identical prefixes get identical hashes. Returns
    (hashes, token_tuples), one entry per full block.
    """
    hashes: list[int] = []
    tuples: list[tuple] = []
    prev = 0
    for i in range(len(token_ids) // block_size):
        toks = tuple(token_ids[i * block_size:(i + 1) * block_size])
        prev = hash((prev, toks))
        hashes.append(prev)
        tuples.append(toks)
    return hashes, tuples


class BlockManager:
    """Allocates fixed-size KV blocks, with content-hash prefix caching.

    A full prompt block is registered by its chained content hash once prefilled;
    a later sequence with the same prefix reuses those physical blocks instead of
    recomputing them. Blocks are ref-counted; an unreferenced cached block stays
    available for reuse and is only physically reclaimed (evicted) when we run
    out of free blocks.

    Reserve-at-admission still holds: a sequence's full potential is reserved up
    front (minus any reused prefix), so decode never runs out of blocks.
    """

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_prefix_cache = enable_prefix_cache
        self.free: deque[int] = deque(range(num_blocks))
        self.ref = [0] * num_blocks              # refcount per physical block
        self.cache: dict[int, int] = {}          # content hash -> block id
        self.hash_of: dict[int, int] = {}        # block id -> content hash
        self.tokens_of: dict[int, tuple] = {}    # block id -> tokens (collision check)

    def blocks_for(self, num_tokens: int) -> int:
        return max(1, math.ceil(num_tokens / self.block_size))

    def available(self) -> int:
        """Free blocks + reclaimable cached blocks (cached but unreferenced)."""
        return len(self.free) + sum(1 for b in self.hash_of if self.ref[b] == 0)

    def _alloc_one(self) -> int:
        if self.free:
            return self.free.popleft()
        for b in list(self.cache.values()):      # evict oldest unreferenced cached block
            if self.ref[b] == 0:
                self.cache.pop(self.hash_of.pop(b))
                self.tokens_of.pop(b, None)
                return b
        raise RuntimeError("out of KV blocks")

    def match_prefix(self, hashes: list[int], tuples: list[tuple]) -> list[int]:
        """Leading cached block ids matching this prefix; ref++'d so they can't be
        evicted while the caller decides whether to admit."""
        if not self.enable_prefix_cache:
            return []
        matched: list[int] = []
        for h, toks in zip(hashes, tuples):
            b = self.cache.get(h)
            if b is None or self.tokens_of.get(b) != toks:
                break
            matched.append(b)
        for b in matched:
            self.ref[b] += 1
        return matched

    def unref(self, blocks: list[int]) -> None:
        for b in blocks:
            self.ref[b] -= 1

    def alloc(self, n: int) -> list[int]:
        blocks = []
        for _ in range(n):
            b = self._alloc_one()
            self.ref[b] += 1
            blocks.append(b)
        return blocks

    def register(self, block_id: int, h: int, toks: tuple) -> None:
        """Make a freshly prefilled full block reusable by later sequences."""
        if not self.enable_prefix_cache or h in self.cache:
            return
        self.cache[h] = block_id
        self.hash_of[block_id] = h
        self.tokens_of[block_id] = toks

    def free_seq(self, seq: Sequence) -> None:
        for b in seq.block_table:
            self.ref[b] -= 1
            if self.ref[b] == 0 and b not in self.hash_of:  # cached blocks stay (evictable)
                self.free.append(b)
        seq.block_table = []
