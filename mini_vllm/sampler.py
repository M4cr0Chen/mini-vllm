import mlx.core as mx


def sample_tokens(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    """Sample next-token ids from last-position logits.

    logits: (B, V). Returns (B,) int32. Greedy when temperature == 0.
    """
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)
    logits = logits * (1.0 / temperature)
    if 0.0 < top_p < 1.0:
        logits = _top_p_mask(logits, top_p)
    return mx.random.categorical(logits, axis=-1)


def _top_p_mask(logits: mx.array, top_p: float) -> mx.array:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability reaches top_p, mask the rest. Always keeps at least the top-1."""
    probs = mx.softmax(logits, axis=-1)
    order = mx.argsort(-logits, axis=-1)                  # indices, descending
    sorted_probs = mx.take_along_axis(probs, order, axis=-1)
    # exclusive prefix sum: a token is removed once everything before it already
    # covers top_p, so the token that crosses the threshold is retained.
    excl_cumsum = mx.cumsum(sorted_probs, axis=-1) - sorted_probs
    remove_sorted = excl_cumsum > top_p
    inv = mx.argsort(order, axis=-1)                      # invert the permutation
    remove = mx.take_along_axis(remove_sorted, inv, axis=-1)
    return mx.where(remove, -1e9, logits)
