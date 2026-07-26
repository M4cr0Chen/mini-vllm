import mlx.core as mx


def _is_scalar(x) -> bool:
    return isinstance(x, (int, float))


def _as_col(x, B: int) -> mx.array:
    """Coerce a scalar or (B,) array of per-row params into shape (B, 1)."""
    if _is_scalar(x):
        return mx.full((B, 1), float(x))
    return mx.array(x).reshape(B, 1)


def sample_tokens(logits: mx.array, temperature, top_p) -> mx.array:
    """Sample next-token ids from last-position logits.

    logits: (B, V). ``temperature`` / ``top_p`` may be scalars or (B,) arrays
    (per-sequence sampling). Returns (B,) int. Greedy where temperature == 0.
    """
    # Fast path: uniform greedy (the benchmark default) — no sort, no rng.
    if _is_scalar(temperature) and float(temperature) == 0.0:
        return mx.argmax(logits, axis=-1)
    # Fast path: uniform sampling without nucleus filtering.
    if _is_scalar(temperature) and _is_scalar(top_p) and float(top_p) >= 1.0:
        return mx.random.categorical(logits * (1.0 / float(temperature)), axis=-1)

    B = logits.shape[0]
    temp = _as_col(temperature, B)                     # (B, 1)
    topp = _as_col(top_p, B)                            # (B, 1)
    greedy = mx.argmax(logits, axis=-1)                # (B,)
    safe_temp = mx.where(temp == 0, 1.0, temp)
    scaled = logits / safe_temp
    scaled = _top_p_mask(scaled, topp)
    sampled = mx.random.categorical(scaled, axis=-1)   # (B,)
    return mx.where(temp[:, 0] == 0, greedy, sampled)


def _top_p_mask(logits: mx.array, top_p) -> mx.array:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability reaches top_p, mask the rest. ``top_p`` broadcasts as (B, 1);
    rows with top_p >= 1 are left unfiltered. Always keeps at least the top-1."""
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
