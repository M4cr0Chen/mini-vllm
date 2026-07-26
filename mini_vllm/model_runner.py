import mlx.core as mx
from mlx_lm import load

try:
    from mlx_lm.models.cache import make_prompt_cache
except ImportError:  # older/newer layouts
    from mlx_lm.cache import make_prompt_cache

from .config import Config


class ModelRunner:
    """Owns the MLX model + tokenizer and runs forward passes.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model, self.tokenizer = load(config.model)
        eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if not eos_ids:
            eos_ids = [self.tokenizer.eos_token_id]
        self.eos_ids = {int(t) for t in eos_ids if t is not None}

    def make_cache(self):
        return make_prompt_cache(self.model)

    def forward(self, input_ids, cache) -> mx.array:
        """input_ids: (B, T) array-like. Returns last-position logits (B, V)."""
        x = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        logits = self.model(x, cache=cache)   # (B, T, V)
        return logits[:, -1, :]               # (B, V)
