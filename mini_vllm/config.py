from dataclasses import dataclass, fields


@dataclass
class Config:
    """Engine configuration."""

    model: str
    max_num_seqs: int = 64             # max sequences decoded concurrently
    block_size: int = 16               # tokens per KV block
    kv_cache_memory_gb: float = 4.0    # budget for the paged KV pools
    max_model_len: int = 8192          # hard cap on prompt+output length
    enable_prefix_cache: bool = True   # reuse KV blocks for shared prompt prefixes

    @classmethod
    def from_kwargs(cls, model: str, **kwargs) -> "Config":
        """Build a Config, silently ignoring unknown kwargs so the public
        API can accept vLLM-style options we don't implement yet."""
        known = {f.name for f in fields(cls)}
        return cls(model=model, **{k: v for k, v in kwargs.items() if k in known})
