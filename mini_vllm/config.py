from dataclasses import dataclass, fields


@dataclass
class Config:
    """Engine configuration."""

    model: str
    max_num_seqs: int = 64          # scheduler batch ceiling (M2)
    kv_block_size: int = 256        # tokens per KV block (M2)
    max_model_len: int = 8192       # hard cap on prompt+output length

    @classmethod
    def from_kwargs(cls, model: str, **kwargs) -> "Config":
        """Build a Config, silently ignoring unknown kwargs so the public
        API can accept vLLM-style options we don't implement yet."""
        known = {f.name for f in fields(cls)}
        return cls(model=model, **{k: v for k, v in kwargs.items() if k in known})
