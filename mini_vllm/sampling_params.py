from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Per-request sampling configuration (mirrors vLLM/nano-vllm)."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
