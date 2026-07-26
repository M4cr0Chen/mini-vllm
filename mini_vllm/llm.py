from .engine import LLMEngine
from .sampling_params import SamplingParams


class LLM:
    """Public entry point, mirroring nano-vllm/vLLM.

        llm = LLM("mlx-community/Qwen3-0.6B-4bit")
        outs = llm.generate(["Hello!"], SamplingParams(max_tokens=64))
        print(outs[0]["text"])
    """

    def __init__(self, model: str, **kwargs):
        self.engine = LLMEngine(model, **kwargs)
        self.tokenizer = self.engine.tokenizer

    def generate(self, prompts, sampling_params=None) -> list[dict]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()
        return self.engine.generate(prompts, sampling_params)
