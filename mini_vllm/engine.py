from .config import Config
from .model_runner import ModelRunner
from .sampler import sample_tokens
from .sampling_params import SamplingParams
from .sequence import Sequence, SequenceStatus


class LLMEngine:
    """Drives generation."""

    def __init__(self, model: str, **kwargs):
        self.config = Config.from_kwargs(model, **kwargs)
        self.model_runner = ModelRunner(self.config)
        self.tokenizer = self.model_runner.tokenizer

    def _run_sequence(self, seq: Sequence) -> None:
        mr = self.model_runner
        sp = seq.sampling_params
        cache = mr.make_cache()
        seq.status = SequenceStatus.RUNNING

        # Prefill: consume the whole prompt, get logits for the first new token.
        logits = mr.forward([seq.token_ids], cache)   # (1, V)
        max_new = min(sp.max_tokens, self.config.max_model_len - len(seq))
        for _ in range(max_new):
            token_id = int(sample_tokens(logits, sp.temperature, sp.top_p).item())
            if not sp.ignore_eos and token_id in mr.eos_ids:
                break
            seq.append_token(token_id)
            logits = mr.forward([[token_id]], cache)  # (1, V) decode step
        seq.status = SequenceStatus.FINISHED

    def generate(self, prompts: list[str], sampling_params) -> list[dict]:
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        results = []
        for prompt, sp in zip(prompts, sampling_params):
            token_ids = self.tokenizer.encode(prompt)
            seq = Sequence(token_ids, sp)
            self._run_sequence(seq)
            results.append(
                {
                    "text": self.tokenizer.decode(seq.output_ids),
                    "token_ids": seq.output_ids,
                }
            )
        return results
