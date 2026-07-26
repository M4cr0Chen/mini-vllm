import mlx.core as mx

from .block_manager import BlockManager
from .config import Config
from .model_runner import ModelRunner
from .sampler import sample_tokens
from .sampling_params import SamplingParams
from .scheduler import Scheduler
from .sequence import Sequence, SequenceStatus


class LLMEngine:
    """Continuous-batching engine: prefill admitted sequences, then decode all
    running sequences together, replacing finished ones as capacity frees up."""

    def __init__(self, model: str, **kwargs):
        self.config = Config.from_kwargs(model, **kwargs)
        self.model_runner = ModelRunner(self.config)
        self.tokenizer = self.model_runner.tokenizer
        self.block_manager = BlockManager(self.model_runner.num_blocks, self.config.block_size)
        self.scheduler = Scheduler(self.config, self.block_manager)

    def _sample(self, logits: mx.array, seqs: list[Sequence]) -> list[int]:
        if all(s.sampling_params.temperature == 0.0 for s in seqs):
            toks = mx.argmax(logits, axis=-1)
        else:
            temps = mx.array([s.sampling_params.temperature for s in seqs])
            topps = mx.array([s.sampling_params.top_p for s in seqs])
            toks = sample_tokens(logits, temps, topps)
        return toks.tolist()

    def _accept(self, seq: Sequence, token_id: int) -> None:
        sp = seq.sampling_params
        if not sp.ignore_eos and token_id in self.model_runner.eos_ids:
            seq.status = SequenceStatus.FINISHED
            return
        seq.append_token(token_id)
        if seq.num_output_tokens >= sp.max_tokens or len(seq) >= self.config.max_model_len:
            seq.status = SequenceStatus.FINISHED

    def generate(self, prompts: list[str], sampling_params) -> list[dict]:
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        seqs = [Sequence(self.tokenizer.encode(p), sp)
                for p, sp in zip(prompts, sampling_params)]
        for seq in seqs:
            self.scheduler.add(seq)

        while self.scheduler.has_unfinished():
            # Admission + prefill (one sequence at a time).
            while True:
                seq = self.scheduler.try_admit()
                if seq is None:
                    break
                logits = self.model_runner.prefill(seq)        # (1, V)
                token = self._sample(logits, [seq])[0]
                seq.num_cached = len(seq.token_ids)            # prompt now cached
                self._accept(seq, token)
                if seq.is_finished:
                    self.block_manager.free_seq(seq)
                else:
                    self.scheduler.running.append(seq)

            # Batched decode over all running sequences.
            running = self.scheduler.running
            if running:
                logits = self.model_runner.decode(running)     # (B, V)
                tokens = self._sample(logits, running)
                for seq, token in zip(running, tokens):
                    seq.num_cached += 1
                    self._accept(seq, token)
                self.scheduler.drop_finished()

        return [{"text": self.tokenizer.decode(s.output_ids), "token_ids": s.output_ids}
                for s in seqs]
