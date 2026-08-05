import itertools
from enum import Enum, auto

from .sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """State for a single generation request: its tokens, sampling params, and
    lifecycle status.

    Paging state:
      * ``block_table`` — physical KV block ids assigned to this sequence.
      * ``num_cached``  — how many tokens already have their KV written to the
        pool. The next token to process is ``token_ids[num_cached]`` at position
        ``num_cached``.
    """

    _counter = itertools.count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = next(Sequence._counter)
        self.prompt_token_ids = list(token_ids)
        self.token_ids = list(token_ids)          # prompt + generated
        self.output_ids: list[int] = []           # generated only
        self.sampling_params = sampling_params
        self.status = SequenceStatus.WAITING
        self.block_table: list[int] = []
        self.num_cached = 0
        self.block_hashes: list[int] = []         # chained hash per full prompt block
        self.prefill_start = 0                    # tokens reused from the prefix cache

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.output_ids.append(token_id)
