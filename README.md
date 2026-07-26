# mini-vllm

A clean, minimal (~1k line) vLLM-style LLM inference engine for **Apple Silicon**, built on **[MLX](https://github.com/ml-explore/mlx)**.

It's the Apple-Silicon analog of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — a minimal ~1,200-line reimplementation of vLLM that is, however, CUDA-only. `mini-vllm` preserves the same optimization techniques (paged attention, prefix caching, continuous batching, kernel fusion) but runs natively on your Mac.

## Why MLX

Rather than fight CUDA-isms on Apple hardware, `mini-vllm` leans into MLX:

| vLLM / nano-vllm (CUDA) | mini-vllm (MLX) |
| --- | --- |
| flash-attn kernel | `mx.fast.scaled_dot_product_attention` |
| CUDA graphs (decode launch overhead) | lazy eval + `mx.async_eval` (+ optional `mx.compile`) |
| torch.compile fusion | `mx.compile` |
| explicit host↔device KV-cache copies | unified memory — no copies |
| tensor parallelism (NCCL) | single unified-memory device |

## Install

Requires Apple Silicon and Python 3.10–3.13 ([MLX](https://pypi.org/project/mlx/) wheels).

```bash
uv venv --python 3.13
uv pip install -e .
```

## Usage

```python
from mini_vllm import LLM, SamplingParams

llm = LLM("mlx-community/Qwen3-0.6B-4bit")

prompt = llm.tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is the capital of France?"}],
    add_generation_prompt=True, tokenize=False,
)
out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=128))
print(out[0]["text"])
```

Or run the bundled demo:

```bash
python example.py
```

## Benchmarks

Apple **M4** (base), `Qwen3-0.6B-4bit`:

| mode | throughput |
| --- | --- |
| single-stream decode | ~99 tok/s |
| continuous batching (256 reqs, 64 concurrent) | ~310 tok/s aggregate decode |

Continuous batching gives ~3× over single-stream on a base M4. The paged-attention decode path still has headroom — gather materialization and per-step host work — later targets via `mx.async_eval` pipelining and `mx.compile`. Numbers scale up substantially on M-series Pro/Max.

## Credits

Inspired by [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [vLLM](https://github.com/vllm-project/vllm). Built on [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm).
