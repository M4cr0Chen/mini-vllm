import mlx.core as mx

from mini_vllm import LLM, SamplingParams

mx.random.seed(0)

llm = LLM("mlx-community/Qwen3-0.6B-4bit")

prompts = [
    "What is the capital of France? Answer in one word.",
    "Write a haiku about the ocean.",
    "Name three primary colors.",
]

# Apply the model's chat template so the instruct model behaves.
texts = [
    llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
    )
    for p in prompts
]

outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=128))

for prompt, out in zip(prompts, outs):
    print("=" * 70)
    print("PROMPT:", prompt)
    print("OUTPUT:", out["text"].strip())
    print(f"({len(out['token_ids'])} tokens)")
