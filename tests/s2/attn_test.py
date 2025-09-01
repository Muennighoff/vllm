# test_prompt_sink_qwen3.py
import os
import re
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizer

# --- enable the Triton prompt-sink path
# os.environ["VLLM_USE_TRITON_PROMPT_SINK"] = "1"
# os.environ["VLLM_SLIDING_WINDOW"] = "128"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "1"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["SW"] = "128"
os.environ["SW_regular_rope"] = "1"

# install monkey-patch
from vllm_triton_hook import install_triton_prompt_sink_patch
# install_triton_prompt_sink_patch()

MODEL = "Qwen/Qwen3-8B"
# MODEL = "Qwen/Qwen3-1.7B"
# MODEL = "openai/gpt-oss-20b"
PRIME = "806917567"

def build_prompts(tokenizer: PreTrainedTokenizer) -> list[str]:
    # Example variations with different "special numbers" or questions
    special_numbers = [23, 37, 53] * 20
    questions = [
        "Prime factorize 806917567.",
        "Prime factorize 9999991.",
        "Prime factorize 1234567.",
    ] * 20

    prompts = []
    for num, q in zip(special_numbers, questions):
        user_msg = (
            f"The special number in this prompt is {num}.\n"
            f"You will think step-by-step for a while before answering.\n"
            f"Rules:\n"
            f"1. Write a long chain-of-thought to solve the question.\n"
            f"2. Do not include the special number anywhere in your chain of thought.\n"
            f"The question is: {q}\n"
        )
        messages = [
            {"role": "user", "content": user_msg}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        text += "<think>\n"
        prompts.append(text)

    return prompts

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        MODEL,
        enforce_eager=True,      # ensure eager path so dense K/V is available
        trust_remote_code=True,
    )

    prompts = build_prompts(tok)

    s = SamplingParams(
        temperature=0.0,     # deterministic to simplify checking
        top_p=1.0,
        max_tokens=12000,     # long decode
        # if your vLLM build surfaces sliding window via SamplingParams, add it there;
        # otherwise it’s enforced by our patch reading VLLM_SLIDING_WINDOW
    )

    outputs = llm.generate(prompts, sampling_params=s)
    text_out = outputs[0].outputs[0].text
    ids = outputs[0].outputs[0].token_ids
    print(text_out)
    print("num_generated_tokens:", len(ids))

if __name__ == "__main__":
    main()
