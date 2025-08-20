# test_prompt_sink_qwen3.py
import os
import re
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizer

# --- enable the Triton prompt-sink path
os.environ["VLLM_USE_TRITON_PROMPT_SINK"] = "1"
os.environ["VLLM_SLIDING_WINDOW"] = "128"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# install monkey-patch
from vllm_triton_hook import install_triton_prompt_sink_patch
install_triton_prompt_sink_patch()

MODEL = "Qwen/Qwen3-8B"
PRIME = "806917567"

def build_prompt(tokenizer: PreTrainedTokenizer) -> str:
    # A prompt that forces long thinking then asks for the prime at the end.
    user_msg = (
        f"The special number in this prompt is 23.\n"
        f"You will think step-by-step for a while before answering.\n"
        f"Rules:\n"
        f"1. Write a long chain-of-thought to solve the question.\n"
        f"2. Do not include the special number anywhere in your chain of thought.\n"
        f"The question is: Prime factorize 806917567.\n"
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
    return text

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        MODEL,
        enforce_eager=True,      # ensure eager path so dense K/V is available
        trust_remote_code=True,
    )

    prompt = build_prompt(tok)

    s = SamplingParams(
        temperature=0.0,     # deterministic to simplify checking
        top_p=1.0,
        max_tokens=1200,     # long decode
        # if your vLLM build surfaces sliding window via SamplingParams, add it there;
        # otherwise it’s enforced by our patch reading VLLM_SLIDING_WINDOW
    )

    outputs = llm.generate(prompt, sampling_params=s)
    text_out = outputs[0].outputs[0].text
    prompt = prompt + text_out + "\n\nWhat is the special number mentioned earlier? Output in format <SPECIAL>: {special number}.\n\n"
    print(prompt)
    s = SamplingParams(
        temperature=0.0,     # deterministic to simplify checking
        top_p=1.0,
        max_tokens=100,     # long decode
        # if your vLLM build surfaces sliding window via SamplingParams, add it there;
        # otherwise it’s enforced by our patch reading VLLM_SLIDING_WINDOW
    )
    outputs = llm.generate(prompt, sampling_params=s)
    text_out = outputs[0].outputs[0].text
    ids = outputs[0].outputs[0].token_ids
    print(text_out)
    print("num_generated_tokens:", len(ids))

    tail = text_out[-5000:]
    found = re.search(rf"\b{PRIME}\b", tail) is not None

    if found:
        print("[PASS] Model recalled the prime from the prompt after long decode.")
    else:
        print("[FAIL] Model did not recall the prime near the end. "
              "If sink is off (or window too small), this often fails.")
        exit(1)

if __name__ == "__main__":
    main()
