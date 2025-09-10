import os
import gc
import torch
import multiprocessing as mp
from typing import Dict, Optional

# ---------- Worker that runs a single configuration ----------
def run_config(name: str, model_name: str, prompt: str, env_vars: Dict[str, str], hf_overrides: Optional[Dict], max_tokens: int, q: mp.Queue):
    try:
        # Set env vars for this child process only
        if env_vars:
            os.environ.update(env_vars)

        # Import inside process to ensure clean CUDA state per process (spawn)
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        s = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)

        # Create model with optional hf overrides
        if hf_overrides:
            model = LLM(model_name, hf_overrides=hf_overrides)
        else:
            model = LLM(model_name)

        out = model.generate(text, sampling_params=s)
        result_text = out[0].outputs[0].text

        # cleanup inside process (not strictly necessary since process will exit, but tidy)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        q.put((name, True, result_text))
    except Exception as e:
        # send back the exception info
        q.put((name, False, f"EXCEPTION: {type(e).__name__}: {e}"))

# ---------- Main: spawn one process per configuration, sequentially ----------
if __name__ == "__main__":
    # Use spawn to avoid inheriting CUDA context
    ctx = mp.get_context("spawn")

    model_name = "/data/niklas/s2/Qwen3-1.7B"
    # prompt = "Prime factorize the non-prime 806917567."
    prompt = "Please compute and list the prime factorization of the integer 806917567 for me, step by step."
    #prompt = "Please prime factorize the non-prime 806917567 into its factors."
    # Define the configs exactly like your original script
    configs = [
        ("flash", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}, None),
        # ("flash_sw", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "SW": "32", "SW_regular_rope": "1"}, None),
        ("flash_sw", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "SW": "32"}, None),        
        ("triton", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1"}, None),
        ("triton_sw", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1", "SWT": "32"}, {"use_sliding_window": True, "sliding_window": 32}),
    ]

    q = ctx.Queue()
    results = {}


    ### Test that texts are same within SW ###
    for name, env_vars, hf_overrides in configs:
        print(f"\n=== Running config: {name} ===")
        p = ctx.Process(target=run_config, args=(name, model_name, prompt, env_vars, hf_overrides, 32, q))
        p.start()
        p.join()  # wait until the process fully exits (this ensures GPU memory is freed)

        # Get result from queue
        if not q.empty():
            cfg_name, ok, payload = q.get()
            if not ok:
                raise RuntimeError(f"Config {cfg_name} failed with: {payload}")
            results[cfg_name] = payload
            print(f"-> {cfg_name} output captured (len {len(payload)} chars).")
        else:
            raise RuntimeError(f"No result returned for config {name} (queue empty).")

    print("\n" + "-" * 10)
    print("flash:     ", results["flash"])
    print("-" * 10)
    print("flash_sw:  ", results["flash_sw"])
    print("-" * 10)
    print("triton:    ", results["triton"])
    print("-" * 10)
    print("triton_sw: ", results["triton_sw"])
    print("-" * 10)

    assert results["flash"] == results["flash_sw"], "Flash+SW and Flash outputs differ!"
    assert results["flash"] == results["triton"], "Triton and Flash outputs differ!"
    assert results["flash"] == results["triton_sw"], "Triton+SW and Flash outputs differ!"

    ### Test that texts differ outside SW ###
    for name, env_vars, hf_overrides in configs:
        print(f"\n=== Running config: {name} ===")
        p = ctx.Process(target=run_config, args=(name, model_name, prompt, env_vars, hf_overrides, 64, q))
        p.start()
        p.join()  # wait until the process fully exits (this ensures GPU memory is freed)

        # Get result from queue
        if not q.empty():
            cfg_name, ok, payload = q.get()
            if not ok:
                raise RuntimeError(f"Config {cfg_name} failed with: {payload}")
            results[cfg_name] = payload
            print(f"-> {cfg_name} output captured (len {len(payload)} chars).")
        else:
            raise RuntimeError(f"No result returned for config {name} (queue empty).")

    print("\n" + "-" * 10)
    print("flash:     ", results["flash"])
    print("-" * 10)
    print("flash_sw:  ", results["flash_sw"])
    print("-" * 10)
    print("triton:    ", results["triton"])
    print("-" * 10)
    print("triton_sw: ", results["triton_sw"])
    print("-" * 10)

    assert results["flash"] == results["triton"], "Triton and Flash outputs differ!"
    assert results["flash"] != results["flash_sw"], "Flash+SW and Flash are equal!"
    assert results["flash"] != results["triton_sw"], "Triton+SW and Flash are equal!"
    assert results["triton_sw"] == results["flash_sw"], "Triton+SW and Flash+SW differ!"
