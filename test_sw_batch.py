import os
import gc
import torch
import multiprocessing as mp
from typing import Dict, Optional, List

# ---------- Worker that runs a single configuration on a batch of prompts ----------
def run_config(name: str, model_name: str, prompts: List[str], env_vars: Dict[str, str],
               hf_overrides: Optional[Dict], max_tokens: int, q: mp.Queue):
    try:
        # Set env vars for this child process only
        if env_vars:
            os.environ.update(env_vars)

        # Import inside process to ensure clean CUDA state per process (spawn)
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Build a list of templated texts (one per prompt) for chat generation
        texts = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            texts.append(text)

        s = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)

        # Create model with optional hf overrides
        if hf_overrides:
            model = LLM(model_name, hf_overrides=hf_overrides)
        else:
            model = LLM(model_name)

        # Generate on a batch of texts
        out = model.generate(texts, sampling_params=s)

        # Extract outputs for each input in the batch
        batch_outputs = []
        for item in out:
            try:
                batch_outputs.append(item.outputs[0].text)
            except Exception:
                batch_outputs.append(str(item))

        # cleanup inside process (not strictly necessary since process will exit, but tidy)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        q.put((name, True, batch_outputs))
    except Exception as e:
        q.put((name, False, f"EXCEPTION: {type(e).__name__}: {e}"))

# ---------- Main: spawn one process per configuration, sequentially ----------
if __name__ == "__main__":
    ctx = mp.get_context("spawn")

    model_name = "/data/niklas/s2/Qwen3-1.7B"

    # Two similar but slightly different prompts (different lengths)
    prompts = [
        "Prime factorize the non-prime 806917567.",
        "Please compute and list the prime factorization of the integer 806917567 for me, step by step."
    ]

    # Define the configs as in your script
    configs = [
        ("flash", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}, None),
        ("flash_sw", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "SW": "32", "SW_regular_rope": "1"}, None),
        ("triton", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1"}, None),
        ("triton_sw", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "SWT": "32"}, {"use_sliding_window": True, "sliding_window": 32}),
    ]

    q = ctx.Queue()
    results = {}

    # ---------- Phase A: short outputs (inside SW) ----------
    print("\n=== PHASE A: max_tokens=32 (expect SW and non-SW parity where intended) ===")
    for name, env_vars, hf_overrides in configs:
        print(f"\n--- Running config: {name} ---")
        p = ctx.Process(target=run_config, args=(name, model_name, prompts, env_vars, hf_overrides, 32, q))
        p.start()
        p.join()

        if not q.empty():
            cfg_name, ok, payload = q.get()
            if not ok:
                raise RuntimeError(f"Config {cfg_name} failed with: {payload}")
            results[cfg_name] = payload  # payload is a list of outputs (one per prompt)
            print(f"-> {cfg_name} outputs captured (batch size {len(payload)}).")
            for i, out in enumerate(payload):
                print(f"  [{i}] len={len(out)} chars")
        else:
            raise RuntimeError(f"No result returned for config {name} (queue empty).")

    print("\nPHASE A outputs:")
    for name in ["flash", "flash_sw", "triton", "triton_sw"]:
        print("-" * 10, name, "-" * 10)
        for i, out in enumerate(results[name]):
            print(f"[{i}] {out}")
        print()


    # Element-wise equality assertions across configs for phase A
    for i in range(len(prompts)):
        assert results["flash"][i] == results["flash_sw"][i], f"Phase A: flash vs flash_sw differ for item {i}!"
        assert results["flash"][i] == results["triton"][i], f"Phase A: flash vs triton differ for item {i}!"
        assert results["flash"][i] == results["triton_sw"][i], f"Phase A: flash vs triton_sw differ for item {i}!"

    # ---------- Phase B: longer outputs (outside SW) ----------
    print("\n=== PHASE B: max_tokens=64 (expect differences between SW and non-SW) ===")
    for name, env_vars, hf_overrides in configs:
        print(f"\n--- Running config: {name} ---")
        p = ctx.Process(target=run_config, args=(name, model_name, prompts, env_vars, hf_overrides, 64, q))
        p.start()
        p.join()

        if not q.empty():
            cfg_name, ok, payload = q.get()
            if not ok:
                raise RuntimeError(f"Config {cfg_name} failed with: {payload}")
            results[cfg_name] = payload
            print(f"-> {cfg_name} outputs captured (batch size {len(payload)}).")
            for i, out in enumerate(payload):
                print(f"  [{i}] len={len(out)} chars")
        else:
            raise RuntimeError(f"No result returned for config {name} (queue empty).")

    print("\nPHASE B outputs:")
    for name in ["flash", "flash_sw", "triton", "triton_sw"]:
        print("-" * 10, name, "-" * 10)
        for i, out in enumerate(results[name]):
            print(f"[{i}] {out}")
        print()

    # Element-wise assertions for phase B
    for i in range(len(prompts)):
        # triton and flash should match for same max_tokens
        assert results["flash"][i] == results["triton"][i], f"Phase B: triton and flash differ for item {i}!"
        # SW variants should differ from non-SW (for your expectation)
        assert results["flash"][i] != results["flash_sw"][i], f"Phase B: flash and flash_sw unexpectedly equal for item {i}!"
        assert results["flash"][i] != results["triton_sw"][i], f"Phase B: flash and triton_sw unexpectedly equal for item {i}!"
        # SW implementations should match each other element-wise
        assert results["triton_sw"][i] == results["flash_sw"][i], f"Phase B: triton_sw and flash_sw differ for item {i}!"

    print("All checks passed for batch size 2.")
