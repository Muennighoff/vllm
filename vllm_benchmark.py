# SW=2048 VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=2 python /data/niklas/s2/scripts/generate.py

###### BENCHMARKING ######
### 32 Texts, 4096 max toks ###
# CUDA_VISIBLE_DEVICES=5 python /data/niklas/s2/scripts/generate.py
# 4344.125590878791 toks/s
# with FlashInfer: 4991.882942649869 toks/s
# (seems equiv to: VLLM_ATTENTION_BACKEND=FLASHINFER_VLLM_V1): CUDA_VISIBLE_DEVICES=6 python /data/niklas/s2/scripts/generate.py)
# with (not global) sliding_window=512: 5757.767123588528 toks/s
# with (not global) sliding_window=512 and FlashInfer: 6929.007006951192 toks/s

# SW=2048 CUDA_VISIBLE_DEVICES=5 python /data/niklas/s2/scripts/generate.py
# 578.3636140204685 toks/s
# with FlashInfer: 586.3713944745573 toks/s

# SW=512 CUDA_VISIBLE_DEVICES=5 python /data/niklas/s2/scripts/generate.py
# 1030.4718771603539 toks/s

# SW_regular_rope=1 SW=512 CUDA_VISIBLE_DEVICES=5 python /data/niklas/s2/scripts/generate.py
# 1447.9118274074674 toks/s
# VLLM_ATTENTION_BACKEND=FLEX_ATTENTION CUDA_VISIBLE_DEVICES=6 python /data/niklas/s2/scripts/generate.py
# Needs VLLM_ENABLE_V1_MULTIPROCESSING=0 & tensor_parallel_size=1 (or enforce_eager=True)
# with enforce_eager:
# torch 2.8: 170.66402481777794 toks/s
# torch 2.7.1: 158.43715367785887 toks/s
# no enforce_eager:
# torch 2.9: 460.6654434795455 toks/s

# VLLM_ATTENTION_BACKEND=XFORMERS_VLLM_V1 CUDA_VISIBLE_DEVICES=6 python /data/niklas/s2/scripts/generate.py
# torch 2.9, block_size=64: 2220.8933877009285 toks/s
# torch 2.7.1, block_size=64: 2101.4861146982316 toks/s

# VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 CUDA_VISIBLE_DEVICES=6 python /data/niklas/s2/scripts/generate.py
# torch 2.7.1: 3754.069393028881 toks/s
# with FlashInfer: 4338.991243530769 toks/s
# with sliding_window=512 and FlashInfer: 4202.187879235802 toks/s
# with sliding_window=512 and FlashInfer and SWT=512: 4167.270110753185 toks/s

import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = LLM(model_name)#,enforce_eager=True)#, block_size=64)#, tensor_parallel_size=1) # , block_size=64) # enforce_eager=True)
model = LLM(model_name, hf_overrides={"use_sliding_window": True, "sliding_window": int(os.environ.get("SW", "512"))})
# model = LLM(model_name, hf_overrides={"use_sliding_window": True, "sliding_window": int(os.environ.get("SW", "512"))}, gpu_memory_utilization=0.8) # Set enforce_eager=True for easier debugging but slower
# model = LLM(model_name, hf_overrides={"use_sliding_window": False, "sliding_window": None})
# model = LLM(model_name, disable_sliding_window=True)
# debug with pdb # https://github.com/vllm-project/vllm/issues/13120
# also requires `VLLM_ENABLE_V1_MULTIPROCESSING=0`
# Attention backends: ['FLASH_ATTN', 'FLASH_ATTN_VLLM_V1', 'TRITON_ATTN_VLLM_V1', 'XFORMERS', 'ROCM_FLASH', 'ROCM_AITER_MLA', 'ROCM_AITER_MLA_VLLM_V1', 'ROCM_AITER_FA', 'TORCH_SDPA', 'FLASHINFER', 'FLASHINFER_VLLM_V1', 'TRITON_MLA', 'TRITON_MLA_VLLM_V1', 'FLASHMLA_VLLM_V1', 'FLASHMLA', 'CUTLASS_MLA', 'PALLAS', 'PALLAS_VLLM_V1', 'IPEX', 'DUAL_CHUNK_FLASH_ATTN', 'DIFFERENTIAL_FLASH_ATTN', 'NO_ATTENTION', 'FLEX_ATTENTION', 'TREE_ATTN', 'XFORMERS_VLLM_V1']
# Can use:
# FLASH_ATTN; TORCH_SDPA is only CPU; etc
# e.g. via VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
# model = LLM(model_name, enforce_eager=True)

# prepare the model input
# prompt = "Prime factorize 806917567" # 21 toks: '<|im_start|>user\nPrime factorize 806917567<|im_end|>\n<|im_start|>assistant\n'
prompt = "Prime factorize the non-prime 806917567." #
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
text += "<think>\n"


s = SamplingParams(
    temperature=1,
    top_p=0.95,
    max_tokens=4096,
#    max_tokens=100_000,
)

import time
start_time = time.time()
output = model.generate(
    [text] * 32,
    sampling_params=s
)
end_time = time.time()
toks = [len(o.outputs[0].token_ids) for o in output]
print(output[0].outputs[0].text)
print(f"{toks} toks")
print(f"{end_time - start_time}s")
print(f"{sum(toks) / (end_time - start_time)} toks/s")