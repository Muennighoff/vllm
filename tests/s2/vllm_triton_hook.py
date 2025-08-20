# vllm_triton_hook.py
import os
import torch
from vllm.attention.utils.sink_wrapper import union_prompt_sliding_decode_step

_USE_TRITON = os.getenv("VLLM_USE_TRITON_PROMPT_SINK", "0") == "1"
_SLIDING_WINDOW = int(os.getenv("VLLM_SLIDING_WINDOW", "128"))

def install_triton_prompt_sink_patch():
    if not _USE_TRITON:
        return

    # Import inside to avoid hard dependency when env var is off
    import vllm.attention.backends.flash_attn as fa_backend

    if getattr(fa_backend, "_ORIG_FA_FORWARD", None) is None:
        fa_backend._ORIG_FA_FORWARD = fa_backend.flash_attn_with_kvcache

    def _patched_fa_forward(*args, **kwargs):
        """
        Expected signature in vLLM (roughly):
        flash_attn_with_kvcache(q, k_cache, v_cache, attn_metadata, ...)
        We only intercept the DECODE path where q is [B, H, 1, D]
        and 'attn_metadata' has .prompt_lens and .seq_lens etc.
        """
        attn_md = kwargs.get("attn_metadata", None) or (len(args) >= 4 and args[3])
        q = args[0]  # [B, H, 1, D] or [B, H, D]
        k_cache = args[1]
        v_cache = args[2]

        # fast path: if we can’t figure out shapes/metadata, fallback
        if not hasattr(attn_md, "is_decoding") or not attn_md.is_decoding:
            return fa_backend._ORIG_FA_FORWARD(*args, **kwargs)

        # Pull minimal metadata
        B, H = q.shape[0], q.shape[1]
        D = q.shape[-1]
        t_positions = attn_md.seq_lens.to(q.device) - 1  # current t per sequence (or supply directly)
        prompt_lens = attn_md.prompt_lens.to(q.device)   # [B], number of prompt tokens

        # Get dense K/V for dev (replace with real gather; some eager paths already have them)
        # Here we assume attn_md exposes dense k/v (many eager configs do)
        if hasattr(attn_md, "dense_k") and attn_md.dense_k is not None:
            k_dense = attn_md.dense_k  # [B, H, T, D]
            v_dense = attn_md.dense_v
        else:
            # If not available, fallback to original
            return fa_backend._ORIG_FA_FORWARD(*args, **kwargs)

        # shape fix: q might be [B,H,1,D]
        q_ = q.squeeze(-2).contiguous()  # [B,H,D]

        out = union_prompt_sliding_decode_step(
            q_, k_dense, v_dense,
            prompt_lens=prompt_lens,
            t_positions=t_positions,
            window_size=_SLIDING_WINDOW,
        )
        # restore [B,H,1,D] if needed
        out = out.unsqueeze(-2)
        return out

    fa_backend.flash_attn_with_kvcache = _patched_fa_forward
