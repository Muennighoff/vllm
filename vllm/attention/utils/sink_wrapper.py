# triton_union_wrapper.py
import torch
from typing import List, Tuple
from vllm.attention.ops.triton_fla_sink_sw import union_prompt_sliding_attn_triton

@torch.no_grad()
def gather_dense_kv_from_pages(
    k_pages: torch.Tensor,  # [num_pages, page_size, H, D] or similar
    v_pages: torch.Tensor,
    block_table: torch.Tensor,  # [num_blocks] -> page indices (or token indices)
    head_dim_last: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DEV-ONLY: Gather dense K/V [H, T, D] for a single sequence from paged cache.
    Adapt to your actual KV layout. For a quick hack, if vLLM already exposes
    a dense view in the eager path, use that instead of this.
    """
    # --- Replace with your actual mapping ---
    # Here we assume block_table gives linear token indices [T].
    # If it gives page indices, expand to tokens via (page_idx * page_size + offset)
    tok_idx = block_table.to(torch.long)  # [T]
    # k_pages, v_pages assumed to be [T, H, D] for demo:
    K = k_pages.index_select(dim=0, index=tok_idx)  # [T, H, D]
    V = v_pages.index_select(dim=0, index=tok_idx)  # [T, H, D]
    K = K.transpose(0, 1)  # [H, T, D]
    V = V.transpose(0, 1)
    return K, V


@torch.no_grad()
def union_prompt_sliding_decode_step(
    q: torch.Tensor,             # [B, H, D]
    k_dense: torch.Tensor,       # [B, H, T, D]
    v_dense: torch.Tensor,       # [B, H, T, D]
    prompt_lens: torch.Tensor,   # [B]
    t_positions: torch.Tensor,   # [B]
    window_size: int,
):
    B, H, D = q.shape
    _, _, T, _ = k_dense.shape

    # flatten rows: [B*H, ...]
    q2 = q.reshape(B * H, D).contiguous()
    k2 = k_dense.reshape(B * H, T, D).contiguous()
    v2 = v_dense.reshape(B * H, T, D).contiguous()

    # duplicate per-head metadata
    prompt_lens2 = prompt_lens.repeat_interleave(H)
    t_pos2 = t_positions.repeat_interleave(H)

    o2 = union_prompt_sliding_attn_triton(q2, k2, v2, prompt_lens2, t_pos2, window_size)
    return o2.reshape(B, H, D)
