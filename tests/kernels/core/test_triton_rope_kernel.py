import os
import torch

from vllm.attention.ops.triton_unified_attention import unified_attention


def build_inv_freq(rotary_dim: int, base: float, device: torch.device):
    idx = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (float(base) ** (idx / float(rotary_dim)))
    return inv_freq.contiguous()


def apply_rope_ref(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor):
    # x: [T, H, D]
    # positions: [T]
    # inv_freq: [D/2]
    T, H, D = x.shape
    assert D % 2 == 0
    hd2 = D // 2
    x1 = x[..., :hd2]
    x2 = x[..., hd2:]
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos = torch.cos(angles).to(x.dtype)
    sin = torch.sin(angles).to(x.dtype)
    # Broadcast to heads
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat([y1, y2], dim=-1)


def run_case(device: str = "cuda"):
    torch.manual_seed(0)
    assert torch.cuda.is_available()
    dev = torch.device(device)

    # Problem sizes
    head_size = 64
    rotary_dim = 64
    num_heads = 4
    num_kv_heads = 2
    num_queries_per_kv = num_heads // num_kv_heads
    seq_len = 16
    block_size = seq_len
    base = 10000.0

    # Build raw Q/K/V
    q_raw = torch.randn(1, num_heads, head_size, device=dev, dtype=torch.float16)
    k_tokens_raw = torch.randn(seq_len, num_kv_heads, head_size, device=dev, dtype=torch.float16)
    v_tokens = torch.randn(seq_len, num_kv_heads, head_size, device=dev, dtype=torch.float16)

    # Build caches [num_blks, blk_size, num_kv_heads, head_size]
    num_blks = 1
    k_cache_raw = k_tokens_raw.unsqueeze(0)  # [1, seq_len, n_kv, d]
    v_cache = v_tokens.unsqueeze(0)

    # Metadata
    block_table = torch.tensor([[0]], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    max_seqlen_q = torch.tensor(1, dtype=torch.int32, device=dev)
    max_seqlen_k = torch.tensor(seq_len, dtype=torch.int32, device=dev)

    # Scales
    k_descale = torch.ones((1, num_kv_heads), dtype=torch.float32, device=dev)
    v_descale = torch.ones((1, num_kv_heads), dtype=torch.float32, device=dev)

    # Window (use a large window to include all keys)
    window = (seq_len, 0)
    softcap = 0.0
    sm_scale = 1.0

    # Build reference rotation using absolute positions
    inv_freq = build_inv_freq(rotary_dim, base, dev).to(q_raw.dtype)
    q_ref = apply_rope_ref(q_raw, torch.tensor([seq_len - 1], device=dev, dtype=torch.long), inv_freq)
    k_ref_tokens = apply_rope_ref(k_tokens_raw, torch.arange(seq_len, device=dev, dtype=torch.long), inv_freq)
    k_cache_ref = k_ref_tokens.unsqueeze(0)

    # Path A: External RoPE (rotate Q/K outside, kernel without rope)
    out_ext = torch.empty_like(q_raw)
    unified_attention(
        q=q_ref,
        k=k_cache_ref,
        v=v_cache,
        out=out_ext,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seq_lens,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=sm_scale,
        causal=True,
        window_size=window,
        block_table=block_table,
        softcap=softcap,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        alibi_slopes=None,
        qq_bias=None,
        sinks=None,
        global_lens=None,
        rope_inv_freq=None,
        rotary_dim=None,
    )

    # Path B: In-kernel RoPE (raw Q/K and pass inv_freq)
    out_kernel = torch.empty_like(q_raw)
    unified_attention(
        q=q_raw,
        k=k_cache_raw,
        v=v_cache,
        out=out_kernel,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seq_lens,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=sm_scale,
        causal=True,
        window_size=window,
        block_table=block_table,
        softcap=softcap,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        alibi_slopes=None,
        qq_bias=None,
        sinks=None,
        global_lens=None,
        rope_inv_freq=inv_freq,
        rotary_dim=rotary_dim,
    )

    # Compare
    diff = (out_ext - out_kernel).abs().max().item()
    print("max_diff:", diff)
    assert diff < 7e-3, f"RoPE in-kernel mismatch: max diff {diff}"


if __name__ == "__main__":
    # Ensure we use the Triton attention backend code path in case of side effects
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")
    run_case("cuda")
    print("PASS")


