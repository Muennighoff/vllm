# triton_union_prompt_sw.py
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_decode_union_kernel(
    Q, K, V, O,
    # scalar ints
    T, D, P,  # total seq len T, head_dim D, prompt_len P
    T_DEC,    # current decode position t (0-based)
    W,        # sliding window size
    # strides
    stride_qd, stride_kd, stride_vd, stride_od,
    # constants
    BLOCK_N: tl.constexpr,   # keys per tile
    BLOCK_D: tl.constexpr,   # head dim per tile (must divide D)
):
    # Single program handles one query vector (1 x D) against [0..T)
    # We assume Q points to the query row (shape [D]), O to output row [D].
    # K, V are [T, D].

    # --- compute two ranges
    p_len = P
    t = T_DEC
    w = W

    # prompt region
    startA = 0
    endA = p_len            # [0, P)

    # sliding region, clipped to exclude prompt area
    leftB = t - w + 1
    leftB = tl.maximum(leftB, p_len)
    leftB = tl.maximum(leftB, 0)
    rightB = t + 1          # [leftB, t+1)

    # init streaming softmax state
    NEG_INF = -1.0e9
    m_i = NEG_INF
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # helper to process a tile range [s, e)
    def process_range(s, e, m_i, l_i, acc):
        # iterate in steps of BLOCK_N
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        # load Q (split over BLOCK_D)
        q = tl.load(Q + offs_d * stride_qd, mask=offs_d < D, other=0.0)

        for n_start in range(0, 1_000_000):  # upper bound; we 'break' manually
            n_off = s + n_start * BLOCK_N + offs_n
            mask_n = n_off < e
            if tl.all(~mask_n):
                # no more work
                break
            # [BLOCK_N, BLOCK_D]
            k_tile = tl.load(K + n_off[:, None] * stride_kd + offs_d[None, :] * stride_kd // stride_kd,
                             mask=mask_n[:, None] & (offs_d[None, :] < D),
                             other=0.0)
            # scores = q @ k^T => [BLOCK_N]
            # q: [D], k_tile: [BLOCK_N, D]
            scores = tl.sum(k_tile * q[None, :], axis=1) * (1.0 / math.sqrt(1.0))  # scale will be applied below

            # causal mask: only necessary for range B upper edge, but since we clip rightB=t+1,
            # prompt region is all <= t, so safe. Here we leave as-is.

            # numerically stable streaming softmax with scale 1/sqrt(D)
            # (apply the 1/sqrt(D) scale here for correctness)
            scale = 1.0 / tl.sqrt(tl.float32(D))
            scores = scores * scale

            max_j = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, max_j)

            # exp(scores - m_new)
            scores_exp = tl.exp(scores - m_new)

            # compute l_new
            l_new = tl.exp(m_i - m_new) * l_i + tl.sum(scores_exp, axis=0)

            # load V tile: [BLOCK_N, BLOCK_D]
            v_tile = tl.load(V + n_off[:, None] * stride_vd + offs_d[None, :] * stride_vd // stride_vd,
                             mask=mask_n[:, None] & (offs_d[None, :] < D),
                             other=0.0)

            # acc update:
            acc = acc * (tl.exp(m_i - m_new) * l_i / l_new) + tl.sum((scores_exp[:, None] / l_new) * v_tile, axis=0)

            m_i = m_new
            l_i = l_new
        return m_i, l_i, acc

    # process prompt region
    if endA > startA:
        m_i, l_i, acc = process_range(startA, endA, m_i, l_i, acc)

    # process sliding region
    if rightB > leftB:
        m_i, l_i, acc = process_range(leftB, rightB, m_i, l_i, acc)

    # write output
    tl.store(O + tl.arange(0, BLOCK_D) * stride_od, acc, mask=tl.arange(0, BLOCK_D) < D)


def union_prompt_sliding_attn_triton(
    q: torch.Tensor,   # [B*H, D] or [N, D] queries at decode (N = B*H)
    k: torch.Tensor,   # [B*H, T, D]
    v: torch.Tensor,   # [B*H, T, D]
    prompt_lens: torch.Tensor,  # [B], int32
    t_positions: torch.Tensor,  # [B], int32, current decode positions (0-based)
    window_size: int,
) -> torch.Tensor:
    """
    Computes attention for each (batch, head) row independently over
    [0..prompt_len) union [max(prompt_len, t-W+1)..t].

    Assumes k/v are contiguous per (batch*head) row.
    """
    assert q.ndim == 2 and k.ndim == 3 and v.ndim == 3
    device = q.device
    dtype = q.dtype
    N, D = q.shape
    _, T, Dk = k.shape
    assert D == Dk and v.shape == k.shape

    # For simplicity in this prototype, assume H=1 per sequence row in q/k/v input,
    # i.e., the caller pre-flattened (B*H) as rows and duplicated prompt_lens/t_positions per-head.

    BLOCK_D = 128 if D >= 128 else 64
    BLOCK_N = 128

    o = torch.empty_like(q, dtype=dtype, device=device)

    # Make strides in elements
    stride_qd = q.stride(1)
    stride_kd = k.stride(2)  # step along D in the last dim
    stride_vd = v.stride(2)
    stride_od = o.stride(1)

    # Launch one program per row
    grid = (N,)

    # We need prompt_len / t for each row. For the prototype we launch one-by-one in Python loop
    # to keep the kernel simple. It’s decode-time anyway.
    for i in range(N):
        b = i  # row id
        # In practice, you'll pass per-head prompt_len / t; demo assumes they’re broadcasted already
        P = int(prompt_lens[b].item())
        T_DEC = int(t_positions[b].item())
        triton_kernel = _attn_decode_union_kernel[grid](
            q[b], k[b], v[b], o[b],
            T, D, P, T_DEC, window_size,
            stride_qd, stride_kd, stride_vd, stride_od,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
        )
    return o
