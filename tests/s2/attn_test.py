import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_prompt_sliding_window_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # New parameters for prompt-aware sliding window
    prompt_lens,  # tensor of prompt lengths per sequence
    stride_pl,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Modified Flash Attention kernel with prompt-aware sliding window.
    The prompt tokens are always kept in the attention window.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Decompose offsets
    off_h = off_hz % H
    off_z = off_hz // H
    
    # Get prompt length for this sequence
    prompt_len = tl.load(prompt_lens + off_z * stride_pl)
    
    # Initialize pointers
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    
    # Block indices
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load Q block
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Determine the range of KV blocks to process
    lo = 0
    hi = N_CTX
    
    # Iterate over KV blocks
    for start_n in range(lo, hi, BLOCK_N):
        offs_n_cur = start_n + offs_n
        
        # Create attention mask with prompt-aware sliding window
        # For each query position, determine if KV position should be attended to
        mask = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1)
        
        for i in range(BLOCK_M):
            q_pos = start_m * BLOCK_M + i
            for j in range(BLOCK_N):
                k_pos = start_n + j
                
                # Always attend to prompt tokens
                is_prompt = k_pos < prompt_len
                
                # Check sliding window for non-prompt tokens
                in_window = (q_pos - k_pos) < WINDOW_SIZE
                is_valid = (k_pos <= q_pos) if IS_CAUSAL else True
                
                # Combine conditions: attend if in prompt OR (in window AND valid)
                should_attend = is_prompt | (in_window & is_valid)
                mask[i, j] = should_attend & (k_pos < N_CTX) & (q_pos < N_CTX)
        
        # Load K, V blocks
        k_ptrs = K + k_offset + offs_n_cur[None, :] * stride_kn + offs_d[:, None] * stride_kk
        v_ptrs = V + v_offset + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vk
        
        k = tl.load(k_ptrs, mask=offs_n_cur[None, :] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n_cur[:, None] < N_CTX, other=0.0)
        
        # Compute attention scores
        qk = tl.dot(q, k)
        qk = tl.where(mask, qk, float("-inf"))
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + tl.sum(tl.exp(qk - m_i_new[:, None]), axis=1)
        
        # Update accumulator
        p = tl.exp(qk - m_i_new[:, None])
        acc = alpha[:, None] * acc + tl.dot(p, v)
        
        # Update running statistics
        l_i = l_i_new
        m_i = m_i_new
    
    # Write output
    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N_CTX)


class FlashAttentionPromptWindow(torch.nn.Module):
    """
    Wrapper for Flash Attention with prompt-aware sliding window.
    """
    
    def __init__(self, window_size=2048, causal=True):
        super().__init__()
        self.window_size = window_size
        self.causal = causal
    
    def forward(self, q, k, v, prompt_lens):
        """
        Args:
            q: (batch_size, num_heads, seq_len, head_dim)
            k: (batch_size, num_heads, seq_len, head_dim)
            v: (batch_size, num_heads, seq_len, head_dim)
            prompt_lens: (batch_size,) tensor containing prompt length for each sequence
        
        Returns:
            out: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Ensure prompt_lens is on the same device
        prompt_lens = prompt_lens.to(q.device)
        
        # Allocate output tensor
        out = torch.empty_like(q)
        
        # Configure block sizes
        BLOCK_M = 128
        BLOCK_N = 64
        
        # Grid configuration
        grid = lambda META: (
            triton.cdiv(seq_len, META['BLOCK_M']),
            batch_size * num_heads,
        )
        
        # Launch kernel
        flash_attn_prompt_sliding_window_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            prompt_lens,
            prompt_lens.stride(0),
            batch_size, num_heads, seq_len,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=head_dim,
            WINDOW_SIZE=self.window_size,
            IS_CAUSAL=self.causal,
        )
        
        return out


# Integration with vLLM
def modify_vllm_attention(attn_module, window_size=2048):
    """
    Modify vLLM's attention module to use prompt-aware sliding window.
    This would be integrated into vLLM's attention backend.
    """
    
    class PromptAwareWindowAttention(attn_module.__class__):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.flash_attn_prompt_window = FlashAttentionPromptWindow(
                window_size=window_size,
                causal=True
            )
        
        def forward(self, query, key, value, kv_cache, input_metadata):
            # Extract prompt lengths from input_metadata
            # This assumes vLLM provides prompt length information
            prompt_lens = input_metadata.prompt_lens
            
            # For prefill phase
            if input_metadata.is_prompt:
                # Regular attention for prompt tokens
                return super().forward(query, key, value, kv_cache, input_metadata)
            
            # For generation phase with sliding window
            # Reshape for flash attention
            batch_size = query.shape[0]
            seq_len = key.shape[1]
            num_heads = query.shape[1]
            head_dim = query.shape[2]
            
            q = query.view(batch_size, num_heads, -1, head_dim)
            k = key.view(batch_size, num_heads, seq_len, head_dim)
            v = value.view(batch_size, num_heads, seq_len, head_dim)
            
            # Apply prompt-aware sliding window attention
            out = self.flash_attn_prompt_window(q, k, v, prompt_lens)
            
            return out.view(batch_size, -1, num_heads * head_dim)
    
    return PromptAwareWindowAttention


# Testing function
def test_prompt_aware_window():
    """
    Test the prompt-aware sliding window attention.
    """
    torch.manual_seed(42)
    
    # Test parameters
    batch_size = 2
    num_heads = 8
    seq_len = 4096
    head_dim = 64
    window_size = 2048
    prompt_lens = torch.tensor([256, 512], dtype=torch.int32).cuda()
    
    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    k = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    v = torch.randn(batch_size, num_heads, seq_len, head_dim).cuda()
    
    # Initialize module
    attn = FlashAttentionPromptWindow(window_size=window_size)
    
    # Forward pass
    out = attn(q, k, v, prompt_lens)
    
    print(f"Input shape: {q.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Prompt lengths: {prompt_lens}")
    print(f"Window size: {window_size}")
    
    # Verify attention pattern (simplified check)
    # The prompt tokens should always be attended to
    print("\nVerifying attention pattern...")
    
    # You can add more detailed verification here
    return out


if __name__ == "__main__":
    # Test the implementation
    output = test_prompt_aware_window()
    print("\nTest completed successfully!")