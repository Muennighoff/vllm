# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""
from dataclasses import dataclass
import itertools
import os
from typing import Any,ClassVar, Optional

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.layer import Attention
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.utils.fa_utils import (flash_attn_supports_fp8,
                                           get_flash_attn_version,
                                           is_flash_attn_varlen_func_available)

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (flash_attn_varlen_func,
                                               get_scheduler_metadata,
                                               reshape_and_cache_flash)

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              get_kv_cache_layout)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# NOTE(woosuk): This is an arbitrary number. Tune it if needed.
_DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH = 16


class FlashAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes.")

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Optional aot scheduling
    scheduler_metadata: Optional[torch.Tensor] = None
    prefix_scheduler_metadata: Optional[torch.Tensor] = None
    max_num_splits: int = 0

    causal: bool = True


def _get_sliding_window_configs(
        vllm_config: VllmConfig) -> set[Optional[tuple[int, int]]]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[Optional[tuple[int, int]]] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[FlashAttentionMetadata]):
    attn_cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.NEVER if get_flash_attn_version() == 2 \
        else AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config)
        self.num_heads_kv = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = (get_flash_attn_version() == 3)
        self.use_full_cuda_graph = self.compilation_config.full_cuda_graph
        if self.use_full_cuda_graph:
            if not self.aot_schedule:
                raise ValueError(
                    "AoT scheduling is required for full cuda graph.")
            capture_sizes = self.compilation_config.cudagraph_capture_sizes
            if not capture_sizes:
                raise ValueError(
                    "cudagraph_capture_sizes should not be None when "
                    "full_cuda_graph is True.")
            self.max_cudagraph_size = max(capture_sizes)
            if self.max_cudagraph_size > 992:
                # This condition derives from FA3's internal heuristic.
                # TODO(woosuk): Support larger cudagraph sizes.
                raise ValueError(
                    "Capture size larger than 992 is not supported for "
                    "full cuda graph.")

            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = _DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: Optional[tuple[int, int]] = None

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few 
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = int(common_attn_metadata.seq_lens_cpu.max())
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(
                    self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        def schedule(batch_size, cu_query_lens, max_query_len, seqlens,
                     max_seq_len, causal):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype)
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=self.max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor([0, num_actual_tokens],
                                                dtype=torch.int32,
                                                device=self.device)
            prefix_kv_lens = torch.tensor([common_prefix_len],
                                          dtype=torch.int32,
                                          device=self.device)
            suffix_kv_lens = (seq_lens_cpu[:num_reqs] - common_prefix_len).to(
                self.device, non_blocking=True)
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False)
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=suffix_kv_lens,
                                          max_seq_len=max_seq_len -
                                          common_prefix_len,
                                          causal=True)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=seq_lens,
                                          max_seq_len=max_seq_len,
                                          causal=causal)

        if self.use_full_cuda_graph:
            assert scheduler_metadata is not None
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        max_num_splits = 0
        if (self.use_full_cuda_graph
                and num_actual_tokens <= self.max_cudagraph_size):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal)
        return attn_metadata

    def can_run_in_cudagraph(
            self, common_attn_metadata: CommonAttentionMetadata) -> bool:
        # Full CUDA Graph always supported (FA2 support checked separately)
        return True

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        sinks: Optional[torch.Tensor] = None,
        rope: Optional[Any] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        FlashAttentionBackend.validate_head_size(head_size)

        if attn_type not in [
                AttentionType.DECODER, AttentionType.ENCODER_ONLY
        ]:
            raise NotImplementedError("Encoder/decoder cross-attention "
                                      "is not implemented for "
                                      "FlashAttentionImpl")

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        if is_quantized_kv_cache(self.kv_cache_dtype) \
            and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device.")

        self.sinks = sinks
        if self.sinks is not None:
            assert self.vllm_flash_attn_version == 3, (
                "Sinks are only supported in FlashAttention 3")
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer")

        ### ADDED ###
        self.prompt_slot_end = None
        self.prompt_keys = None
        self.prompt_values = None
        self.rope = rope

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        k_pos: Optional[torch.Tensor] = None,
        gather_index: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlashAttentionImpl")

        if attn_metadata is None:
            # Profiling run.
            return output

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, ):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(query[:num_actual_tokens],
                                                   key[:num_actual_tokens],
                                                   value[:num_actual_tokens],
                                                   output[:num_actual_tokens],
                                                   attn_metadata, layer)

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            
            # NOTE(niklas): places the key and value tensors into the cache
            # i.e. key_cache.sum() == 0 before this on the first entry
            # & after key_cache.sum() == key.sum()
            # The func is implemented in a cuda kernel (void reshape_and_cache_flash())
            # It places it at the attn_metadata.slot_mapping locations as if the
            # 1st dim comes first and has block_size many slots (2nd dim) and then moves to the 
            # 2nd elemnt in 1st dim etc.
            # I.e. if slot_mapping is 16,17... and block_size is 16, then items are at
            # [1, 0], [1, 1], ... [2, 0] ...
            # the cache is different for each layer but slots are same
            # You know when you are in regular generation mode when key.shape[0] == 1
            # i.e. for first pass it is larger due to prompt, then all cached
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape(
                    (num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale)
            query = query.reshape((num_tokens, num_heads, head_size))

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            # seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            # max_seqlen_k = attn_metadata.max_seq_len
            # block_table = attn_metadata.block_table

            # NOTE(niklas): block_table indexes into the key/value cache specifically
            # it is of shape [bs, x] where x is e.g. 2560
            # for each batch element it has indices where its caches are stored in the key/value cache
            # e.g.
            # tensor([[1, 2, 0,  ..., 0, 0, 0],
            #         [1, 3, 0,  ..., 0, 0, 0]], device='cuda:0', dtype=torch.int32)
            # means that for the 2nd batch element it is in the first and third blocks
            # (in the first as it shares the same prompt with 1st batch)
            # (also in third as vLLM only allows prompt sharing in increments of ~16
            # since the prompt is 23 tokens, another 7 are in block 3 (despite also being in block 2)
            # i.e. (key_cache[2] == key_cache[3]).all() == True
            # scheduler_metadata = attn_metadata.scheduler_metadata
            descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

            #if layer.layer_name == "model.layers.0.self_attn.attn":
            #    if seqused_k > 1415:
            #        import pdb; pdb.set_trace() # Check if we are in sliding window mode
                # import pdb; pdb.set_trace() # Check where keys get appended to cache/stored
                # print(cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, self.sliding_window, (key_cache != 0).sum())

            if os.environ.get("SW"):
                # import pdb; pdb.set_trace()  # Check if we are in sliding window mode
                num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
                flat_k_cache = key_cache.reshape(num_blocks * block_size, num_kv_heads, head_size)
                flat_v_cache = value_cache.reshape(num_blocks * block_size, num_kv_heads, head_size)
                # import pdb; pdb.set_trace()  # Check if we are in sliding window mode
                k_compact = flat_k_cache.index_select(0, gather_index).contiguous()   # (total_k, n_kv, d)
                v_compact = flat_v_cache.index_select(0, gather_index).contiguous()
                try:
                    q = self.rope(pos, query[:num_actual_tokens])[0]
                    k_compact = self.rope(k_pos, k_compact)[0]
                except:
                    import pdb; pdb.set_trace()  # Check if we are in sliding window mode

                flash_attn_varlen_func(
                    q=q,
                    k=k_compact,
                    v=v_compact,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    cu_seqlens_k=cu_seqlens_k,     # varlen path
                    seqused_k=None,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=None,             # FA3 requires None
                    window_size=(-1, -1),          # we preselected keys
                    block_table=None,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=None,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )

            # === Config ===
            # sliding window: number of left tokens to keep from the tail (env overrides)
            elif (sw := os.environ.get("SW")):
                # import pdb; pdb.set_trace()  # Check if we are in sliding window mode
                sw = int(sw)

                # Prompt tokens to keep if provided via env; otherwise keep full prompt on first pass
                prompt_toks_env = os.environ.get("PROMPTTOKS")
                prompt_toks_env = int(prompt_toks_env) if prompt_toks_env is not None else None

                # === Flatten paged KV cache to slot-major rows ===
                num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
                flat_k_cache = key_cache.reshape(num_blocks * block_size, num_kv_heads, head_size)
                flat_v_cache = value_cache.reshape(num_blocks * block_size, num_kv_heads, head_size)

                # === Metadata / containers ===
                seq_lens = attn_metadata.seq_lens
                B = seq_lens.shape[0]
                chosen_per_seq = []   # list of 1-D long tensors (slot indices) per sequence in batch
                lengths = []          # lengths per sequence
                # For first-pass detection: store per-sequence prompt slots on self.prompt_slots (list of tensors)
                is_first_pass = (not hasattr(self, "prompt_slots")) or (max_seqlen_q > 1)
                device = flat_k_cache.device
                block_size_arange = torch.arange(block_size, device=device, dtype=torch.long).unsqueeze(0) # (block_size,)

                for b in range(B):
                    if (L := int(seq_lens[b])) <= 0:
                        print("WARNING: Sequence length is 0 for batch element", b)
                        chosen_per_seq.append(torch.empty(0, dtype=torch.long, device=device))
                        lengths.append(0)
                        continue

                    if (num_blocks_used := (L + block_size - 1) // block_size) == 0:
                        print("WARNING: No blocks used for sequence", b)
                        slots_b = torch.empty(0, dtype=torch.long, device=device)
                    else:
                        # slice first num_blocks_used block ids
                        block_ids = block_table[b, :num_blocks_used]  # (num_blocks_used,)
                        # compute base indices for each block: (num_blocks_used, 1)
                        # & expand with arange: (num_blocks_used, block_size)
                        # then reshape into a flat list of slots & take exactly L of them
                        slots_b = ((block_ids * block_size).unsqueeze(1) + block_size_arange).reshape(-1)[:L]
                    
                    # === Decide what to keep for this sequence: prompt (first pass) or stored prompt + window (later) ===
                    if is_first_pass:
                        # Take prompt_end = PROMPTTOKS + 1 if provided, else full prompt length L
                        prompt_end = prompt_toks_env + 1 if prompt_toks_env is not None else L
                        prompt_end = min(prompt_end, L)
                        # prefix = first prompt_end tokens
                        prefix = slots_b[:prompt_end]
                        chosen = torch.cat([prefix, slots_b[prompt_end:][-sw:]], dim=0)
                        # store the prompt prefix for future calls (one entry per batch element)
                        if (not hasattr(self, "prompt_slots")) or (B > len(self.prompt_slots)):
                            self.prompt_slots = [None] * B
                        self.prompt_slots[b] = prefix.detach().clone()
                    else:
                        # Subsequent passes: reuse stored prefix for this sequence (if any)
                        if hasattr(self, "prompt_slots") and self.prompt_slots is not None:
                            prefix = self.prompt_slots[b]
                        else:
                            print("WARNING: No stored prefix for sequence", b, "falling back to no prefix")
                            # Fall back to keeping nothing of the prefix (unlikely)
                            prefix = torch.empty(0, dtype=torch.long, device=device)

                        tail = slots_b[-sw:]
                        chosen = torch.cat([prefix, tail[~torch.isin(tail, prefix)]], dim=0)

                    lengths.append(int(chosen.numel()))
                    # If chosen ended up empty but the sequence has tokens, ensure we keep at least the last token
                    if lengths[-1] == 0 and slots_b.numel() > 0:
                        print("WARNING: Chosen tokens is empty for sequence", b, "falling back to last token")
                        chosen = slots_b[-1:].clone()
                    chosen_per_seq.append(chosen)

                # === Build compact varlen K/V: concatenate chosen rows and form cu_seqlens_k ===
                # gather_index is total_k long
                gather_index = torch.cat(chosen_per_seq, dim=0)    # 1D long tensor
                k_compact = flat_k_cache.index_select(0, gather_index).contiguous()   # (total_k, n_kv, d)
                v_compact = flat_v_cache.index_select(0, gather_index).contiguous()

                # Build cu_seqlens_k from lengths (int32)
                cu = [0] + list(itertools.accumulate(lengths))
                cu_seqlens_k = torch.tensor(cu, dtype=torch.int32, device=flat_k_cache.device)
                max_seqlen_k = max(lengths)

                # Apply RoPE
                if max_seqlen_q == 1: # Clamp pos to avoid growing larger than full window
                    pos = torch.tensor(lengths).to(device) - 1
                q = self.rope(pos, query[:num_actual_tokens])[0]
                k_pos = torch.cat([torch.arange(l) for l in lengths])
                k_compact = self.rope(k_pos.to(device), k_compact)[0]

                # === Call FA3 in varlen mode (we disable kernel SW; FA3 needs alibi=None) ===
                flash_attn_varlen_func(
                    q=q,
                    k=k_compact,
                    v=v_compact,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=max_seqlen_q,
                    cu_seqlens_k=cu_seqlens_k,     # varlen path
                    seqused_k=None,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=None,             # FA3 requires None
                    window_size=(-1, -1),          # we preselected keys
                    block_table=None,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=None,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )

            # # Store kv cache of prompt on first pass
            # if (sliding_window := os.environ.get("SLIDING_WINDOW")) is not None:
            #     sliding_window = int(sliding_window)
            #     if (self.prompt_keys is None) or (max_seqlen_q > 1):
            #         promptidx = int(os.environ.get("PROMPTTOKS", attn_metadata.slot_mapping.shape[0])) - 1
            #         # If prompt is longer than sliding window:
            #         # RuntimeError: k must have shape (batch_size_k, seqlen_k, num_heads_k, head_size)
            #         # Below handles this via a promptidx to pay attention to all before it & last sliding window at the end
            #         slots = torch.cat(
            #             (attn_metadata.slot_mapping[:promptidx + 1], attn_metadata.slot_mapping[promptidx + 1:][-sliding_window:]),
            #             dim=0
            #         )
            #         flat_k_cache = flat_k_cache[slots].unsqueeze(0)
            #         flat_v_cache = flat_v_cache[slots].unsqueeze(0)

            #         try:
            #             self.prompt_keys = flat_k_cache[:,:slots[promptidx].item() + 1].clone()
            #         except:
            #             if layer.layer_name == "model.layers.0.self_attn.attn":
            #                 import pdb; pdb.set_trace() # Check if we are in sliding window mode
            #         self.prompt_values = flat_v_cache[:,:slots[promptidx].item() + 1].clone()
            #         self.prompt_slot_end = slots[promptidx].item()
            #     else:
            #         cur_slot = attn_metadata.slot_mapping.item()
            #         window_start = max(0, cur_slot - sliding_window)
            #         slots = torch.arange(
            #             max(self.prompt_slot_end, window_start) + 1, cur_slot + 1,
            #             device=attn_metadata.slot_mapping.device
            #         )
            #         flat_k_cache = torch.cat((self.prompt_keys, flat_k_cache[slots].unsqueeze(0)), dim=1)
            #         flat_v_cache = torch.cat((self.prompt_values, flat_v_cache[slots].unsqueeze(0)), dim=1)

            #     max_seqlen_k = flat_k_cache.shape[1]

            # # NOTE(niklas) in flash_attn block -> page
            # # https://github.com/Dao-AILab/flash-attention/blob/a1c2e22817960fd68933d46747db39d930ac2c8f/flash_attn/cute/interface.py#L102C32-L102C52
            
            # # Options for passing (https://github.com/Dao-AILab/flash-attention/blob/a1c2e22817960fd68933d46747db39d930ac2c8f/flash_attn/cute/interface.py#L97):
            # # 1) Blocks / Pages (default)
            # # key_cache = torch.Size([36500, 16, 8, 128])
            # # 2) Batches (vanilla attention e.g. in transformers etc)
            # # key_cache = (batch_size, seqlen_k, num_head_kv, head_dim)
            # # 3) Varlen (requires cu_seqlens_k)
            # # key_cache = (seqlen_k, num_head_kv, head_dim)
            # flash_attn_varlen_func(
            #     q=query[:num_actual_tokens],
            #     k=flat_k_cache, # key_cache, # torch.Size([36500, 16, 8, 128]) # num_blocks, block_size, num_kv_heads, head_size
            #     # Can also pass as (batch_size, seqlen_k, num_head_kv, head_dim)
            #     v=flat_v_cache, # value_cache, # torch.Size([36500, 16, 8, 128])
            #     out=output[:num_actual_tokens],
            #     cu_seqlens_q=cu_seqlens_q,
            #     max_seqlen_q=max_seqlen_q,
            #     seqused_k=None, # seqused_k,
            #     max_seqlen_k=max_seqlen_k,
            #     softmax_scale=self.scale,
            #     causal=attn_metadata.causal,
            #     alibi_slopes=self.alibi_slopes,
            #     window_size=(-1, -1), # self.sliding_window, # (3, 0)
            #     block_table=None, # block_table, # torch.Size([1, 2560]); [1, 2, 0, 0 ... 0]
            #     softcap=self.logits_soft_cap,
            #     scheduler_metadata=None, # scheduler_metadata,
            #     fa_version=self.vllm_flash_attn_version,
            #     q_descale=layer._q_scale.expand(descale_shape),
            #     k_descale=layer._k_scale.expand(descale_shape),
            #     v_descale=layer._v_scale.expand(descale_shape),
            #     num_splits=attn_metadata.max_num_splits,
            #     s_aux=self.sinks,
            # )
            else:
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=self.sliding_window,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )

            return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention")

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads)

        # Call flash attention directly on Q, K, V tensors
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (num_queries_per_kv > 1 and not use_sliding_window
                          and not use_alibi and np.all(query_lens == 1))
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (num_reqs * num_kv_heads *
                           cdiv(num_queries_per_kv, q_tile_size))
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    fa_version: int,
    prefix_scheduler_metadata: Optional[torch.Tensor] = None,
    suffix_scheduler_metadata: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert alibi_slopes is None, ("Cascade attention does not support ALiBi.")
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window.")

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=sliding_window,
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape)
        if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape)
        if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape)
        if v_descale is not None else None,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=sliding_window,
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape)
        if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape)
        if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape)
        if v_descale is not None else None,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                      suffix_lse)
