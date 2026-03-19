# coding=utf-8
# Copyright 2018 HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMa model."""

from typing import Dict, List, Optional, Union

import torch
from flash_attn import bert_padding
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
)
from torch import nn
from torch.utils.checkpoint import CheckpointFunction

# Import FlexAttention for block diffusion
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
FLEX_ATTENTION_AVAILABLE = True


from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, LlamaConfig, ParallelismArgs
from nanotron.config.models_config import RandomInit, SpectralMupInit, DiffusionArgs
from nanotron.block_diffusion import (
    transition,
    FLEX_ATTENTION_AVAILABLE
)
from nanotron.generation.generate_store import AttachableStore
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.nn.activations import ACT2FN
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from nanotron.random import RandomStates
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator
from nanotron.utils import checkpoint_method

logger = logging.get_logger(__name__)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, end: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.end = end
        self.theta = theta
        # TODO @nouamane: Figure out why we can't set `DTypeInvariantTensor` ...
        # TODO @thomasw21: Complex buffers break DDP, instead we store float and view them as complex
        self.freqs_cis: torch.Tensor
        self._initialized_buffer = False

    def init_rotary_embeddings(self):
        if self._initialized_buffer is True:
            # Buffer if already initialized
            return
        self.register_buffer(
            "freqs_cis",
            torch.empty(self.end, self.dim // 2, 2, dtype=torch.float, device="cuda"),
            persistent=False,
        )
        assert self.freqs_cis.device.type == "cuda"
        # TODO @nouamane: One we figure out how to do the DTypeInvariantTensor, this can be removed and changed to an assert
        if self.freqs_cis.dtype != torch.float:
            self.freqs_cis = self.freqs_cis.to(torch.float)
        assert self.freqs_cis.dtype == torch.float
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device="cpu")[: (self.dim // 2)] / self.dim)
        ).to(
            "cuda"
        )  # should be computed on CPU, otherwise different results with Transformers.
        t = torch.arange(self.end, device="cuda")
        freqs = torch.outer(t, freqs).float()
        complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = torch.view_as_real(complex_freqs)
        self.freqs_cis.copy_(freqs)
        self._initialized_buffer = True

    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_length, num_heads, d_qk]
        position_ids: Optional[torch.LongTensor],  # [batch_size, seq_length]
    ):
        batch_size, seq_length, num_heads, inner_dim = x.shape
        while (
            position_ids is not None and position_ids[-1, -1] >= self.end
        ) or seq_length >= self.end:  # TODO @nouamane: check if this causes cpu-gpu sync
            self.end *= 2
            self._initialized_buffer = False
        if self._initialized_buffer is False:
            print(f"Initializing rotary embeddings with end={self.end}")
            self.init_rotary_embeddings()
        dtype = x.dtype
        assert inner_dim % 2 == 0
        x = x.view(
            batch_size, seq_length, num_heads, inner_dim // 2, 2
        )  # [batch_size, q_length, num_heads, inner_dim]
        if x.dtype == torch.bfloat16:
            x = x.float()
        complex_x = torch.view_as_complex(x)  # [batch_size, q_length, num_heads, inner_dim // 2]
        if position_ids is None:
            freqs_cis = self.freqs_cis[None, :seq_length, None, :]
        else:
            # TODO(kunhao): Should None follow the num_heads dimension?
            if position_ids[-1, -1] < 0 or position_ids[-1, -1] >= self.end:  # Quick test hopefully
                raise ValueError(f"Position ids must be in the range [0, {self.end}), but got {position_ids}")
            freqs_cis = self.freqs_cis[position_ids][:, :, None, :]
        complex_freqs = torch.view_as_complex(freqs_cis)
        x_out = torch.view_as_real(complex_x * complex_freqs).view(batch_size, seq_length, num_heads, inner_dim)
        return x_out.type(dtype)


## Copy from transformers. Non interleaved version of RoPE. Will be refactored later
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, end: int, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.end = end
        self.theta = theta
        self.init_rotary_embeddings()

    def init_rotary_embeddings(self):
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device="cpu") / self.dim)
        )  # important to compute on CPU
        self.register_buffer(
            "inv_freq", torch.empty(self.dim // 2, dtype=torch.float, device="cuda"), persistent=False
        )
        self.inv_freq = self.inv_freq.to(
            torch.float
        )  # make it float32 before copy to avoid precision loss during copy_
        self.inv_freq.copy_(inv_freq)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,  # [batch_size, seq_length, num_heads, d_qk]
        position_ids: Optional[torch.LongTensor],  # [batch_size, seq_length]
    ):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=2):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class GLUActivation(nn.Module):
    def __init__(self, act_fn_name: str):
        super().__init__()
        self.act = ACT2FN[act_fn_name]

    def forward(self, merged_states: torch.Tensor):
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        return self.act(gate_states) * up_states


class MLP(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        gate_up_contiguous_chunks = (
            config.intermediate_size,  # shape of gate_linear
            config.intermediate_size,  # shape of up_linear
        )
        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=gate_up_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.down_proj = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        self.split_silu_mul = GLUActivation(config.hidden_act)

    def forward(self, hidden_states):  # [seq_length, batch_size, hidden_dim]
        merged_states = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(self.split_silu_mul(merged_states))
        return {"hidden_states": hidden_states}


class CoreAttention(nn.Module):
    def __init__(self, config: LlamaConfig, parallel_config: Optional[ParallelismArgs], layer_idx: int):
        super().__init__()
        # TODO @thomasw21: GPT has a weird `d_kv` config which I'm guessing is essentically a `d_qkv`
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), f"Hidden size {config.hidden_size} must be divisible by number of attention heads {config.num_attention_heads}."
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.is_using_mup = config.is_using_mup

        self.checkpoint_attention = False  # Because flash_attn already does checkpointing

    @checkpoint_method(attr_name="checkpoint_attention")
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_sequence_mask: torch.Tensor,
        kv_sequence_mask: torch.Tensor,
        block_mask=None,
        diffusion_mode: Optional[torch.Tensor] = None,
    ):
        if diffusion_mode is not None and diffusion_mode.item() == 2:
            if not FLEX_ATTENTION_AVAILABLE:
                raise RuntimeError("FlexAttention not available. Requires PyTorch 2.5+")
            
            if block_mask is None:
                raise RuntimeError("block_mask is None but diffusion_mode=2")
            
            q = query_states.transpose(1, 2)
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
            
            attn_output = flex_attention(q, k, v, block_mask=block_mask)
            
            return attn_output.transpose(1, 2)

        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        # TODO @thomasw21: Compute once, instead of computing for each layers.
        cu_seqlens_q = torch.zeros((q_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
        cu_seqlens_k = torch.zeros((kv_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
        torch.cumsum(q_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_q[1:])
        torch.cumsum(kv_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_k[1:])

        # TODO(kunhao): flash attn's causal means that the query can only attend to the keys before it. This is not
        # what we want if we are using kv cache. This is a hack as we always have q_length == 1 when using kv cache.
        if diffusion_mode is not None and diffusion_mode.item() == 1:
            causal = False
            print("DEBUG: FlashAttention - causal=False (diffusion_mode=1)")
        else:
            causal = False if q_sequence_mask.shape[1] == 1 else True

        # NOTE: this scale is for µTransfer,
        # in SP, we use sqrt(1/d_h)
        softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_sequence_mask.shape[1],
            max_seqlen_k=kv_sequence_mask.shape[1],
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=False,
        )

        return attn_output


def pad_to_right(tensor, mask, new_tensor=None):
    """Transform a left-padded tensor into a right-padded tensor. (Useful for prefilling key/value states)
    Args:
        tensor: (batch_size, seqlen, d1, d2)
        mask: (batch_size, seqlen)
        new_tensor: (batch_size, new_tensor_seqlen, d1, d2)
    Returns:
        new_tensor: (batch_size, new_tensor_seqlen, d1, d2)
        right_padded_mask: (batch_size, seqlen)
    """
    # First, we need to find the number of padding for each row
    unpad_seqlens = mask.sum(1)
    # Then, we need to find the maximum length of the tensor
    max_seqlen = mask.shape[1]
    # We can then create the indices to select the padded values
    # The indices are the same for each row
    indices = torch.arange(max_seqlen, device=mask.device)
    # We can then create the mask for the padded values
    right_padded_mask = indices < unpad_seqlens[:, None]
    # We select the useful values
    useful_values = tensor[mask]
    # We create the new tensor (if not provided)
    new_tensor = torch.zeros_like(tensor) if new_tensor is None else new_tensor
    # We fill the new tensor with the useful values
    new_tensor[:, : right_padded_mask.shape[1], :, :][right_padded_mask] = useful_values
    return new_tensor, right_padded_mask


class CausalSelfAttention(nn.Module, AttachableStore):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

        super().__init__()
        # Tensor parallel considerations: We split tensors along head dimension
        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        try:
            assert (
                config.num_key_value_heads % tp_pg.size() == 0
            ), f"Number of key/value heads ({config.num_key_value_heads}) must be divisible by TP size ({tp_pg.size()})."
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            # If num_key_value_heads is not defined, we assume that it is equal to num_attention_heads
            config.num_key_value_heads = config.num_attention_heads
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = config.num_attention_heads != config.num_key_value_heads  # Whether we are using GQA or not
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        self.is_using_mup = config.is_using_mup

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        # build the slice config for self.qkv for save/load
        # shard are done within the contiguous chunk
        qkv_contiguous_chunks = (
            config.num_attention_heads * self.d_qk,  # shape of q
            config.num_key_value_heads * self.d_qk,  # shape of k
            config.num_key_value_heads * self.d_qk,  # shape of v
        )
        self.qkv_proj = TensorParallelColumnLinear(
            self.d_model,
            config.num_attention_heads * self.d_qk + 2 * config.num_key_value_heads * self.d_qk,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=qkv_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        # TODO(kunhao): We want to have only one version per device and not one version per layer.
        if config.rope_interleaved:
            self.rotary_embedding = RotaryEmbedding(
                dim=self.d_qk,
                end=config.max_position_embeddings,
                theta=config.rope_theta,
            )
        else:
            self.rotary_embedding = LlamaRotaryEmbedding(
                dim=self.d_qk,
                end=config.max_position_embeddings,
                theta=config.rope_theta,
            )
        self.rope_interleaved = config.rope_interleaved

        # NOTE: Only supported for training (TODO(fmom): position_ids not supported yet)
        self.flash_rotary_embedding = FlashRotaryEmbedding(
            dim=self.d_qk, base=config.rope_theta, interleaved=config.rope_interleaved
        )

        self.o_proj = TensorParallelRowLinear(
            config.num_attention_heads * self.d_qk,
            self.d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )

        self.attention = CoreAttention(
            config,
            parallel_config=parallel_config,
            layer_idx=layer_idx,
        )

        self.prefill_kv_len = (
            config.max_position_embeddings
        )  # TODO @nouamane: compute based on free memory, because in rope we can surpass max_position_embeddings

    def forward(
        self,
        hidden_states,  # [seq_length, batch_size, hidden_size]
        sequence_mask,  # [batch_size, seq_length]
        position_ids: Optional[torch.Tensor] = None,
        mask_positions: Optional[torch.Tensor] = None,
        noise_weights: Optional[torch.Tensor] = None,
        block_mask: Optional[torch.Tensor] = None,
        diffusion_mode: Optional[torch.Tensor] = None,
    ):
        qkv_states = self.qkv_proj(
            hidden_states
        )  # [seq_length, batch_size, n_local_q_heads * d_qk + 2 * n_local_kv_heads * d_qk]
        q_length, batch_size, _ = qkv_states.shape

        # Split QKV states
        if self.is_gqa:
            query_states, key_states, value_states = torch.split(
                qkv_states,
                [
                    self.n_local_q_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                ],
                dim=-1,
            )

            query_states = (
                query_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
            )
            key_states = (
                key_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
            )
            value_states = (
                value_states.transpose(0, 1).contiguous().view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
            )
        else:
            query_states, key_states, value_states = (
                qkv_states.view(q_length, batch_size, 3, self.n_local_q_heads, self.d_qk)
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )  # [3, batch_size, seq_length, n_local_q_heads, d_qk]

        store = self.get_local_store()
        if store is not None:  # Inference case
            return self._forward_inference(
                query_states, key_states, value_states, sequence_mask, batch_size, q_length, store
            )
        else:  # Training case
            return self._forward_training(
                query_states, key_states, value_states, sequence_mask, batch_size, q_length,
                position_ids, mask_positions, noise_weights, block_mask, diffusion_mode
            )

    def _forward_inference(self, query_states, key_states, value_states, sequence_mask, batch_size, q_length, store):
        # ... (no change to inference)
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache

        assert key_states.requires_grad is False
        assert value_states.requires_grad is False

        if "position_offsets" in store:
            old_position_offsets = store["position_offsets"]
            position_ids = old_position_offsets[:, None] + sequence_mask
        else:
            position_ids = torch.cumsum(sequence_mask, dim=-1, dtype=torch.int32) - 1
        position_offsets = position_ids[:, -1]

        # Compute rotary embeddings
        # Note: keep track of old rotary embedding end to check if we need to enlarge k_cache and v_cache
        old_rotary_embed_end = self.rotary_embedding.end
        if self.rope_interleaved:
            query_states = self.rotary_embedding(query_states, position_ids=position_ids)
            key_states = self.rotary_embedding(key_states, position_ids=position_ids)
        else:
            cos, sin = self.rotary_embedding(value_states, position_ids)
            query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Compute rotary embeddings
            # Note: keep track of old rotary embedding end to check if we need to enlarge k_cache and v_cache
            old_rotary_embed_end = self.rotary_embedding.end
            # interleaved version.
            if self.rope_interleaved:
                query_states = self.rotary_embedding(query_states, position_ids=position_ids)
                key_states = self.rotary_embedding(key_states, position_ids=position_ids)
            # non interleaved version.
            else:
                cos, sin = self.rotary_embedding(value_states, position_ids)
                query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

            if "key" not in store:
                # First inference iteration (Prefill)
                # TODO @nouamane: support custom masking
                # assert that [ False, False, False, False,  True,  True,  True,  True,  True,  True] is accepted
                # but [ False, False, False, False,  True,  True,  False,  False,  True,  True] is not (can't mask in the middle of sequence)
                assert ~(
                    sequence_mask[:, :-1] & (~sequence_mask[:, 1:])  # True is never followed by False
                ).any(), "Can't mask in the middle of sequence, please make sure that pads are at the left of the sequence if existing"

                # preallocate k_cache, v_cache to self.prefill_kv_len
                k_cache = torch.zeros(
                    (
                        batch_size,
                        self.prefill_kv_len,
                        self.n_local_kv_heads,
                        self.d_qk,
                    ),
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                v_cache = torch.zeros(
                    (batch_size, self.prefill_kv_len, self.n_local_kv_heads, self.d_v),
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                # Remove pad tokens from key_states and concatenate samples in key_unpad
                # cu_seqlens_k is the cumulative sequence lengths of key_states
                (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = bert_padding.unpad_input(
                    query_states,
                    sequence_mask,
                )
                (key_unpad, indices_k, cu_seqlens_k, max_seqlen_k) = bert_padding.unpad_input(
                    key_states, sequence_mask
                )
                (value_unpad, _, _, _) = bert_padding.unpad_input(value_states, sequence_mask)

                # NOTE: this scale is for µTransfer,
                # in SP, we use sqrt(1/d_h)
                softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
                output_unpad = flash_attn_varlen_func(
                    q=query_unpad,  # (total_q, n_local_q_heads, d_qk)
                    k=key_unpad,  # (total_kv, n_local_kv_heads, d_qk)
                    v=value_unpad,  # (total_kv, n_local_kv_heads, d_v)
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=True,  # True in prefill phase, False in subsequent phases
                    return_attn_probs=False,
                )  # (total_unpadded, n_local_q_heads, d_v)

                attention_output = bert_padding.pad_input(
                    output_unpad, indices_q, batch_size, q_length
                )  # (batch_size, q_length, n_local_q_heads, d_v)

                pad_to_right(key_states, sequence_mask, new_tensor=k_cache)
                pad_to_right(value_states, sequence_mask, new_tensor=v_cache)

            else:
                # Pull pre-computed key/value states
                # Subsequent inference iterations (q_length=1)
                k_cache = store["key"]
                v_cache = store["value"]

                # NOTE(fmom): According to flash_attn_with_kvcache, "If you pass in k / v, you must make sure that the cache is large enough to hold the new values"
                # Since rotary embedding has changed (to enable larger context), we need to enlarge k_cache and v_cache
                if self.rotary_embedding.end > old_rotary_embed_end:
                    k_cache = torch.cat(
                        [
                            k_cache,
                            torch.zeros(
                                (
                                    batch_size,
                                    self.rotary_embedding.end - old_rotary_embed_end,
                                    self.n_local_kv_heads,
                                    self.d_qk,
                                ),
                                dtype=query_states.dtype,
                                device=query_states.device,
                            ),
                        ],
                        dim=1,
                    )

                    v_cache = torch.cat(
                        [
                            v_cache,
                            torch.zeros(
                                (
                                    batch_size,
                                    self.rotary_embedding.end - old_rotary_embed_end,
                                    self.n_local_kv_heads,
                                    self.d_v,
                                ),
                                dtype=query_states.dtype,
                                device=query_states.device,
                            ),
                        ],
                        dim=1,
                    )

                assert (
                    k_cache.shape[1] == self.rotary_embedding.end
                ), f"Cache size {k_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"
                assert (
                    v_cache.shape[1] == self.rotary_embedding.end
                ), f"Cache size {v_cache.shape[1]} is smaller than rotary embedding end {self.rotary_embedding.end}"

                # [batch_size, seq_length, num_heads, d_qk]
                query_states = query_states.view(
                    batch_size, q_length, self.n_local_q_heads, self.d_qk
                )  # [batch_size, q_length, self.n_heads, d_qk]
                kv_length = key_states.shape[1]
                key_states = key_states.view(
                    batch_size, kv_length, self.n_local_kv_heads, self.d_qk
                )  # [batch_size, kv_length, self.n_heads, d_qk]
                value_states = value_states.view(
                    batch_size, kv_length, self.n_local_kv_heads, self.d_v
                )  # [batch_size, kv_length, self.n_heads, d_v]

                # NOTE: this scale is for µTransfer,
                # in SP, we use sqrt(1/d_h)
                softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
                attention_output = flash_attn_with_kvcache(
                    query_states,
                    k_cache,
                    v_cache,
                    key_states,
                    value_states,
                    rotary_cos=None,
                    rotary_sin=None,
                    # TODO @nouamane: seems like this doesn't help to indicate padding in (for first iteration it's just 0)
                    cache_seqlens=position_offsets.contiguous(),
                    softmax_scale=softmax_scale,
                    causal=True,
                    rotary_interleaved=False,  # the value is not used unless rotary_cos/sin is provided. https://github.com/Dao-AILab/flash-attention
                )

            store.update(
                {
                    "key": k_cache,  # flash-attn has updated with new key_states using cache_seqlens
                    "value": v_cache,
                    "position_offsets": position_offsets,
                }
            )

        attention_output = (
            attention_output.contiguous().view(batch_size, q_length, self.n_local_q_heads * self.d_v).transpose(0, 1)
        )
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}

    def _forward_training(self, query_states, key_states, value_states, sequence_mask, batch_size, q_length, 
                          position_ids, mask_positions, noise_weights, block_mask, diffusion_mode):
        # Apply rotary embeddings to query/key states
        key_value_states = torch.cat([key_states.unsqueeze(0), value_states.unsqueeze(0)], dim=0)
        key_value_states = key_value_states.permute(1, 2, 0, 3, 4).contiguous()
        
        query_states, key_value_states = self.flash_rotary_embedding(
            query_states, kv=key_value_states, position_ids=position_ids
        )
        
        key_states, value_states = torch.split(key_value_states, 1, dim=2)

        q_sequence_mask = sequence_mask
        kv_sequence_mask = sequence_mask

        kv_length = key_states.shape[1]
        
        # Check diffusion mode (2 = Block Diffusion)
        if diffusion_mode is not None and diffusion_mode.item() == 2:
            # Block Diffusion: pass tensors to CoreAttention without reshaping
            attention_output = self.attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                q_sequence_mask=q_sequence_mask,
                kv_sequence_mask=kv_sequence_mask,
                block_mask=block_mask,
                diffusion_mode=diffusion_mode
            )
        else:
            # Standard path
            query_states = query_states.view(
                batch_size * q_length, self.n_local_q_heads, self.d_qk
            )
            key_states = key_states.view(
                batch_size * kv_length, self.n_local_kv_heads, self.d_qk
            )
            value_states = value_states.view(
                batch_size * kv_length, self.n_local_kv_heads, self.d_v
            )

            attention_output = self.attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                q_sequence_mask=q_sequence_mask,
                kv_sequence_mask=kv_sequence_mask,
            )

        attention_output = (
            attention_output.contiguous().view(batch_size, q_length, self.n_local_q_heads * self.d_v).transpose(0, 1)
        )
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )

        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg)

        self.recompute_layer = parallel_config.recompute_layer

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
        mask_positions: Union[torch.Tensor, TensorPointer],
        noise_weights: Union[torch.Tensor, TensorPointer],
        block_mask: Union[torch.Tensor, TensorPointer],
        diffusion_mode: Union[torch.Tensor, TensorPointer],
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(
            hidden_states=hidden_states, 
            sequence_mask=sequence_mask, 
            position_ids=position_ids,
            mask_positions=mask_positions,
            noise_weights=noise_weights,
            block_mask=block_mask,
            diffusion_mode=diffusion_mode
        )
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
        hidden_states = hidden_states + residual

        return hidden_states, output["sequence_mask"], position_ids, mask_positions, noise_weights, block_mask, diffusion_mode

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        sequence_mask: torch.Tensor,
        position_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        noise_weights: torch.Tensor,
        block_mask: torch.Tensor,
        diffusion_mode: torch.Tensor,
    ) -> List[torch.Tensor]:
        return CheckpointFunction.apply(self._core_forward, True, hidden_states, sequence_mask, position_ids, mask_positions, noise_weights, block_mask, diffusion_mode)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
        position_ids: Union[torch.Tensor, TensorPointer],
        mask_positions: Union[torch.Tensor, TensorPointer],
        noise_weights: Union[torch.Tensor, TensorPointer],
        block_mask: Union[torch.Tensor, TensorPointer],
        diffusion_mode: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:

        if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
            hidden_states, sequence_mask, position_ids, mask_positions, noise_weights, block_mask, diffusion_mode = self._checkpointed_forward(
                hidden_states, sequence_mask, position_ids, mask_positions, noise_weights, block_mask, diffusion_mode
            )
        else:
            hidden_states, sequence_mask, position_ids, mask_positions, noise_weights, block_mask, diffusion_mode = self._core_forward(
                hidden_states, sequence_mask, position_ids, mask_positions, noise_weights, block_mask, diffusion_mode
            )

        return {
            "hidden_states": hidden_states,
            "sequence_mask": sequence_mask,
            "position_ids": position_ids,
            "mask_positions": mask_positions,
            "noise_weights": noise_weights,
            "block_mask": block_mask,
            "diffusion_mode": diffusion_mode,
        }


class DiffusionModule(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self._debug_counter = 0  # Counter for debug logging

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor):
        device = input_ids.device
        
        # Default outputs (dummies for P2P)
        noisy_input_ids = input_ids
        input_mask_updated = input_mask
        position_ids = torch.empty(0, device=device)
        mask_positions = torch.empty(0, device=device)
        noise_weights = torch.empty(0, device=device)
        doc_ids = torch.empty(0, device=device)
        block_ids = torch.empty(0, device=device)
        block_mask = None  # FlexAttention BlockMask
        diffusion_mode = torch.tensor(0, dtype=torch.int, device=device)

        if self.config.diffusion_config is not None:
            diff_config = self.config.diffusion_config
            batch_size, seq_len = input_ids.shape
            
            # Debug: Check for diffusion trigger
            if torch.distributed.get_rank() == 0:
                print(f"DEBUG: DiffusionModule - config present. Diffusion: {diff_config.diffusion}, Block: {diff_config.block_diffusion}", flush=True)

            if diff_config.block_diffusion:
                diffusion_mode = torch.tensor(2, dtype=torch.int, device=device)
                
                block_size = diff_config.block_size
                mask_token_id = diff_config.mask_token_id
                t_lower = diff_config.t_lower
                t_upper = diff_config.t_upper
                bos_token_id = self.config.bos_token_id
                eos_token_id = self.config.eos_token_id
                
                # 1. Calculate block IDs
                block_ids, pos_ids, d_ids = calculate_block_ids(
                    input_ids, block_size, bos_token_id, eos_token_id
                )
                doc_ids = d_ids
                
                # 1.5. Create FlexAttention block mask (requires batch_size=1)
                if batch_size != 1:
                    raise RuntimeError(f"FlexAttention requires batch_size=1, got {batch_size}")
                if FLEX_ATTENTION_AVAILABLE:
                    block_mask = create_flex_block_mask(doc_ids.squeeze(0), block_ids.squeeze(0))
                else:
                    raise RuntimeError("FlexAttention not available. Requires PyTorch 2.5+")
                
                # 2. Apply block-wise masking
                masked_input, mask_pos, t_per_position = apply_block_masking(
                    input_ids, block_ids, t_lower, t_upper, mask_token_id, is_training=True
                )
                mask_positions = mask_pos
                
                # 3. Concatenate [noisy, clean]
                noisy_input_ids = torch.cat([masked_input, input_ids], dim=1)
                input_mask_updated = torch.cat([input_mask, input_mask], dim=1)
                
                # 4. Position IDs for RoPE (repeated)
                position_ids = torch.cat([pos_ids, pos_ids], dim=1)
                
                # 5. Noise weights
                noise_weights = 1.0 / (t_per_position + 1e-3)
                
                # DEBUG: Comprehensive logging for block diffusion (limited to first 10 iterations)
                self._debug_counter += 1
                should_log = torch.distributed.get_rank() == 0 and self._debug_counter <= 10
                
                if should_log:
                    num_masked = mask_positions.sum().item()
                    total_tokens = mask_positions.numel()
                    mask_ratio = num_masked / total_tokens
                    
                    # t statistics
                    t_min = t_per_position.min().item()
                    t_max = t_per_position.max().item()
                    t_mean = t_per_position.mean().item()
                    
                    # noise weight statistics
                    nw_min = noise_weights.min().item()
                    nw_max = noise_weights.max().item()
                    nw_mean = noise_weights.mean().item()
                    
                    # block_ids statistics
                    num_blocks = block_ids.max().item() + 1
                    
                    # Check if mask_token_id appears correctly
                    num_mask_tokens = (masked_input == mask_token_id).sum().item()
                    
                    print(f"\n{'='*60}")
                    print(f"DEBUG DiffusionModule (Block Diffusion) - iter {self._debug_counter}:")
                    print(f"  Config: block_size={block_size}, t_lower={t_lower}, t_upper={t_upper}")
                    print(f"  Input shape: {input_ids.shape}, mask_token_id={mask_token_id}")
                    print(f"  Num blocks: {num_blocks}")
                    print(f"  Masking: {num_masked}/{total_tokens} = {mask_ratio:.1%}")
                    print(f"  Mask tokens in noisy input: {num_mask_tokens}")
                    print(f"  t values: min={t_min:.3f}, max={t_max:.3f}, mean={t_mean:.3f}")
                    print(f"  Noise weights: min={nw_min:.3f}, max={nw_max:.3f}, mean={nw_mean:.3f}")
                    print(f"  Position IDs shape: {position_ids.shape}")
                    print(f"  Noisy input shape: {noisy_input_ids.shape}")
                    
                    # Sample first few tokens for verification
                    print(f"  Sample input_ids[0,:10]: {input_ids[0,:10].tolist()}")
                    print(f"  Sample masked_input[0,:10]: {masked_input[0,:10].tolist()}")
                    print(f"  Sample mask_positions[0,:10]: {mask_positions[0,:10].tolist()}")
                    print(f"  Sample block_ids[0,:10]: {block_ids[0,:10].tolist()}")
                    print(f"  Sample t_per_position[0,:10]: {[f'{x:.2f}' for x in t_per_position[0,:10].tolist()]}")
                    print(f"{'='*60}\n")
                
            elif diff_config.diffusion:
                diffusion_mode = torch.tensor(1, dtype=torch.int, device=device)
                
                mask_token_id = diff_config.mask_token_id
                sampling_eps = diff_config.sampling_eps
                maskable_mask = input_mask.bool()
                
                # Sample noise level until we get at least one masked token
                # This prevents missing gradients for the token embedding layer
                max_attempts = 100
                for attempt in range(max_attempts):
                    t = (1 - sampling_eps) * torch.rand(batch_size, device=device) + sampling_eps
                    noisy_input_ids = transition(input_ids, t[:, None], maskable_mask, mask_token_id)
                    mask_positions = (noisy_input_ids == mask_token_id).long()
                    
                    # Check if we have at least one masked token
                    if mask_positions.sum().item() > 0:
                        break
                    
                    # Safety: if we've tried many times and still no mask, force higher noise
                    if attempt == max_attempts - 1:
                        t = torch.full((batch_size,), 0.5, device=device)
                        noisy_input_ids = transition(input_ids, t[:, None], maskable_mask, mask_token_id)
                        mask_positions = (noisy_input_ids == mask_token_id).long()
                
                noise_weights = torch.reciprocal(t)
                
                # Debug: Check masking
                if torch.distributed.get_rank() == 0:
                     num_masked = mask_positions.sum().item()
                     total_tokens = mask_positions.numel()
                     print(f"DEBUG: Diffusion active. Masked tokens: {num_masked}/{total_tokens} ({num_masked/total_tokens:.2%})")


        return {
            "noisy_input_ids": noisy_input_ids,
            "input_mask_updated": input_mask_updated,
            "position_ids": position_ids,
            "mask_positions": mask_positions,
            "noise_weights": noise_weights,
            "doc_ids": doc_ids,
            "block_ids": block_ids,
            "block_mask": block_mask,
            "diffusion_mode": diffusion_mode
        }


class Embedding(nn.Module, AttachableStore):
    def __init__(self, tp_pg: dist.ProcessGroup, config: LlamaConfig, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.pg = tp_pg

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor):  # [batch_size, seq_length]
        store = self.get_local_store()
        if store is not None:
            if "past_length" in store:
                past_length = store["past_length"]
            else:
                past_length = torch.zeros(1, dtype=torch.long, device=input_ids.device).expand(input_ids.shape[0])

            cumsum_mask = input_mask.cumsum(-1, dtype=torch.long)
            # Store new past_length in store
            store["past_length"] = past_length + cumsum_mask[:, -1]

        # Format input in `[seq_length, batch_size]` to support high TP with low batch_size
        input_ids = input_ids.transpose(0, 1)
        input_embeds = self.token_embedding(input_ids)
        return {"input_embeds": input_embeds}



class LlamaModel(nn.Module):
    """Build pipeline graph"""

    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()

        # Declare all the nodes
        self.p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))
        self.config = config
        self.parallel_config = parallel_config
        self.parallel_context = parallel_context
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        self.diffusion_block = PipelineBlock(
            p2p=self.p2p,
            module_builder=DiffusionModule,
            module_kwargs={"config": config},
            module_input_keys={"input_ids", "input_mask", "position_ids"},
            module_output_keys={
                "noisy_input_ids", 
                "input_mask_updated", 
                "position_ids", 
                "mask_positions", 
                "noise_weights", 
                "block_mask",
                "diffusion_mode"
            },
        )

        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=Embedding,
            module_kwargs={
                "tp_pg": parallel_context.tp_pg,
                "config": config,
                "parallel_config": parallel_config,
            },
            module_input_keys={"input_ids", "input_mask"},
            module_output_keys={"input_embeds"},
        )
        log_rank(f"Initialize RoPE Theta = {config.rope_theta}", logger=logger, level=logging.INFO, rank=0)
        if config.rope_interleaved:
            log_rank(
                "The RoPE interleaved version differs from the Transformers implementation. It's better to set rope_interleaved=False if you need to convert the weights to Transformers",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=LlamaDecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={
                        "hidden_states", 
                        "sequence_mask", 
                        "position_ids", 
                        "mask_positions", 
                        "noise_weights", 
                        "block_mask",
                        "diffusion_mode"
                    },
                    module_output_keys={
                        "hidden_states", 
                        "sequence_mask", 
                        "position_ids", 
                        "mask_positions", 
                        "noise_weights", 
                        "block_mask",
                        "diffusion_mode"
                    },
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=TritonRMSNorm,
            module_kwargs={"hidden_size": config.hidden_size, "eps": config.rms_norm_eps},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )  # TODO

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Understand that this means that we return sharded logits that are going to need to be gathered
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                # TODO @thomasw21: refactor so that we store that default in a single place.
                "mode": self.tp_mode,
                "async_communication": tp_linear_async_communication,
                "tp_recompute_allgather": parallel_config.tp_recompute_allgather,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )

        self.cast_to_fp32 = PipelineBlock(
            p2p=self.p2p,
            module_builder=lambda: lambda x: x.float(),
            module_kwargs={},
            module_input_keys={"x"},
            module_output_keys={"output"},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        position_ids: Optional[Union[torch.Tensor, TensorPointer]] = None,
    ):
        return self.forward_with_hidden_states(input_ids=input_ids, input_mask=input_mask, position_ids=position_ids)[0]

    def forward_with_hidden_states(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        position_ids: Optional[Union[torch.Tensor, TensorPointer]] = None,
    ):
        # all tensors are optional as most ranks don't need anything from the dataloader.

        diff_output = self.diffusion_block(input_ids=input_ids, input_mask=input_mask, position_ids=position_ids)
        
        # Use noisy_input_ids for embedding if available (Rank 0), or pass through whatever came out
        # Actually, PipelineBlock forward returns dict.
        # If rank != 0, diff_output values are TensorPointers.
        # If rank == 0, they are Tensors.
        
        # We need to route noisy_input_ids to Embedding input_ids
        noisy_input_ids = diff_output["noisy_input_ids"]
        
        # Embedding expects "input_ids", "input_mask"
        # We pass noisy_input_ids as "input_ids"
        emb_output = self.token_position_embeddings(
            input_ids=noisy_input_ids, 
            input_mask=input_mask # Embedding uses input_mask for store['past_length'], keep original?
            # Actually, for block diffusion we updated input_mask.
        )

        hidden_encoder_states = {
            "hidden_states": emb_output["input_embeds"],
            "sequence_mask": diff_output["input_mask_updated"],
            "position_ids": diff_output["position_ids"],
            "mask_positions": diff_output["mask_positions"],
            "noise_weights": diff_output["noise_weights"],
            "block_mask": diff_output["block_mask"],
            "diffusion_mode": diff_output["diffusion_mode"],
        }
        for encoder_block in self.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits, hidden_states, hidden_encoder_states["diffusion_mode"], hidden_encoder_states["mask_positions"], hidden_encoder_states["noise_weights"]

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        model_config = self.config
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            LlamaDecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 3 * d_ff * model_config.hidden_size,
            # This is the last lm_head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        world_size = self.parallel_context.world_pg.size()
        try:
            num_key_values_heads = self.config.num_key_value_heads
        except AttributeError:
            num_key_values_heads = self.config.num_attention_heads

        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=num_key_values_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
        )

        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s


@torch.jit.script
def masked_mean(loss, label_mask, dtype):
    # type: (torch.Tensor, torch.Tensor, torch.dtype) -> torch.Tensor
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


class Loss(nn.Module):
    def __init__(self, tp_pg: dist.ProcessGroup):
        super().__init__()
        self.tp_pg = tp_pg
        self._debug_counter = 0  # Counter for debug logging

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
        mask_positions: Optional[torch.Tensor] = None,
        noise_weights: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        diffusion_mode: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if diffusion_mode is not None and diffusion_mode.item() > 0:
            mode = diffusion_mode.item()
            # Handle diffusion loss
            loss = sharded_cross_entropy(
                sharded_logits,
                label_ids.transpose(0, 1).contiguous(),
                group=self.tp_pg,
                dtype=torch.float,
            ).transpose(0, 1) # [B, L]

            if mode == 2: # Block Diffusion
                # For block diffusion, input is concatenated [noisy, clean]. logits are [2L, B, V].
                # We need to split logits
                seq_len = label_ids.shape[1] # This is original seq_len (L)
                batch_size = label_ids.shape[0]
                # logits are [2*seq_len, batch, vocab]
                
                noisy_logits = sharded_logits[:seq_len, :, :] # [L, B, V]
                # clean_logits = sharded_logits[seq_len:, :, :] # [L, B, V] - not used for loss
                
                # CRITICAL: Autoregressive alignment
                # logits[i] predicts token at position i+1
                # We need to align: logits[:-1] with labels[1:]
                noisy_logits_aligned = noisy_logits[:-1, :, :]  # [L-1, B, V] - drop last logit
                labels_aligned = label_ids[:, 1:].transpose(0, 1).contiguous()  # [L-1, B] - drop first label
                mask_positions_aligned = mask_positions[:, 1:]  # [B, L-1] - mask status for predicted positions
                noise_weights_aligned = noise_weights[:, 1:]  # [B, L-1] - weights for predicted positions
                
                # 1. Diffusion Loss on noisy part (aligned)
                diff_loss = sharded_cross_entropy(
                    noisy_logits_aligned, # [L-1, B, V]
                    labels_aligned,  # [L-1, B]
                    group=self.tp_pg,
                    dtype=torch.float
                ).transpose(0, 1) # [B, L-1]
                
                # Reference implementation divides by total sequence length, not mask count:
                # loss = (losses * weights * mask).sum() / input_seq.size(0)
                # For batched input, we divide by total tokens (batch_size * seq_len)
                # Note: keep using original seq_len for normalization to match reference
                total_tokens = batch_size * seq_len
                
                # Convert mask_positions to float for multiplication (using aligned mask)
                mask_float = mask_positions_aligned.float()
                
                # Weight by noise level and apply mask (using aligned weights)
                weighted_diff_loss = (diff_loss * mask_float * noise_weights_aligned).sum() / total_tokens
                
                # Debug logging (limited to first 10 iterations)
                self._debug_counter += 1
                should_log = torch.distributed.get_rank() == 0 and self._debug_counter <= 10
                
                if should_log:
                    num_masked = mask_float.sum().item()
                    mask_ratio = num_masked / total_tokens
                    avg_weight = (noise_weights_aligned * mask_float).sum().item() / (num_masked + 1e-8)
                    avg_loss_per_masked = (diff_loss * mask_float).sum().item() / (num_masked + 1e-8)
                    
                    # Additional debug: unweighted loss and AR-like loss for comparison
                    unweighted_loss = (diff_loss * mask_float).sum().item() / total_tokens
                    
                    # Loss on ALL tokens (like AR) for reference
                    all_token_loss = diff_loss.mean().item()
                    
                    # Loss on non-masked tokens (should be low if model predicts well)
                    non_mask_float = 1 - mask_float
                    num_non_masked = non_mask_float.sum().item()
                    if num_non_masked > 0:
                        avg_loss_non_masked = (diff_loss * non_mask_float).sum().item() / num_non_masked
                    else:
                        avg_loss_non_masked = 0.0
                    
                    print(f"\n{'='*60}")
                    print(f"DEBUG Block Diffusion Loss (Llama, iter {self._debug_counter}):")
                    print(f"  Shapes (aligned): logits={noisy_logits_aligned.shape}, labels={labels_aligned.shape}, mask={mask_positions_aligned.shape}")
                    print(f"  Original shapes: labels={label_ids.shape}, mask={mask_positions.shape}")
                    print(f"  mask_ratio={mask_ratio:.3f} ({int(num_masked)}/{total_tokens} tokens)")
                    print(f"  noise_weights_aligned: min={noise_weights_aligned.min().item():.3f}, max={noise_weights_aligned.max().item():.3f}, avg_on_masked={avg_weight:.3f}")
                    print(f"  Per-token loss (masked only): {avg_loss_per_masked:.4f}")
                    print(f"  Per-token loss (non-masked): {avg_loss_non_masked:.4f}")
                    print(f"  Per-token loss (ALL tokens): {all_token_loss:.4f}")
                    print(f"  Unweighted loss (ref formula w/o weights): {unweighted_loss:.4f}")
                    print(f"  Final weighted loss: {weighted_diff_loss.item():.4f}")
                    print(f"  ")
                    print(f"  Expected: final_loss ≈ mask_ratio × avg_loss_per_masked × avg_weight")
                    print(f"            = {mask_ratio:.3f} × {avg_loss_per_masked:.3f} × {avg_weight:.3f} = {mask_ratio * avg_loss_per_masked * avg_weight:.4f}")
                    print(f"{'='*60}\n")
                
                loss = weighted_diff_loss
                return {"loss": loss}
                
            elif mode == 1: # Vanilla Diffusion
                 # Slice to align targets: we need to check if the TARGET (next token) was masked.
                 # Input: x_0, ..., x_{S-1}. Labels: x_1, ..., x_S.
                 # Mask positions corresponds to Input (x_0...x_{S-1}).
                 # We want to know if x_1...x_S were masked.
                 # We have mask status for x_1...x_{S-1} (from mask_positions[1:]).
                 # We DO NOT have mask status for x_S (it wasn't in input).
                 # So we must drop the last prediction step.
                 
                 # sharded_logits is [L, B, V], slice to [L-1, B, V] for predicting x_1...x_{S-1}
                 logits_sliced = sharded_logits[:-1, :, :]
                 labels_sliced = label_ids[:, :-1]         # x_1...x_{S-1}
                 mask_sliced = mask_positions[:, 1:]       # Mask status of x_1...x_{S-1}
                 
                 loss = sharded_cross_entropy(
                    logits_sliced,
                    labels_sliced.transpose(0, 1).contiguous(),
                    group=self.tp_pg,
                    dtype=torch.float
                 ).transpose(0, 1) # [B, S-1]

                 mask = mask_sliced.float().type_as(loss)
                 dsigma = noise_weights # [B]
                 
                 loss_per_token = loss * mask
                 if dsigma.dim() == 1:
                     dsigma = dsigma.unsqueeze(1)
                 
                 # Formula: (dsigma[:, None] * loss).sum() / loss_mask.sum() - matches DiffuLLaMA
                 loss_weighted = (loss_per_token.sum(dim=1) * dsigma.squeeze()).sum()
                 
                 total_masked = mask.sum()
                 if total_masked > 0:
                     loss_val = loss_weighted / total_masked
                 else:
                     loss_val = torch.tensor(0.0, device=loss.device, requires_grad=True)
                 return {"loss": loss_val}


        loss = sharded_cross_entropy(
            sharded_logits,
            label_ids.transpose(0, 1).contiguous(),
            group=self.tp_pg,
            dtype=torch.float,
        ).transpose(0, 1)
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        return {"loss": loss}


class LossWithZLoss(Loss):
    def __init__(self, tp_pg: dist.ProcessGroup, z_loss_coefficient: float):
        super().__init__(tp_pg)
        self.z_loss_coef = z_loss_coefficient

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        loss, z_loss = sharded_cross_entropy(
            sharded_logits,
            label_ids.transpose(0, 1).contiguous(),
            group=self.tp_pg,
            dtype=torch.float,
            z_loss_coef=self.z_loss_coef,
        )
        loss = masked_mean(loss.transpose(0, 1), label_mask, dtype=torch.float)
        z_loss = masked_mean(z_loss.detach().transpose(0, 1), label_mask, dtype=torch.float)
        return {"loss": loss, "z_loss": z_loss}


class LlamaForTraining(NanotronModel):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)

        # Choose the appropriate loss class based on config
        loss_kwargs = {
            "tp_pg": parallel_context.tp_pg,
        }
        if config.z_loss_enabled:
            loss_kwargs["z_loss_coefficient"] = config.z_loss_coefficient

        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=LossWithZLoss if config.z_loss_enabled else Loss,
            module_kwargs=loss_kwargs,
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
                "mask_positions", 
                "noise_weights", 
                "doc_ids", 
                "diffusion_mode"
            },
            module_output_keys={"loss", "z_loss"} if config.z_loss_enabled else {"loss"},
        )

        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
        position_ids: Optional[Union[torch.Tensor, TensorPointer]] = None,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        sharded_logits, _, diffusion_mode, mask_positions, noise_weights, doc_ids = self.model.forward_with_hidden_states(
            input_ids=input_ids,
            input_mask=input_mask,
            position_ids=position_ids,
        )
        loss = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
            mask_positions=mask_positions,
            noise_weights=noise_weights,
            doc_ids=doc_ids,
            diffusion_mode=diffusion_mode,
        )
        if self.config.z_loss_enabled:
            return {"loss": loss["loss"], "z_loss": loss["z_loss"]}
        else:
            return {"loss": loss["loss"]}

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """Initialize model parameters randomly.
        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
        init_method = config.model.init_method
        if isinstance(init_method, RandomInit):
            parametrizator_cls = StandardParametrizator
        elif isinstance(init_method, SpectralMupInit):
            parametrizator_cls = SpectralMupParametrizator
        else:
            raise ValueError(f"Unknown init method {init_method}")

        parametrizator = parametrizator_cls(config=config)

        log_rank(
            f"Parametrizing model parameters using {parametrizator.__class__.__name__}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)

            module_name, param_name = param_name.rsplit(".", 1)

            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"

            if full_param_name in initialized_parameters:
                # Already initialized
                continue

            module = model.get_submodule(module_name)
            parametrizator.parametrize(param_name, module)

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_embeddings_lm_head_tied_names(self):
        """Get the names of the tied embeddings and lm_head weights"""
        if self.config.tie_word_embeddings is True:
            return ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
        else:
            return []

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)


def get_flops(
    num_layers,
    hidden_size,
    num_heads,
    num_key_value_heads,
    vocab_size,
    seq_len,
    ffn_hidden_size,
    batch_size=1,
):
    """Counts flops in an decoder-only model
    Args:
        num_layers: number of decoder layers
        hidden_size: hidden size of the model
        num_heads: number of heads in the model
        num_key_value_heads: number of key/value heads in the model
        ffn_hidden_size: hidden size of the FFN
        vocab_size: size of the vocabulary
        seq_len: sequence length of the decoder
        batch_size: batch size
    Returns:
        model_flops: flops in the model (should be independent of the hardware and model implementation)
        hardware_flops: flops in the hardware (actual flops performed on the hardware). Check 6.3 in https://arxiv.org/pdf/2205.05198.pdf
    """
    if num_key_value_heads is None:
        num_key_value_heads = num_heads
    hidden_size_per_head = hidden_size // num_heads
    # In the following we mark the reduced dimension with parentheses
    # decoder
    # self attention
    ## qkv projection
    decoder_qkv_proj_flops_fwd = (
        2 * num_layers * batch_size * seq_len * (hidden_size) * num_heads * hidden_size_per_head
        + 2 * num_layers * batch_size * seq_len * (hidden_size) * 2 * num_key_value_heads * hidden_size_per_head
    )
    ## qk logits
    decoder_qk_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * seq_len
    ## v logits
    decoder_v_logits_flops_fwd = 2 * num_layers * batch_size * num_heads * seq_len * (seq_len) * hidden_size_per_head
    ## attn out
    decoder_attn_out_flops_fwd = (
        2 * num_layers * batch_size * num_heads * seq_len * (hidden_size_per_head) * hidden_size
    )
    # FF
    ## 1st layer
    decoder_ffn_1_flops_fwd = 4 * num_layers * batch_size * seq_len * (hidden_size) * ffn_hidden_size
    ## 2nd layer
    decoder_ffn_2_flops_fwd = 2 * num_layers * batch_size * seq_len * (ffn_hidden_size) * hidden_size
    
    decoder_flops_fwd = (
        decoder_qkv_proj_flops_fwd
        + decoder_qk_logits_flops_fwd
        + decoder_v_logits_flops_fwd
        + decoder_attn_out_flops_fwd
        + decoder_ffn_1_flops_fwd
        + decoder_ffn_2_flops_fwd
    )

    # lm head
    lm_head_flops_fwd = 2 * batch_size * seq_len * (hidden_size) * vocab_size

    # the bwd pass requires double the flops in case of matmuls to calculate the gradients with respect to
    # both input and weight tensors
    model_flops = 3 * (decoder_flops_fwd + lm_head_flops_fwd)  # 1 for fwd + 2 for bwd

    hardware_flops = model_flops  # TODO: This is a placeholder for now

    return model_flops, hardware_flops
