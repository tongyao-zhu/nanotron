from typing import Dict, List, Optional, Tuple, Union
import os
import torch
import torch.nn.functional as F
from flash_attn.modules.mha import flash_attn_varlen_kvpacked_func
from torch import nn
from torch.utils.checkpoint import CheckpointFunction

# Import FlexAttention for block diffusion
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, ParallelismArgs
from nanotron.config.models_config import Qwen2Config, RandomInit, SpectralMupInit, DiffusionArgs
from nanotron.block_diffusion import (
    transition,
    create_flex_block_mask,
    FLEX_ATTENTION_AVAILABLE
)
from nanotron.logging import log_rank
from nanotron.models import NanotronModel
from nanotron.nn.activations import ACT2FN
from nanotron.nn.attention import ALL_ATTENTION_FUNCTIONS, get_attention_mask
from nanotron.nn.layer_norm import LlamaRMSNorm as RMSNorm
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.nn.rotary import RotaryEmbedding
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
from nanotron.logging import LogMixin
from nanotron.nn.llama3_ring_attention import llama3_flash_attn_varlen_kvpacked_func, llama3_flash_attn_prepare_cu_seqlens
logger = logging.get_logger(__name__)

DEBUG = False
if os.environ.get("DEBUG", "false") == "true":
    DEBUG = True
    print("DEBUG mode enabled")


class CoreAttention(nn.Module):
    """Core attention module that can use different attention implementations"""

    def __init__(
        self,
        config: Qwen2Config,
        tp_pg: dist.ProcessGroup,
        cp_pg: dist.ProcessGroup,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.local_num_heads = self.num_heads // tp_pg.size()
        self.local_num_kv_heads = self.num_kv_heads // tp_pg.size()
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self._attn_implementation = config._attn_implementation
        self.cp_pg = cp_pg
        self.sliding_window_size = config.sliding_window_size
        self.simple_causal_mask = True
        self.flex_attention_mask = config.flex_attention_mask if hasattr(config, "flex_attention_mask") else None

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_length: Optional[int],
        attention_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        attention_func = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]
        cu_seqlens = kwargs.get("cu_seqlens", None)

        if self._attn_implementation == "ring_flash_triton":
            query_states = query_states.view(-1, seq_length, self.local_num_heads, self.head_dim)
            key_states = key_states.view(-1, seq_length, self.local_num_kv_heads, self.head_dim)
            value_states = value_states.view(-1, seq_length, self.local_num_kv_heads, self.head_dim)
        elif self._attn_implementation == "ring":
            query_states = query_states.view(-1, self.local_num_heads, self.head_dim)
            key_states = key_states.view(-1, self.local_num_kv_heads, self.head_dim)
            value_states = value_states.view(-1, self.local_num_kv_heads, self.head_dim)
        else:
            raise NotImplementedError(f"Attention implementation {self._attn_implementation} not implemented")

        attn_output = attention_func(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            max_seqlen=seq_length,
            dropout=dropout,
            scaling=None,
            sliding_window=self.sliding_window_size,
            ring_pg=self.cp_pg,
            document_ids=kwargs.get("document_ids", None) if self._attn_implementation == "flex_attention" else None,
            flex_attention_mask=self.flex_attention_mask if self._attn_implementation == "flex_attention" else None,
            **kwargs,
        )[0]

        return attn_output.view(-1, self.local_num_heads * self.head_dim)


class Qwen2Attention(LogMixin, nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        cp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.tp_pg_size = tp_pg.size()
        self.cp_pg_size = cp_pg.size()
        self.cp_pg = cp_pg

        # Head configuration
        self.num_heads = config.num_attention_heads
        self.local_num_heads = self.num_heads // self.tp_pg_size

        # KV head configuration
        self.num_kv_heads = config.num_key_value_heads
        self.local_num_kv_heads = self.num_kv_heads // self.tp_pg_size

        # Dimensions
        self.head_dim = config.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.local_q_size = self.local_num_heads * self.head_dim
        self.local_kv_size = self.local_num_kv_heads * self.head_dim

        # TP mode configuration
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        qkv_contiguous_chunks = (
            self.q_size,  # Q chunk size
            self.kv_size,  # K chunk size
            self.kv_size,  # V chunk size
        )
        self.qkv_proj = TensorParallelColumnLinear(
            self.hidden_size,
            self.q_size + 2 * self.kv_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=config.attention_bias,  # Qwen2 uses bias for QKV, Llama doesn't
            async_communication=tp_linear_async_communication,
            contiguous_chunks=qkv_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.o_proj = TensorParallelRowLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )
        if config._use_qkv_packed:
            from nanotron.nn.rotary import FlashRotaryEmbedding
            self.rotary_emb = FlashRotaryEmbedding(
                dim=self.head_dim,
                base=config.rope_theta,
                interleaved=config.rope_interleaved,
                seq_len_interpolation_factor=config.rope_seq_len_interpolation_factor,
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.max_position_embeddings,
                base=config.rope_theta,
                interleaved=config.rope_interleaved,
                seq_len_scaling_factor=1,
                fused=config._fused_rotary_emb,
            )
        self.attention = CoreAttention(config, tp_pg, cp_pg, layer_idx)
        self.simple_causal_mask = True
        self._use_qkv_packed = config._use_qkv_packed
        self.sliding_window_size = config.sliding_window_size
        self.log_attn_probs = config.log_attn_probs
        self.heads_k_stride = config.ring_attn_heads_k_stride
        # TODO: support SFT

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        diffusion_mode: Optional[torch.Tensor] = None,
        block_mask=None,
    ):
        # Compute seq_length from cu_seqlens
        if cu_seqlens is not None:
            if isinstance(cu_seqlens, dict):
                # llama3_ring_attention case
                seq_length = cu_seqlens["max_seqlen_q"] // self.cp_pg_size
            else:
                # Regular flash attention case - compute from cu_seqlens differences
                if cu_seqlens.numel() > 1:
                    seq_length = int((cu_seqlens[1] - cu_seqlens[0]).item())
                else:
                    # Fallback: compute from hidden_states shape
                    seq_length = hidden_states.shape[0] // self.cp_pg_size
        else:
            # Fallback: compute from hidden_states shape (assumes batch_size=1)
            seq_length = hidden_states.shape[0] // self.cp_pg_size

        qkv = self.qkv_proj(hidden_states)
        
        if diffusion_mode is not None and diffusion_mode.item() == 2:
            attn_output = self._forward_block_diffusion(qkv, seq_length, block_mask)
        else:
            causal = True
            if diffusion_mode is not None and diffusion_mode.item() == 1:
                causal = False

            if self._use_qkv_packed:
                attn_output = self._forward_packed(qkv, seq_length, cu_seqlens, causal=causal)
            else:
                raise NotImplementedError("Non-packed QKV not implemented for Qwen2Attention")
                
        output = self.o_proj(attn_output)
        return {"hidden_states": output}

    def _forward_packed(self, qkv, seq_length, cu_seqlens, causal=True):
        assert cu_seqlens is not None, "cu_seqlens must be provided for packed attention"
        q = qkv[..., : self.local_num_heads * self.head_dim]  # Not contiguous, similar to flash_attn
        kv = qkv[..., self.local_num_heads * self.head_dim :]  # Not contiguous, similar to flash_attn
        q = q.view(-1, seq_length, self.local_num_heads, self.head_dim)
        kv = kv.view(-1, seq_length, 2, self.local_num_kv_heads, self.head_dim)
        if self.config.no_rope_layer is None or (self.layer_idx + 1) % self.config.no_rope_layer != 0:
            seqlen_offset = dist.get_rank(self.cp_pg) * seq_length
            q, kv = self.rotary_emb(
                q, kv, seqlen_offset=seqlen_offset, max_seqlen=seq_length*self.cp_pg_size
            )
        else:
            log_rank(f"skipping rotary for layer {self.layer_idx + 1}", logger=logger, level=logging.DEBUG, rank=0)
            self.sliding_window_size = None # WARNING: we skip sliding window for no-rope

        q = q.view(-1, self.local_num_heads, self.head_dim)
        kv = kv.view(-1, 2, self.local_num_kv_heads, self.head_dim)
        max_seqlen = seq_length


        if self.config._attn_implementation == "llama3_ring_attention":
            attn_output = llama3_flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_seqlens_q=cu_seqlens["cu_seqlens_q"],
                cu_seqlens_k=cu_seqlens["cu_seqlens_k"],
                max_seqlen_q=cu_seqlens["max_seqlen_q"],
                max_seqlen_k=cu_seqlens["max_seqlen_k"],
                heads_k_stride=self.heads_k_stride,
                local_k_slice=cu_seqlens["local_k_slice"],
                dropout_p=0.0,
                softmax_scale=None,
                causal=causal,
                alibi_slopes=None,
                window_size=(self.sliding_window_size - 1, 0) if self.sliding_window_size is not None else (-1, -1),
                deterministic=False,
                return_attn_probs=self.log_attn_probs,
                group=self.cp_pg,
            )  # Not contiguous, similar to flash_attn
        else:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            attn_output = flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                0.0,
                softmax_scale=None,
                causal=causal,
                alibi_slopes=None,
                window_size=(self.sliding_window_size - 1, 0) if self.sliding_window_size is not None else (-1, -1),
                deterministic=False,
                return_attn_probs=self.log_attn_probs,
            )  # Not contiguous, similar to flash_attn

        if self.log_attn_probs:
            attn_output, attn_probs, _ = attn_output
            self.tbi_logger({"attn_probs": attn_probs})
        return attn_output.reshape(-1, self.local_num_heads * self.head_dim)

    def _forward_block_diffusion(
        self, 
        qkv: torch.Tensor,
        seq_length: int, 
        block_mask,
    ) -> torch.Tensor:
        if not FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError("FlexAttention not available. Requires PyTorch 2.5+")
        
        if block_mask is None:
            raise RuntimeError("block_mask is None but diffusion_mode=2")
        
        q = qkv[..., : self.local_num_heads * self.head_dim]
        kv = qkv[..., self.local_num_heads * self.head_dim :]
        
        total_tokens = qkv.shape[0]
        batch_size = total_tokens // seq_length
        
        q = q.view(batch_size, seq_length, self.local_num_heads, self.head_dim)
        kv = kv.view(batch_size, seq_length, 2, self.local_num_kv_heads, self.head_dim)
        
        if self.config.no_rope_layer is None or (self.layer_idx + 1) % self.config.no_rope_layer != 0:
            # For block diffusion, split into noisy and clean halves, apply RoPE separately to each
            half_seq_length = seq_length // 2
            
            # Split q and kv into noisy and clean halves
            q_noisy = q[:, :half_seq_length, :, :]  # [batch_size, half_seq_length, num_heads, head_dim]
            q_clean = q[:, half_seq_length:, :, :]  # [batch_size, half_seq_length, num_heads, head_dim]
            kv_noisy = kv[:, :half_seq_length, :, :, :]  # [batch_size, half_seq_length, 2, num_kv_heads, head_dim]
            kv_clean = kv[:, half_seq_length:, :, :, :]  # [batch_size, half_seq_length, 2, num_kv_heads, head_dim]
            
            # Both halves start from position 0, so use seqlen_offset=0 for both
            seqlen_offset = 0
            max_seqlen = half_seq_length
            
            # Apply RoPE to noisy half (positions 0 to half_seq_length-1)
            q_noisy, kv_noisy = self.rotary_emb(
                q_noisy, kv_noisy, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen
            )
            
            # Apply RoPE to clean half (also positions 0 to half_seq_length-1)
            q_clean, kv_clean = self.rotary_emb(
                q_clean, kv_clean, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen
            )
            
            # Concatenate back
            q = torch.cat([q_noisy, q_clean], dim=1)  # [batch_size, seq_length, num_heads, head_dim]
            kv = torch.cat([kv_noisy, kv_clean], dim=1)  # [batch_size, seq_length, 2, num_kv_heads, head_dim]
        
        k = kv[:, :, 0, :, :]
        v = kv[:, :, 1, :, :]
        
        if self.local_num_kv_heads != self.local_num_heads:
            n_rep = self.local_num_heads // self.local_num_kv_heads
            k = k.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(batch_size, seq_length, self.local_num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(batch_size, seq_length, self.local_num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_output = flex_attention(q, k, v, block_mask=block_mask)
        
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(-1, self.local_num_heads * self.head_dim)
        
        return attn_output


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        intermediate_size: int,
    ) -> None:
        super().__init__()

        # Get TP mode and communication settings
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        gate_up_contiguous_chunks = (
            intermediate_size,  # shape of gate_linear
            intermediate_size,  # shape of up_linear
        )

        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,  # Qwen2 doesn't use bias for gate_up_proj
            async_communication=tp_linear_async_communication,
            contiguous_chunks=gate_up_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )

        # Define down projection
        self.down_proj = TensorParallelRowLinear(
            intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,  # Qwen2 doesn't use bias for down_proj
            async_communication=tp_linear_async_communication,
        )

        # Define activation function (silu followed by multiplication)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        # Apply gate_up_proj to get gate and up projections
        merged_states = self.gate_up_proj(hidden_states)

        # Apply activation function (SiLU and Mul)
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        hidden_states = self.act(gate_states) * up_states

        # Apply down projection
        hidden_states = self.down_proj(hidden_states)

        return {"hidden_states": hidden_states}


class Qwen2MoELayer(nn.Module):
    """Mixture of experts Layer for Qwen2 models."""

    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # MoE specific configurations
        self.num_experts = config.moe_config.num_experts  # Total number of experts
        self.num_experts_per_token = config.moe_config.top_k  # Number of experts used per token (top-k)
        self.expert_parallel_size = getattr(parallel_config, "expert_parallel_size", 1)
        self.num_local_experts = self.num_experts // self.expert_parallel_size  # Experts per device

        # Get TP mode configuration
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        # Router for selecting experts
        self.router = TensorParallelColumnLinear(
            self.hidden_size,
            self.num_experts,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
        )

        # Enable shared experts if configured
        self.enable_shared_expert = getattr(config.moe_config, "enable_shared_expert", False)
        if self.enable_shared_expert:
            self.shared_expert = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
            )
            self.shared_expert_gate = TensorParallelColumnLinear(
                self.hidden_size,
                1,
                pg=tp_pg,
                mode=tp_mode,
                bias=False,
                async_communication=tp_linear_async_communication,
            )

        # Create the expert MLPs
        self.experts = nn.ModuleList(
            [
                Qwen2MLP(
                    config=config,
                    parallel_config=parallel_config,
                    tp_pg=tp_pg,
                )
                for _ in range(self.num_local_experts)
            ]
        )

        # Whether to recompute MoE layer during backward pass for memory efficiency
        self.recompute_layer = parallel_config.recompute_layer

        # Token dispatcher type - determines communication pattern
        self.token_dispatcher_type = getattr(config.moe_config, "token_dispatcher_type", "alltoall")
        # For more sophisticated implementations, we would add token dispatcher logic here

    def _compute_router_probabilities(self, hidden_states):
        """Compute routing probabilities for each token to each expert."""
        router_logits = self.router(hidden_states)  # [batch_size*seq_length, num_experts]

        # Get the top-k experts per token
        routing_weights, routing_indices = torch.topk(router_logits, k=self.num_experts_per_token, dim=-1)

        # Apply softmax on the top-k values
        routing_weights = F.softmax(routing_weights, dim=-1)

        return routing_weights, routing_indices

    def _dispatch_tokens(self, hidden_states, routing_weights, routing_indices):
        """
        Dispatches tokens to their selected experts.
        In a full implementation, this would handle the actual token routing logic
        including communication between devices.
        """
        # Simplified implementation - in a complete version this would handle
        # all-to-all or all-gather communications for distributed experts

        hidden_states.shape[0]
        dispatched_inputs = []
        expert_counts = []

        # For each expert, gather the tokens assigned to it
        for expert_idx in range(self.num_local_experts):
            # Find tokens that have this expert in their top-k
            expert_mask = (routing_indices == expert_idx).any(dim=-1)
            tokens_for_expert = hidden_states[expert_mask]

            # Get the routing weights for this expert
            expert_positions = (routing_indices == expert_idx).nonzero(as_tuple=True)
            token_positions, k_positions = expert_positions
            expert_weights = routing_weights[token_positions, k_positions].unsqueeze(-1)

            # Scale inputs by routing weights
            scaled_inputs = tokens_for_expert * expert_weights

            dispatched_inputs.append(scaled_inputs)
            expert_counts.append(len(tokens_for_expert))

        return dispatched_inputs, expert_counts

    def _combine_expert_outputs(self, expert_outputs, routing_indices, original_shape):
        """
        Combines outputs from different experts back to the original tensor layout.
        """
        # Initialize output tensor with zeros
        combined_output = torch.zeros(original_shape, device=expert_outputs[0].device)

        for expert_idx, expert_output in enumerate(expert_outputs):
            if expert_output.shape[0] == 0:  # Skip if no tokens were routed to this expert
                continue

            # Find positions where this expert was in the top-k
            expert_mask = (routing_indices == expert_idx).any(dim=-1)
            combined_output[expert_mask] += expert_output

        return combined_output

    def _core_forward(self, hidden_states):
        """Core forward logic for MoE layer."""
        # Get router probabilities
        routing_weights, routing_indices = self._compute_router_probabilities(hidden_states)

        # Dispatch tokens to experts
        dispatched_inputs, expert_counts = self._dispatch_tokens(hidden_states, routing_weights, routing_indices)

        # Process tokens with their assigned experts
        expert_outputs = []
        for expert_idx, (inputs, count) in enumerate(zip(dispatched_inputs, expert_counts)):
            if count == 0:  # Skip computation if no tokens assigned
                expert_outputs.append(torch.tensor([], device=hidden_states.device))
                continue

            # Forward through the expert
            output = self.experts[expert_idx](hidden_states=inputs)["hidden_states"]
            expert_outputs.append(output)

        # Combine expert outputs
        output = self._combine_expert_outputs(expert_outputs, routing_indices, hidden_states.shape)

        # Add shared expert contribution if enabled
        if self.enable_shared_expert:
            shared_expert_output = self.shared_expert(hidden_states=hidden_states)["hidden_states"]
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = output + shared_gate * shared_expert_output

        return output

    def _checkpointed_forward(self, hidden_states):
        """Apply gradient checkpointing to save memory during training."""
        return CheckpointFunction.apply(self._core_forward, True, hidden_states)

    def forward(self, hidden_states):
        """Forward pass for the MoE layer."""
        if self.recompute_layer and self.training:
            hidden_states = self._checkpointed_forward(hidden_states)
        else:
            hidden_states = self._core_forward(hidden_states)

        return {"hidden_states": hidden_states}


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        cp_pg: dist.ProcessGroup,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Use fused RMSNorm if configured
        norm_class = TritonRMSNorm if config._fused_rms_norm else RMSNorm
        self.input_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Qwen2Attention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            cp_pg=cp_pg,
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)

        # Use MoE layer if this layer is in the MoE layers list
        if config.moe_config and layer_idx in config.moe_config.layers:
            from nanotron.nn.moe import Qwen2MoELayer

            self.mlp = Qwen2MoELayer(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
                layer_idx=layer_idx,
            )
        else:
            self.mlp = Qwen2MLP(
                config=config,
                parallel_config=parallel_config,
                tp_pg=tp_pg,
                intermediate_size=config.intermediate_size,
            )

        self.recompute_layer = parallel_config.recompute_layer

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        cu_seqlens: Union[torch.Tensor, TensorPointer],
        diffusion_mode: Optional[Union[torch.Tensor, TensorPointer]] = None,
        block_mask=None,
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(
            hidden_states=hidden_states, 
            cu_seqlens=cu_seqlens, 
            diffusion_mode=diffusion_mode,
            block_mask=block_mask,
        )
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
        hidden_states = hidden_states + residual

        return hidden_states, cu_seqlens, diffusion_mode, block_mask

    def _checkpointed_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        diffusion_mode: Optional[torch.Tensor] = None,
        block_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return CheckpointFunction.apply(self._core_forward, True, hidden_states, cu_seqlens, diffusion_mode, block_mask)

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        cu_seqlens: Union[torch.Tensor, TensorPointer],
        diffusion_mode: Optional[Union[torch.Tensor, TensorPointer]] = None,
        block_mask=None,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        if self.recompute_layer and not isinstance(hidden_states, TensorPointer):
            hidden_states, cu_seqlens, diffusion_mode, block_mask = self._checkpointed_forward(
                hidden_states, cu_seqlens, diffusion_mode, block_mask
            )
        else:
            hidden_states, cu_seqlens, diffusion_mode, block_mask = self._core_forward(
                hidden_states, cu_seqlens, diffusion_mode, block_mask
            )

        return {
            "hidden_states": hidden_states,
            "cu_seqlens": cu_seqlens,
            "diffusion_mode": diffusion_mode,
            "block_mask": block_mask,
        }


class Embedding(nn.Module):
    def __init__(self, tp_pg: dist.ProcessGroup, config: Qwen2Config, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.pg = tp_pg

    def forward(self, input_ids: torch.Tensor):  # [batch_size, seq_length]
        input_ids = input_ids.view(-1)  # [batch_size*seq_length]
        input_embeds = self.token_embedding(input_ids)  # [batch_size*seq_length, hidden_size]
        return {"input_embeds": input_embeds}


class DiffusionModule(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config

    def calculate_block_id_and_doc_id(self, input_seq: torch.Tensor, eos_token_id: int, block_size: int):
        if input_seq.ndim == 2:
            input_seq = input_seq.squeeze(0)
        assert input_seq.ndim == 1, f"input_seq.ndim is not 1: {input_seq.ndim} ,shape is {input_seq.shape}"

        # construct attention rules & block mask
        # CHANGE 1: Create a mask where True marks the START of a document
        # We shift EOS right by 1, so the token AFTER an EOS becomes a "start".
        is_doc_start = (input_seq == eos_token_id).roll(1, 0)
        is_doc_start[0] = True  # The first token is always the start of a document

        doc_id = is_doc_start.cumsum(0)
        p = torch.arange(input_seq.size(0), device=input_seq.device)
        pos_id = p - torch.where(is_doc_start, p, -1).cummax(0).values
        block_id = pos_id // block_size + doc_id * len(input_seq)
        block_id = torch.cumsum(block_id != block_id.roll(1, 0), 0) - 1

        return pos_id, block_id, doc_id


    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor):
        device = input_ids.device
        
        # Default outputs (dummies for P2P)
        noisy_input_ids = input_ids
        input_mask_updated = input_mask
        mask_positions = torch.empty(0, device=device)
        noise_weights = torch.empty(0, device=device)
        doc_ids = torch.empty(0, device=device)
        block_ids = torch.empty(0, device=device)
        block_mask = None
        t_per_position = None
        diffusion_mode = torch.tensor(0, dtype=torch.int, device=device)

        if self.config.diffusion_config is not None:
            diff_config = self.config.diffusion_config
            batch_size, seq_len = input_ids.shape

            if diff_config.block_diffusion:
                diffusion_mode = torch.tensor(2, dtype=torch.int, device=device)
                
                block_size = diff_config.block_size
                mask_token_id = diff_config.mask_token_id
                t_lower = diff_config.t_lower
                t_upper = diff_config.t_upper
                bos_token_id = self.config.bos_token_id
                eos_token_id = self.config.eos_token_id
                
                # 1. Calculate block IDs
                pos_id, block_id, doc_id = self.calculate_block_id_and_doc_id(input_seq=input_ids, eos_token_id=eos_token_id, block_size=block_size)
                
                # 1.5. Create FlexAttention block mask (requires batch_size=1)
                if batch_size != 1:
                    raise RuntimeError(f"FlexAttention requires batch_size=1, got {batch_size}")
                block_mask = create_flex_block_mask(doc_id.squeeze(0), block_id.squeeze(0))
                # 2. Apply block-wise masking
                noise_range = (t_lower, t_upper) if self.training else (0.0, 1.0)
                rand = torch.rand_like(input_ids, dtype=torch.float32)
                t = torch.empty_like(rand.squeeze(0)).uniform_(*noise_range)[block_id]
                # ensure taht the t is a tensor with the same shape as block_id
                t = t.unsqueeze(0)
                # print(f"shape of t: {t.shape}")
                # print(f"shape of block_id: {block_id.shape}")
                # print(f"shape of input_ids: {input_ids.shape}")
                # print(f"shape of rand: {rand.shape}")
                # print(f"shape of noisy_input_ids: {noisy_input_ids.shape}")
                noisy_seq = input_ids.masked_fill(rand >= (1 - t),mask_token_id)
                # print(f"shape of noisy_seq: {noisy_seq.shape}")
                noisy_input_ids = torch.cat([noisy_seq, input_ids], dim=1)
                # print(f"shape of noisy_input_ids: {noisy_input_ids.shape}")
                t_per_position = t

                # # 3. Concatenate [noisy, clean]
                # noisy_input_ids = torch.cat([masked_input, input_ids], dim=1)
                # input_mask_updated = torch.cat([input_mask, input_mask], dim=1)
                
                
                # # 5. Noise weights
                # noise_weights = 1.0 / (t_per_position + 1e-3)
                
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

        return {
            "noisy_input_ids": noisy_input_ids,
            "input_mask_updated": input_mask_updated,
            "mask_positions": mask_positions,
            "noise_weights": noise_weights,
            # "doc_ids": doc_ids,
            # "block_ids": block_ids,
            "block_mask": block_mask,
            "diffusion_mode": diffusion_mode,
            "t_per_position": t_per_position
        }


class Qwen2Model(nn.Module):
    """Build pipeline graph for Qwen2 model"""

    def __init__(
        self,
        config: Qwen2Config,
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
            module_input_keys={"input_ids", "input_mask"},
            module_output_keys={
                "noisy_input_ids", 
                "input_mask_updated", 
                "mask_positions", 
                "noise_weights", 
                "block_mask",
                "diffusion_mode",
                "t_per_position"
            },
        )

        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=Embedding,
            module_kwargs={
                "config": config,
                "parallel_config": parallel_config,
                "tp_pg": parallel_context.tp_pg,
            },
            module_input_keys={"input_ids"},
            module_output_keys={"input_embeds"},
        )

        # Create decoder layers
        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=Qwen2DecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "cp_pg": parallel_context.cp_pg,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "cu_seqlens", "diffusion_mode", "block_mask"},
                    module_output_keys={"hidden_states", "cu_seqlens", "diffusion_mode", "block_mask"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=TritonRMSNorm if config._fused_rms_norm else RMSNorm,
            module_kwargs={"hidden_size": config.hidden_size, "eps": config.rms_norm_eps},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Return sharded logits that will need to be gathered
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                "mode": self.tp_mode,
                "async_communication": tp_linear_async_communication,
                "tp_recompute_allgather": parallel_config.tp_recompute_allgather,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer] = None,
    ):
        assert input_mask is None, f"input_mask is not None: {input_mask}"
        # Debug: Check for EOS/BOS tokens in input_ids
        if DEBUG and torch.distributed.get_rank() == 0:
            bos_id = self.config.bos_token_id
            eos_id = self.config.eos_token_id
            
            check_ids = input_ids
            if isinstance(check_ids, torch.Tensor):
                num_bos = (check_ids == bos_id).sum().item()
                num_eos = (check_ids == eos_id).sum().item()
                
                print(f"DEBUG: Input Analysis - BOS({bos_id}): {num_bos}, EOS({eos_id}): {num_eos}, Shape: {check_ids.shape}")
                if num_eos == 0 and num_bos == 0:
                    print("DEBUG: WARNING - No BOS/EOS found in batch! Documents might be merged or truncated.")

        diff_output = self.diffusion_block(input_ids=input_ids, input_mask=input_mask)
        noisy_input_ids = diff_output["noisy_input_ids"]
             
        output = self.token_position_embeddings(input_ids=noisy_input_ids)
        
        cu_seqlens: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
        if isinstance(noisy_input_ids, torch.Tensor) and noisy_input_ids.numel() > 0:
            # Compute cu_seqlens from noisy_input_ids shape
            # Assume each batch item is a single contiguous sequence
            batch_size, seq_length = noisy_input_ids.shape
            total_tokens = batch_size * seq_length
            
            # Create cu_seqlens assuming one sequence per batch item
            cu_seqlens = torch.arange(0, total_tokens + 1, seq_length, dtype=torch.int32, device=noisy_input_ids.device)

            if self.config._attn_implementation == "llama3_ring_attention":
                local_sequence_length = seq_length
                total_sequence_length = total_tokens
                assert total_sequence_length == local_sequence_length * self.parallel_context.cp_pg.size()
                assert total_sequence_length % (2 * self.parallel_context.cp_pg.size()) == 0
                (
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                local_k_slice,
                ) = llama3_flash_attn_prepare_cu_seqlens(
                    cu_seqlens,
                    causal=True,
                    rank=self.parallel_context.cp_pg.rank(),
                    world_size=self.parallel_context.cp_pg.size(),
                )
                cu_seqlens = {
                    "cu_seqlens_q": cu_seqlens_q,
                    "cu_seqlens_k": cu_seqlens_k,
                    "max_seqlen_q": max_seqlen_q,
                    "max_seqlen_k": max_seqlen_k,
                    "local_k_slice": local_k_slice,
                }
        
        decoder_states = {
            "hidden_states": output["input_embeds"],
            "cu_seqlens": cu_seqlens,
            "diffusion_mode": diff_output["diffusion_mode"],
            "block_mask": diff_output["block_mask"],
        }

        for decoder_layer in self.decoder:
            decoder_states = decoder_layer(**decoder_states)

        hidden_states = self.final_layer_norm(input=decoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        return sharded_logits, diff_output["diffusion_mode"], diff_output["mask_positions"], diff_output["noise_weights"], diff_output["t_per_position"]


    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model for load balancing."""
        model_config = self.config
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            # Self-attention (qkv proj + attn out) + MLP
            Qwen2DecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 3 * d_ff * model_config.hidden_size,
            # Final LM head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for the model"""
        world_size = self.parallel_context.world_pg.size()

        # Get number of KV heads, accounting for potential absence in config
        try:
            num_key_value_heads = self.config.num_key_value_heads
        except AttributeError:
            num_key_value_heads = self.config.num_attention_heads

        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=num_key_value_heads,
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
    # type: (Tensor, Tensor, torch.dtype) -> Tensor
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()


class Loss(nn.Module):
    def __init__(self, tp_pg: dist.ProcessGroup, mask_token_id: Optional[int] = None):
        super().__init__()
        self.tp_pg = tp_pg
        self._debug_counter = 0  # Counter for debug logging
        self.mask_token_id = mask_token_id

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [batch_size*seq_length, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
        mask_positions: Optional[torch.Tensor] = None,
        noise_weights: Optional[torch.Tensor] = None,
        t_per_position: Optional[torch.Tensor] = None,
        diffusion_mode: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        sharded_logits = sharded_logits.view(label_ids.shape[0], label_ids.shape[1], -1)
        
        if diffusion_mode is not None and diffusion_mode.item() > 0:
            mode = diffusion_mode.item()
            loss = sharded_cross_entropy(
                sharded_logits.transpose(0, 1), # [L, B, V] -> [B, L, V] ... wait, sharded_cross_entropy takes [L, B, V]
                label_ids.transpose(0, 1).contiguous(),
                group=self.tp_pg,
                dtype=torch.float
            ).transpose(0, 1) # [B, L]

            if mode == 2: # Block Diffusion
                mask = (label_ids == self.mask_token_id)
                weights = (1.0 / (t_per_position + 1e-3)).type_as(sharded_logits)
                loss = (loss * weights * mask).sum() / label_ids.size(0)

                return {"loss": loss}
            elif mode == 1: # Vanilla Diffusion
                 # Slice to align targets: we need to check if the TARGET (next token) was masked.
                 # Input: x_0, ..., x_{S-1}. Labels: x_1, ..., x_S.
                 # Mask positions corresponds to Input (x_0...x_{S-1}).
                 # We want to know if x_1...x_S were masked.
                 # We have mask status for x_1...x_{S-1} (from mask_positions[1:]).
                 # We DO NOT have mask status for x_S (it wasn't in input).
                 # So we must drop the last prediction step.
                 
                 logits_sliced = sharded_logits[:, :-1, :] # Predicts x_1...x_{S-1}
                 labels_sliced = label_ids[:, :-1]         # x_1...x_{S-1}
                 mask_sliced = mask_positions[:, 1:]       # Mask status of x_1...x_{S-1}
                 
                 loss = sharded_cross_entropy(
                    logits_sliced.transpose(0, 1),
                    labels_sliced.transpose(0, 1).contiguous(),
                    group=self.tp_pg,
                    dtype=torch.float
                 ).transpose(0, 1) # [B, S-1]

                 mask = mask_sliced.float().type_as(loss)
                 dsigma = noise_weights # [B]
                 
                 loss_per_token = loss * mask
                 if dsigma.dim() == 1:
                     dsigma = dsigma.unsqueeze(1)
                 # ensure taht disgma is not a single number 
                 if dsigma.dim() == 0:
                    raise ValueError("dsigma must be a tensor, not a single number")
                 
                 loss_weighted = (loss_per_token.sum(dim=1) * dsigma.squeeze()).sum()
                 
                 total_masked = mask.sum()
                 if total_masked > 0:
                     loss_val = loss_weighted / total_masked
                 else:
                     loss_val = torch.tensor(0.0, device=loss.device, requires_grad=True)
                 return {"loss": loss_val}
        
        loss = sharded_cross_entropy(sharded_logits.transpose(0, 1), label_ids.transpose(0, 1).contiguous(), group=self.tp_pg, dtype=torch.float).transpose(0, 1)
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        return {"loss": loss}


class LossWithZLoss(Loss):
    def __init__(self, tp_pg: dist.ProcessGroup, z_loss_coefficient: float):
        super().__init__(tp_pg)
        self.z_loss_coef = z_loss_coefficient

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [batch_size*seq_length, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        sharded_logits = sharded_logits.view(label_ids.shape[0], label_ids.shape[1], -1)
        loss, z_loss = sharded_cross_entropy(
            sharded_logits, label_ids.contiguous(), group=self.tp_pg, dtype=torch.float, z_loss_coef=self.z_loss_coef
        )
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        z_loss = masked_mean(z_loss.detach(), label_mask, dtype=torch.float)
        return {"loss": loss, "z_loss": z_loss}

from nanotron.logging import LoggingCollectorMixin
class Qwen2ForTraining(NanotronModel, LoggingCollectorMixin):
    def __init__(
        self,
        config: Qwen2Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.model = Qwen2Model(config=config, parallel_context=parallel_context, parallel_config=parallel_config)

        # Choose the appropriate loss class based on config
        loss_kwargs = {
            "tp_pg": parallel_context.tp_pg,
            "mask_token_id": config.diffusion_config.mask_token_id,
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
                "t_per_position",
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
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer] = None,
        position_ids: Union[torch.Tensor, TensorPointer] = None,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        sharded_logits, diffusion_mode, mask_positions, noise_weights, t_per_position = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        loss = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
            mask_positions=mask_positions,
            noise_weights=noise_weights,
            t_per_position=t_per_position,
            diffusion_mode=diffusion_mode,
        )
        if self.config.z_loss_enabled:
            return {"loss": loss["loss"], "z_loss": loss["z_loss"]}
        else:
            return {"loss": loss["loss"]}

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """Initialize model parameters randomly."""
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
            # Should be similar to ["model.token_position_embeddings.pp_block.token_embedding.weight", "model.lm_head.pp_block.weight"]
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
