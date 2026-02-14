import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import FlexAttention for block mask creation
from torch.nn.attention.flex_attention import create_block_mask
FLEX_ATTENTION_AVAILABLE = True
create_block_mask = torch.compile(create_block_mask, dynamic=False)

# Global flag for debug logging (to avoid repeated logs)
_attn_mask_logged = False


def transition(x_0, sigma, maskable_mask, mask_token_id):
    """
    Diffusion transition function that applies noise to the input sequence (Simple Diffusion).
    
    Args:
        x_0: Original token sequence
        sigma: Noise level (0 to 1)
        maskable_mask: Boolean mask indicating which tokens can be masked
        mask_token_id: ID of the mask token
    
    Returns:
        x_t: Noisy token sequence
    """
    move_chance = sigma
    move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t


def create_flex_block_mask(doc_id: torch.Tensor, block_id: torch.Tensor):
    """
    Create FlexAttention BlockMask for block diffusion.
    Follows the reference implementation from nano-block-diffusion.
    
    Args:
        doc_id: Document IDs of shape [L] (will be repeated for [noisy, clean])
        block_id: Block IDs of shape [L] (will be repeated for [noisy, clean])
    
    Returns:
        BlockMask object for use with flex_attention
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("FlexAttention not available. Requires PyTorch 2.5+")
    
    L = len(doc_id)
    device = doc_id.device
    
    # Repeat for [noisy, clean] sequence structure
    block_id_2l = block_id.repeat(2)  # [2L]
    doc_id_2l = doc_id.repeat(2)  # [2L]
    noisy = torch.arange(2 * L, device=device) < L  # [2L] - first L are noisy
    
    def block_diffusion_mask(b, h, q, kv):
        """Attention mask rules from https://arxiv.org/pdf/2503.09573 section 3.1"""
        blk_q, blk_kv = block_id_2l[q], block_id_2l[kv]
        
        bd = (blk_q == blk_kv) & (noisy[q] == noisy[kv])  # Block Diagonal
        obc = (blk_q > blk_kv) & noisy[q] & (~noisy[kv])  # Offset Block Causal
        bc = (blk_q >= blk_kv) & (~noisy[q]) & (~noisy[kv])  # Block Causal
        
        same_doc = doc_id_2l[q] == doc_id_2l[kv]
        return same_doc & (bd | obc | bc)
    
    S = 2 * L
    return create_block_mask(block_diffusion_mask, None, None, S, S, device=device)
