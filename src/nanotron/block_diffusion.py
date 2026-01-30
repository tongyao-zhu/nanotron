import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_block_diffusion_attention_mask(
    batch_size: int,
    seq_len: int,
    block_size: int,
    noisy_len: int,
    device: torch.device,
    doc_boundaries: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Create attention mask for block diffusion - EXACT match to nano-block-diffusion rules.
    
    Reference implementation (from nano-block-diffusion):
        def block_diffusion_mask(b, h, q, kv):
            blk_q, blk_kv = block_id[q], block_id[kv]
            bd = (blk_q == blk_kv) & (noisy[q] == noisy[kv])  # Block Diagonal
            obc = (blk_q > blk_kv) & noisy[q] & (~noisy[kv])  # Offset Block Causal
            bc = (blk_q >= blk_kv) & (~noisy[q]) & (~noisy[kv])  # Block Causal
            same_doc = doc_id[q] == doc_id[kv]
            return same_doc & (bd | obc | bc)
    
    The input sequence is structured as: [noisy_tokens (L), clean_tokens (L)]
    
    Attention Rules (from paper section 3.1):
    1. BD (Block Diagonal): Same block AND same noise status
    2. OBC (Offset Block Causal): Noisy block i → clean blocks < i
    3. BC (Block Causal): Clean follows standard causal (position ≤)
    
    Args:
        batch_size: Batch size
        seq_len: Total sequence length (should be 2 * noisy_len)
        block_size: Size of each block (diffusion_block_size)
        noisy_len: Length of noisy sequence (first half)
        device: Device to create mask on
        doc_boundaries: Optional document boundary mask
    
    Returns:
        Boolean mask (batch_size, 1, seq_len, seq_len) where True = attend
    """
    assert seq_len == 2 * noisy_len, f"seq_len ({seq_len}) must be 2 * noisy_len ({noisy_len})"
    
    # Initialize mask (False = don't attend)
    mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=device)
    
    # Create block IDs for each position (same for noisy and clean parts)
    block_ids = torch.arange(noisy_len, device=device) // block_size
    
    # Create noisy indicator: True for first L positions, False for last L
    noisy = torch.arange(seq_len, device=device) < noisy_len
    
    # Vectorized implementation of BD, OBC, BC rules
    # Create block_id grid for all positions
    blk_q = torch.cat([block_ids, block_ids], dim=0)  # Shape: (2*L,)
    blk_kv = blk_q  # Same blocks
    
    # For each query position
    # TODO: This loop is slow for large seq_len, try to vectorize fully
    # But this is only done once per batch shape or training step if shapes are constant
    # Actually, we can fully vectorize this
    
    # Broadcasting for full grid
    blk_q_grid = blk_q.unsqueeze(1) # (2L, 1)
    blk_kv_grid = blk_kv.unsqueeze(0) # (1, 2L)
    noisy_q_grid = noisy.unsqueeze(1) # (2L, 1)
    noisy_kv_grid = noisy.unsqueeze(0) # (1, 2L)
    
    # BD: (blk_q == blk_kv) & (noisy[q] == noisy[kv])
    bd = (blk_q_grid == blk_kv_grid) & (noisy_q_grid == noisy_kv_grid)
    
    # OBC: (blk_q > blk_kv) & noisy[q] & (~noisy[kv])
    obc = (blk_q_grid > blk_kv_grid) & noisy_q_grid & (~noisy_kv_grid)
    
    # BC: (blk_q >= blk_kv) & (~noisy[q]) & (~noisy[kv])
    bc = (blk_q_grid >= blk_kv_grid) & (~noisy_q_grid) & (~noisy_kv_grid)
    
    # Combine rules: bd | obc | bc
    can_attend = bd | obc | bc
    
    # Apply to mask
    mask[0, 0, :, :] = can_attend
    # Broadcast to batch
    mask = mask.expand(batch_size, -1, -1, -1)
    
    # Apply document boundaries if provided
    if doc_boundaries is not None:
        if doc_boundaries.dim() == 2:
            doc_boundaries = doc_boundaries.unsqueeze(0).expand(batch_size, -1, -1)
        doc_boundaries = doc_boundaries.unsqueeze(1)
        mask = mask & doc_boundaries
    
    return mask


def calculate_block_ids(
    input_ids: torch.Tensor,
    block_size: int,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate block IDs and position IDs following nano-block-diffusion logic.
    
    Args:
        input_ids: Input token IDs of shape (batch_size, seq_len)
        block_size: Size of each diffusion block
        bos_token_id: Beginning-of-sequence token ID for document boundaries
        eos_token_id: End-of-sequence token ID (used when bos_token_id == eos_token_id)
    
    Returns:
        block_ids: Block ID for each position (batch_size, seq_len)
        pos_ids: Position ID within document (batch_size, seq_len)
        doc_ids: Document ID for each position (batch_size, seq_len)
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Calculate document IDs (increments at each BOS token)
    if bos_token_id is not None:
        is_bos = (input_ids == bos_token_id)
        
        # Handle special case: bos_token_id == eos_token_id (e.g., Qwen 2.5)
        if eos_token_id is not None and bos_token_id == eos_token_id:
            is_bos_filtered = torch.zeros_like(is_bos, dtype=torch.bool)
            
            # Rule 1: First token is BOS and second token is not BOS
            is_bos_filtered[:, 0] = is_bos[:, 0] & ~is_bos[:, 1]
            
            # Rule 2: Second token of consecutive BOS tokens
            is_prev_bos = is_bos[:, :-1]  # Previous position is BOS
            is_curr_bos = is_bos[:, 1:]   # Current position is BOS
            is_bos_filtered[:, 1:] = is_prev_bos & is_curr_bos
            
            is_bos = is_bos_filtered
        
        doc_ids = is_bos.cumsum(dim=1)
    else:
        # Single document per sequence
        doc_ids = torch.zeros_like(input_ids)
    
    # Calculate position IDs (reset at each BOS token)
    pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    if bos_token_id is not None:
        # Reset position at BOS tokens
        bos_positions = torch.where(is_bos, pos, torch.tensor(-1, device=device))
        last_bos_pos = bos_positions.cummax(dim=1).values
        pos_ids = pos - last_bos_pos
    else:
        pos_ids = pos
    
    # Calculate block IDs (position // block_size, unique per document)
    # Add document offset to ensure blocks from different docs are different
    block_id = pos_ids // block_size + doc_ids * seq_len
    
    # Make block IDs sequential (0, 1, 2, ...)
    # Use roll like the reference implementation to avoid -1 in first position
    block_id_rolled = torch.roll(block_id, shifts=1, dims=1)
    # Set first position to different value to ensure it's detected as new block
    block_id_rolled[:, 0] = -1
    is_new_block = (block_id != block_id_rolled)
    block_ids = is_new_block.cumsum(dim=1) - 1
    
    return block_ids, pos_ids, doc_ids


def apply_block_masking(
    tokens: torch.Tensor,
    block_ids: torch.Tensor,
    t_lower: float,
    t_upper: float,
    mask_token_id: int,
    is_training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply block-wise masking.
    
    Args:
        tokens: Input tokens of shape (batch_size, seq_len)
        block_ids: Block ID for each position of shape (batch_size, seq_len)
        t_lower: Lower bound for noise sampling (e.g., 0.3)
        t_upper: Upper bound for noise sampling (e.g., 0.8)
        mask_token_id: ID of the mask token
        is_training: If True, use [t_lower, t_upper]; if False, use [0.0, 1.0]
    
    Returns:
        masked_tokens: Tokens with masking applied
        mask_positions: Boolean mask indicating which tokens were masked
        t_per_position: Noise level per position (same within each block)
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Determine noise range
    noise_range = (t_lower, t_upper) if is_training else (0.0, 1.0)
    
    # 1. Sample random values for masking decision
    rand = torch.rand(batch_size, seq_len, device=device, dtype=torch.float32)
    
    # 2. Sample noise level by creating random tensor and indexing by block_id
    num_blocks = block_ids.max().item() + 1
    
    # Create random t values (one per block per batch)
    t_random = torch.empty(batch_size, num_blocks, device=device, dtype=torch.float32)
    t_random.uniform_(*noise_range)
    
    # Index by block_id to get t per position (same t for positions in same block)
    t = torch.gather(t_random, 1, block_ids)
    
    # 3. Mask where rand >= (1 - t)
    mask_positions = (rand >= (1 - t))
    
    # 4. Apply masking
    masked_tokens = tokens.clone()
    masked_tokens[mask_positions] = mask_token_id
    
    return masked_tokens, mask_positions, t

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
