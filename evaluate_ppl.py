#!/usr/bin/env python3
"""
Evaluate perplexity of a HuggingFace model on a JSONL file.
Outputs a new JSONL file with loss and token_length fields.

Usage:
    python evaluate_ppl.py --test-file input.jsonl --model meta-llama/Llama-2-7b-hf --context-length 2048 --output-file output.jsonl
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPL of a HuggingFace model on a JSONL file")
    parser.add_argument("--test-file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--context-length", type=int, default=4096, help="Context length for evaluation")
    parser.add_argument("--output-file", type=str, default=None, help="Path to output JSONL file (default: input_with_loss.jsonl)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type for model")
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, device: str, dtype: str):
    """Load the model and tokenizer."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )
    
    if device != "auto":
        model = model.to(device)
    
    model.eval()
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def compute_loss_for_text(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context_length: int,
    device: str,
) -> tuple[float, int]:
    """
    Compute the average loss for a text.
    
    If the text is shorter than context_length, evaluate directly.
    If longer, split into chunks and average the loss.
    
    Returns:
        tuple: (average_loss, total_token_length)
    """
    # Tokenize the full text
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encodings.input_ids[0]  # Remove batch dimension
    total_length = len(input_ids)
    
    if total_length == 0:
        return float('nan'), 0
    
    # If text fits in context, evaluate directly
    if total_length <= context_length:
        input_ids = input_ids.unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        
        return loss, total_length
    
    # Otherwise, split into chunks with stride
    # We use non-overlapping chunks for simplicity and to avoid double-counting
    stride = context_length
    total_loss = 0.0
    total_tokens = 0
    
    for start_idx in range(0, total_length, stride):
        end_idx = min(start_idx + context_length, total_length)
        chunk_ids = input_ids[start_idx:end_idx].unsqueeze(0).to(device)
        chunk_length = end_idx - start_idx
        
        if chunk_length == 0:
            continue
        
        with torch.no_grad():
            outputs = model(chunk_ids, labels=chunk_ids)
            # Loss is averaged over tokens, so we need to weight by chunk length
            # Note: The first token doesn't contribute to loss (no prediction target)
            # So actual tokens contributing to loss is (chunk_length - 1)
            chunk_loss = outputs.loss.item()
            contributing_tokens = chunk_length - 1
            
            if contributing_tokens > 0:
                total_loss += chunk_loss * contributing_tokens
                total_tokens += contributing_tokens
    
    if total_tokens == 0:
        return float('nan'), total_length
    
    # Return average loss weighted by number of tokens
    average_loss = total_loss / total_tokens
    return average_loss, total_length


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Write a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    args = parse_args()
    
    # Set output file path
    if args.output_file is None:
        input_path = Path(args.test_file)
        args.output_file = str(input_path.parent / f"{input_path.stem}_with_loss.jsonl")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, args.device, args.dtype)
    
    # Read input data
    print(f"Reading {args.test_file}...")
    data = read_jsonl(args.test_file)
    print(f"Loaded {len(data)} examples")
    
    # Process each example
    results = []
    total_loss = 0.0
    total_tokens = 0
    
    for item in tqdm(data, desc="Evaluating"):
        text = item.get('text', '')
        
        # Compute loss
        loss, token_length = compute_loss_for_text(
            text, model, tokenizer, args.context_length, args.device
        )
        
        # Create new item without 'text' field, but with 'loss' and 'token_length'
        new_item = {k: v for k, v in item.items() if k != 'text'}
        new_item['loss'] = loss
        new_item['token_length'] = token_length
        
        # Compute PPL for display (optional)
        if not math.isnan(loss):
            ppl = math.exp(loss)
            new_item['ppl'] = ppl
            total_loss += loss * (token_length - 1)  # Weight by contributing tokens
            total_tokens += token_length - 1
        else:
            new_item['ppl'] = float('nan')
        
        results.append(new_item)
    
    # Write results
    print(f"Writing results to {args.output_file}...")
    write_jsonl(results, args.output_file)
    
    # Print summary statistics
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        avg_ppl = math.exp(avg_loss)
        print(f"\n=== Summary ===")
        print(f"Total examples: {len(data)}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Average PPL: {avg_ppl:.4f}")
    
    # Also print top-10 highest loss examples
    valid_results = [r for r in results if not math.isnan(r['loss'])]
    sorted_results = sorted(valid_results, key=lambda x: x['loss'], reverse=True)
    
    print(f"\n=== Top 10 Highest Loss Examples ===")
    for i, item in enumerate(sorted_results[:10]):
        print(f"{i+1}. Loss: {item['loss']:.4f}, PPL: {item['ppl']:.2f}, Tokens: {item['token_length']}")
        # Print any other identifying fields (excluding loss, ppl, token_length)
        other_fields = {k: v for k, v in item.items() if k not in ['loss', 'ppl', 'token_length']}
        if other_fields:
            print(f"   Metadata: {other_fields}")


if __name__ == "__main__":
    main()

