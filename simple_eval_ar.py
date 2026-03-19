#!/usr/bin/env python3
"""
Simple autoregressive evaluation script for language models on GSM-8K dataset.
Supports few-shot prompting with configurable number of shots and pass@k metrics.

Usage Examples:
    # 8-shot evaluation (default, greedy decoding)
    python simple_eval_ar.py --model_path /path/to/model

    # Zero-shot evaluation
    python simple_eval_ar.py --model_path /path/to/model --num_shots 0

    # Multiple sampling with pass@k metrics (pass@1, pass@10, pass@32)
    python simple_eval_ar.py \
        --model_path /path/to/model \
        --num_shots 8 \
        --sampling_n 32 \
        --temperature 0.8 \
        --top_p 0.95

    # Quick test with limited samples
    python simple_eval_ar.py --model_path /path/to/model --num_shots 8 --max_samples 100

    # Save detailed results to JSON
    python simple_eval_ar.py --model_path /path/to/model --output_file results/gsm8k_eval.json

    # Full example with pass@k evaluation
    python simple_eval_ar.py \
        --model_path /path/to/model \
        --num_shots 8 \
        --sampling_n 100 \
        --temperature 0.7 \
        --max_samples 500 \
        --output_file results/gsm8k_pass_at_k.json
"""

import argparse
import json
import re
import torch
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.
    
    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: k value for pass@k
    
    Returns:
        pass@k probability
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod([(n - c - i) / (n - i) for i in range(k)])


def extract_answer(text: str) -> Optional[float]:
    """
    Extract numerical answer from model output.
    Looks for patterns like "#### 123" or final number in the text.
    """
    # GSM-8K format: answer after ####
    match = re.search(r'####\s*(-?\d+\.?\d*)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    # Fallback: look for last number in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None


def create_few_shot_prompt(question: str, examples: List[Dict], num_shots: int) -> str:
    """
    Create a few-shot prompt with examples from the dataset.
    """
    prompt_parts = []
    
    # Add few-shot examples
    for i, example in enumerate(examples[:num_shots]):
        prompt_parts.append(f"Question: {example['question']}")
        prompt_parts.append(f"Answer: {example['answer']}\n")
    
    # Add the actual question
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("Answer:")
    
    return "\n".join(prompt_parts)


def evaluate_gsm8k(
    model_path: str,
    num_shots: int = 8,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    sampling_n: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_file: Optional[str] = None
):
    """
    Evaluate a model on GSM-8K dataset using autoregressive generation.
    Supports multiple sampling for pass@k metrics.
    
    Args:
        model_path: Path to the model (HuggingFace format)
        num_shots: Number of few-shot examples (0 for zero-shot)
        max_samples: Maximum number of samples to evaluate (None for all)
        batch_size: Batch size for inference
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 for greedy decoding)
        top_p: Nucleus sampling parameter
        sampling_n: Number of samples to generate per question (for pass@k)
        device: Device to run on
        output_file: Optional path to save detailed results
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)
    
    model.eval()
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading GSM-8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Use training data for few-shot examples
    few_shot_examples = [train_data[i] for i in range(min(num_shots, len(train_data)))]
    
    # Prepare test samples
    test_samples = list(test_data)
    if max_samples:
        test_samples = test_samples[:max_samples]
    
    if sampling_n > 1:
        print(f"Evaluating on {len(test_samples)} samples with {num_shots}-shot prompting...")
        print(f"Generating {sampling_n} samples per question for pass@k metrics...")
    else:
        print(f"Evaluating on {len(test_samples)} samples with {num_shots}-shot prompting...")
    
    results = []
    
    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm(test_samples)):
            # Create prompt
            prompt = create_few_shot_prompt(sample["question"], few_shot_examples, num_shots)
            
            # Extract ground truth answer
            gt_answer_text = sample["answer"]
            gt_answer = extract_answer(gt_answer_text)
            
            # Generate multiple samples for this question
            all_predictions = []
            all_generated_texts = []
            
            for sample_round in range(0, sampling_n, batch_size):
                current_batch_size = min(batch_size, sampling_n - sample_round)
                
                # Replicate prompt for batch
                prompts = [prompt] * current_batch_size
                
                # Tokenize
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(model.device)
                
                # Generate
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                
                if temperature > 0 or sampling_n > 1:
                    gen_kwargs.update({
                        "do_sample": True,
                        "temperature": temperature if temperature > 0 else 0.8,
                        "top_p": top_p,
                    })
                
                outputs = model.generate(**inputs, **gen_kwargs)
                
                # Decode outputs
                for output in outputs:
                    generated_text = tokenizer.decode(
                        output[inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    pred_answer = extract_answer(generated_text)
                    
                    all_generated_texts.append(generated_text)
                    all_predictions.append(pred_answer)
            
            # Check which predictions are correct
            correct_mask = []
            for pred_answer in all_predictions:
                is_correct = False
                if gt_answer is not None and pred_answer is not None:
                    is_correct = abs(gt_answer - pred_answer) < 1e-3
                correct_mask.append(is_correct)
            
            num_correct = sum(correct_mask)
            
            # Store result
            result = {
                "question": sample["question"],
                "ground_truth": gt_answer_text,
                "gt_answer": gt_answer,
                "generated_texts": all_generated_texts,
                "pred_answers": all_predictions,
                "correct_mask": correct_mask,
                "num_correct": num_correct,
                "num_samples": sampling_n
            }
            results.append(result)
            
            # Print first few examples
            if sample_idx < 3:
                print(f"\n{'='*80}")
                print(f"Example {sample_idx + 1}:")
                print(f"Question: {sample['question']}")
                print(f"Ground Truth: {gt_answer}")
                print(f"Samples generated: {sampling_n}")
                print(f"Correct samples: {num_correct}/{sampling_n}")
                if all_generated_texts:
                    print(f"First generated: {all_generated_texts[0][:200]}...")
                    print(f"First predicted: {all_predictions[0]}")
    
    # Calculate metrics
    total_questions = len(results)
    
    if sampling_n == 1:
        # Simple accuracy for greedy decoding
        correct = sum(r["num_correct"] for r in results)
        accuracy = correct / total_questions if total_questions > 0 else 0.0
        
        print(f"\n{'='*80}")
        print(f"Evaluation Results:")
        print(f"  Model: {model_path}")
        print(f"  Dataset: GSM-8K")
        print(f"  Shots: {num_shots}")
        print(f"  Total samples: {total_questions}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"{'='*80}\n")
        
        metrics = {
            "model_path": model_path,
            "num_shots": num_shots,
            "total": total_questions,
            "correct": correct,
            "accuracy": accuracy,
        }
    else:
        # Calculate pass@k metrics
        print(f"\n{'='*80}")
        print(f"Evaluation Results:")
        print(f"  Model: {model_path}")
        print(f"  Dataset: GSM-8K")
        print(f"  Shots: {num_shots}")
        print(f"  Total questions: {total_questions}")
        print(f"  Samples per question: {sampling_n}")
        print()
        
        # Calculate pass@k for various k values
        k_values = [1, 10, 32, 50, 100]
        k_values = [k for k in k_values if k <= sampling_n]
        
        pass_at_k_metrics = {}
        for k in k_values:
            pass_at_k_sum = 0.0
            for result in results:
                n = result["num_samples"]
                c = result["num_correct"]
                pass_at_k_sum += calculate_pass_at_k(n, c, k)
            
            pass_at_k_avg = pass_at_k_sum / total_questions if total_questions > 0 else 0.0
            pass_at_k_metrics[f"pass@{k}"] = pass_at_k_avg
            print(f"  pass@{k}: {pass_at_k_avg:.2%}")
        
        print(f"{'='*80}\n")
        
        metrics = {
            "model_path": model_path,
            "num_shots": num_shots,
            "total_questions": total_questions,
            "sampling_n": sampling_n,
            **pass_at_k_metrics,
        }
    
    # Save detailed results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                **metrics,
                "results": results
            }, f, indent=2)
        
        print(f"Detailed results saved to {output_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate language models on GSM-8K using autoregressive generation"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (HuggingFace format)"
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=8,
        help="Number of few-shot examples (default: 8, use 0 for zero-shot)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy decoding, use 0.7-0.8 for sampling)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter (default: 1.0)"
    )
    parser.add_argument(
        "--sampling_n",
        type=int,
        default=1,
        help="Number of samples to generate per question for pass@k metrics (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save detailed results (JSON format)"
    )
    
    args = parser.parse_args()
    
    evaluate_gsm8k(
        model_path=args.model_path,
        num_shots=args.num_shots,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sampling_n=args.sampling_n,
        device=args.device,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()
