#!/bin/bash
# Run Dream diffusion evaluation on a HuggingFace checkpoint
# Usage: ./run_dream_eval.sh /path/to/checkpoint [task] [batch_size] [mask_token_id] [num_gpus]

set -e

CKPT_PATH="$1"
TASK="${2:-gsm8k_cot}"
BATCH_SIZE="${3:-16}"
MASK_TOKEN_ID="${4:-128255}"
# Read the number of GPUs from nvidia-smi
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# Validate input
if [ -z "$CKPT_PATH" ]; then
    echo "Usage: $0 <checkpoint_path> [task] [batch_size] [mask_token_id]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path  Path to HF checkpoint (required)"
    echo "  task             Eval task: gsm8k_cot, minerva_math, bbh, humaneval (default: gsm8k_cot)"
    echo "  batch_size       Batch size per GPU (default: 8)"
    echo "  mask_token_id    Mask token ID used in training (default: 128255)"
    echo "  num_gpus         Number of GPUs to use (default: 8)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/checkpoint gsm8k_cot 8 128255 8"
    exit 1
fi

# Remove trailing slash if present
CKPT_PATH="${CKPT_PATH%/}"

if [ ! -d "$CKPT_PATH" ]; then
    echo "Error: Checkpoint path does not exist: $CKPT_PATH"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/dream_model_template"
DREAM_EVAL_DIR="/home/aiops/zhuty/Dream/eval"

# Create a copy with _dream suffix to avoid modifying original
DREAM_CKPT_PATH="${CKPT_PATH}_dream"

echo "========================================"
echo "Dream Diffusion Evaluation"
echo "========================================"
echo "Original checkpoint: $CKPT_PATH"
echo "Dream checkpoint:    $DREAM_CKPT_PATH"
echo "Task: $TASK"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Number of GPUs: $NUM_GPUS"
echo "Mask token ID: $MASK_TOKEN_ID"
echo "========================================"

# Step 1: Create Dream checkpoint copy
if [ -d "$DREAM_CKPT_PATH" ]; then
    echo "[1/3] Dream checkpoint already exists, reusing..."
else
    echo "[1/3] Creating Dream checkpoint copy..."
    cp -r "$CKPT_PATH" "$DREAM_CKPT_PATH"
    echo "  Copied to $DREAM_CKPT_PATH"
fi

# Step 2: Copy Dream model files if not present
if [ ! -f "$DREAM_CKPT_PATH/modeling_dream.py" ]; then
    echo "[2/3] Copying Dream model files..."
    cp "$TEMPLATE_DIR"/*.py "$DREAM_CKPT_PATH/"
else
    echo "[2/3] Dream model files already present, skipping..."
fi

# Step 3: Update config.json
echo "[3/3] Updating configs..."
python3 << PYEOF
import json

ckpt = "$DREAM_CKPT_PATH"
mask_token_id = $MASK_TOKEN_ID

with open(f"{ckpt}/config.json", "r") as f:
    config = json.load(f)

# Check if already converted
if config.get("model_type") == "Dream":
    print("  Config already converted to Dream format")
else:
    config.update({
        "architectures": ["DreamModel"],
        "auto_map": {
            "AutoConfig": "configuration_dream.DreamConfig",
            "AutoModel": "modeling_dream.DreamModel",
            "AutoModelForCausalLM": "modeling_dream.DreamModel"
        },
        "model_type": "Dream",
        "mask_token_id": mask_token_id,
        "pad_token_id": config.get("eos_token_id", 128001),
        "attention_bias": False,
        "use_cache": False
    })
    
    with open(f"{ckpt}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  Updated config.json")

# Fix tokenizer padding
with open(f"{ckpt}/tokenizer_config.json", "r") as f:
    tok_config = json.load(f)

if tok_config.get("pad_token") is None:
    tok_config["pad_token"] = tok_config.get("eos_token", "<|end_of_text|>")
    with open(f"{ckpt}/tokenizer_config.json", "w") as f:
        json.dump(tok_config, f, indent=2)
    print("  Updated tokenizer_config.json (added pad_token)")
else:
    print("  Tokenizer already has pad_token")
PYEOF

# Step 4: Run evaluation
echo ""
echo "Running evaluation..."
echo ""

# Set task-specific parameters
case $TASK in
    gsm8k_cot)
        NSHOT=8
        MAX_TOKENS=256
        STEPS=256
        TEMP=0
        ;;
    minerva_math)
        NSHOT=4
        MAX_TOKENS=512
        STEPS=512
        TEMP=0
        ;;
    bbh)
        NSHOT=3
        MAX_TOKENS=512
        STEPS=512
        TEMP=0
        ;;
    humaneval)
        NSHOT=0
        MAX_TOKENS=512
        STEPS=512
        TEMP=0.2
        export HF_ALLOW_CODE_EVAL=1
        ;;
    *)
        NSHOT=0
        MAX_TOKENS=256
        STEPS=256
        TEMP=0
        ;;
esac

OUTPUT_DIR="$DREAM_CKPT_PATH/evals_results/${TASK}-${NSHOT}shot"
mkdir -p "$OUTPUT_DIR"

cd "$DREAM_EVAL_DIR"

# Use accelerate launch for multi-GPU evaluation
accelerate launch --num_processes "$NUM_GPUS" --multi_gpu eval.py --model dream \
    --model_args "pretrained=$DREAM_CKPT_PATH,max_new_tokens=$MAX_TOKENS,diffusion_steps=$STEPS,temperature=$TEMP,top_p=0.95,add_bos_token=false" \
    --tasks "$TASK" \
    --num_fewshot "$NSHOT" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR" \
    --log_samples

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Original checkpoint (unchanged): $CKPT_PATH"
echo "Dream checkpoint: $DREAM_CKPT_PATH"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# Print summary
RESULT_FILE=$(ls -t "$OUTPUT_DIR"/**/results_*.json 2>/dev/null | head -1)
if [ -n "$RESULT_FILE" ]; then
    echo ""
    echo "Quick summary:"
    python3 -c "
import json
with open('$RESULT_FILE') as f:
    r = json.load(f)
results = r.get('results', {})
for task, metrics in results.items():
    print(f'  {task}:')
    for k, v in metrics.items():
        if 'stderr' not in k and isinstance(v, (int, float)):
            print(f'    {k}: {v:.4f}')
"
fi
