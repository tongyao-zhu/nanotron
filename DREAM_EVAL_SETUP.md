# Converting HF Checkpoints for Dream Diffusion Evaluation

## Quick Start

For a new checkpoint at `$CKPT_PATH`:

```bash
# 1. Copy Dream model files (already downloaded)
cp /home/aiops/zhuty/nanotron/dream_model_template/*.py $CKPT_PATH/

# 2. Update config.json
python3 << PYEOF
import json
ckpt = "$CKPT_PATH"
with open(f"{ckpt}/config.json", "r") as f:
    config = json.load(f)

config.update({
    "architectures": ["DreamModel"],
    "auto_map": {
        "AutoConfig": "configuration_dream.DreamConfig",
        "AutoModel": "modeling_dream.DreamModel",
        "AutoModelForCausalLM": "modeling_dream.DreamModel"
    },
    "model_type": "Dream",
    "mask_token_id": 128255,  # Your mask token ID
    "attention_bias": False,
    "use_cache": False
})

with open(f"{ckpt}/config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"Updated {ckpt}/config.json")
PYEOF

# 3. Fix tokenizer padding
python3 << PYEOF
import json
ckpt = "$CKPT_PATH"
with open(f"{ckpt}/tokenizer_config.json", "r") as f:
    config = json.load(f)
config["pad_token"] = config.get("eos_token", "<|end_of_text|>")
with open(f"{ckpt}/tokenizer_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"Updated {ckpt}/tokenizer_config.json")
PYEOF
```

## Run Evaluation

```bash
cd /home/aiops/zhuty/Dream/eval

# GSM8K (math)
python eval.py --model dream \
    --model_args "pretrained=$CKPT_PATH,max_new_tokens=256,diffusion_steps=256,temperature=0,top_p=0.95,add_bos_token=false" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 8 \
    --output_path $CKPT_PATH/evals_results/gsm8k-8shot \
    --log_samples

# Other tasks: minerva_math, bbh, humaneval, mbpp
```

## Key Files Modified

1. **config.json** - Changed architecture to DreamModel, added mask_token_id
2. **tokenizer_config.json** - Added pad_token
3. **modeling_dream.py** - Fixed attention_bias (already done in template)
4. **configuration_dream.py** - Added attention_bias param (already done)

## Notes

- Dream model files location: `/home/aiops/zhuty/nanotron/dream_model_template/` (or re-download from `Dream-org/Dream-v0-Base-7B`)
- mask_token_id=128255 for Llama-3.2 with vocab_size=128256
- Weights are compatible because Llama and Dream use same layer naming

## Reference Checkpoint

Working example: `checkpoints-llama32-1b-mathprosfeos_diff_intra-4k_4node/10500_hf/`
