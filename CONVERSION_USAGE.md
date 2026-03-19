# Checkpoint Conversion Guide

This guide explains how to convert Nanotron checkpoints to HuggingFace format using either GPU or CPU memory.

## Overview

The conversion scripts now support both GPU and CPU devices, allowing you to choose based on your available resources.

## Important Notes

⚠️ **CPU-only conversion requires code modifications**: The Nanotron codebase has been modified to gracefully handle CPU-only environments where CUDA is not available. The following files have been updated to make flash_attn imports optional:
- `src/nanotron/nn/layer_norm.py`
- `src/nanotron/nn/rotary.py`
- `src/nanotron/nn/attention.py`

These modifications allow the conversion script to run on CPU-only machines without requiring CUDA drivers.

## Memory Requirements

### GPU Conversion (Default)
- **Llama 3.2 1B**: ~4-6 GB GPU memory
- **Llama 3.2 3B**: ~12-16 GB GPU memory  
- **OpenCoder 484M**: ~3-5 GB GPU memory

### CPU Conversion
- Uses system RAM instead of GPU memory
- Significantly slower but doesn't require GPU
- Useful when GPU memory is limited or unavailable

## Usage

### 1. Convert Single Checkpoint

#### Using GPU (default):
```bash
bash convert_to_hf.sh /path/to/checkpoint
```

#### Using CPU:
```bash
bash convert_to_hf.sh /path/to/checkpoint cpu
```

Or use the convenience script:
```bash
bash convert_to_hf_cpu.sh /path/to/checkpoint
```

### 2. Convert All Checkpoints in a Directory

#### Using GPU (default):
```bash
bash all_convert_to_hf.sh /path/to/checkpoints/dir false
```

#### Using CPU:
```bash
bash all_convert_to_hf.sh /path/to/checkpoints/dir false cpu
```

#### With automatic deletion of old checkpoints (GPU):
```bash
bash all_convert_to_hf.sh /path/to/checkpoints/dir true
```

#### With automatic deletion of old checkpoints (CPU):
```bash
bash all_convert_to_hf.sh /path/to/checkpoints/dir true cpu
```

### 3. Direct Python Usage

You can also call the conversion script directly with Python:

#### GPU conversion:
```bash
python -m examples.llama.convert_nanotron_to_hf \
    --checkpoint_path=/path/to/checkpoint \
    --save_path=/path/to/output \
    --tokenizer_name=meta-llama/Llama-3.2-1B \
    --config_cls=Qwen2Config \
    --device=cuda
```

#### CPU conversion:
```bash
python -m examples.llama.convert_nanotron_to_hf \
    --checkpoint_path=/path/to/checkpoint \
    --save_path=/path/to/output \
    --tokenizer_name=meta-llama/Llama-3.2-1B \
    --config_cls=Qwen2Config \
    --device=cpu
```

## Parameters

### all_convert_to_hf.sh
- **Argument 1**: Base path containing checkpoint directories
- **Argument 2**: Delete old checkpoints after conversion (`true` or `false`)
- **Argument 3**: Device to use (`cuda` or `cpu`, defaults to `cuda`)

### convert_to_hf.sh
- **Argument 1**: Path to checkpoint directory
- **Argument 2**: Device to use (`cuda` or `cpu`, defaults to `cuda`)

## Performance Considerations

### GPU Conversion
- ✅ Fast (typically 1-5 minutes per checkpoint)
- ✅ Efficient for multiple conversions
- ❌ Requires GPU with sufficient memory

### CPU Conversion
- ✅ No GPU memory required
- ✅ Works on any machine with sufficient RAM
- ❌ Significantly slower (10-30 minutes per checkpoint)
- ❌ May require substantial system RAM

## Troubleshooting

### Out of GPU Memory
If you encounter GPU out-of-memory errors:
1. Use CPU conversion instead: `bash convert_to_hf.sh /path/to/checkpoint cpu`
2. Close other GPU-using processes
3. Convert checkpoints one at a time

### Out of CPU Memory
If CPU conversion runs out of memory:
1. Close other applications to free RAM
2. Consider using a machine with more RAM
3. Try GPU conversion if available

### Slow CPU Conversion
CPU conversion is inherently slower. To speed up:
1. Use GPU if available
2. Convert checkpoints in parallel on different machines
3. Ensure you have sufficient RAM to avoid swapping

## Examples

### Convert all checkpoints in a training run using CPU:
```bash
bash all_convert_to_hf.sh /data/checkpoints/llama-1b-run1 false cpu
```

### Convert single checkpoint and delete old ones using GPU:
```bash
bash all_convert_to_hf.sh /data/checkpoints/llama-3b-run2 true cuda
```

### Convert specific checkpoint using CPU:
```bash
bash convert_to_hf_cpu.sh /data/checkpoints/llama-1b-run1/10000
```
