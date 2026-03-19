#!/bin/bash

# Usage: ./train_unified.sh <DATASET_NAME> [MODEL_SIZE] [ADDITIONAL_ARGS]
# Example: ./train_unified.sh my_dataset 1b "--zero1 --length 2k --lr 1e-4"
# Example: ./train_unified.sh my_dataset 3b "--zero1 --length 2k --lr 1e-4"

DATASET_NAME=$1
MODEL_SIZE=${2:-1b}
ADDITIONAL_ARGS=${3:-}

if [ -z "$DATASET_NAME" ]; then
    echo "Error: Dataset name is required."
    echo "Usage: $0 <DATASET_NAME> [MODEL_SIZE] [ADDITIONAL_ARGS]"
    exit 1
fi

# Source environment
source /home/aiops/zhuty/nano_start.sh
cd /home/aiops/zhuty/nanotron

echo "--------------------------------"
echo "Starting Training"
echo "Dataset: $DATASET_NAME"
echo "Model Size: $MODEL_SIZE"
echo "Additional Args: $ADDITIONAL_ARGS"
echo "--------------------------------"

# read NNODES from environment variable
NNODES=${NUM_NODES:-1}
echo "Nodes: $NNODES"

if [ -f "/home/aiops/zhuty/THIS_IS_MY.txt" ] && [ "$NNODES" -gt 1 ]; then
    echo "THIS_IS_MY.txt exists, setting NCCL_SOCKET_IFNAME to bond0"
    export NCCL_SOCKET_IFNAME=bond0
    export NCCL_DEBUG=INFO
    export GLOO_SOCKET_IFNAME=bond0
    export TP_SOCKET_IFNAME=bond0
fi

# 0. Parse sequence length from ADDITIONAL_ARGS
SEQ_LENGTH_ARG="8k"  # default
if [[ "$ADDITIONAL_ARGS" =~ --length[[:space:]]+([0-9]+k) ]]; then
    SEQ_LENGTH_ARG="${BASH_REMATCH[1]}"
fi

# Map sequence length argument to actual values
case "$SEQ_LENGTH_ARG" in
    1k)
        export SEQUENCE_LENGTH=1024
        SCALING_FACTOR=8
        ;;
    2k)
        export SEQUENCE_LENGTH=2048
        SCALING_FACTOR=4
        ;;
    4k)
        export SEQUENCE_LENGTH=4096
        SCALING_FACTOR=2
        ;;
    8k)
        export SEQUENCE_LENGTH=8192
        SCALING_FACTOR=1
        ;;
    *)
        echo "Error: Invalid length '$SEQ_LENGTH_ARG'. Must be one of: 1k, 2k, 4k, 8k"
        exit 1
        ;;
esac

export MICRO_BATCH_SIZE=1
echo "Sequence Length: $SEQUENCE_LENGTH (${SEQ_LENGTH_ARG})"
echo "Scaling Factor: $SCALING_FACTOR"

# 1. Initial setup for parallelism calculation
GPUS_PER_NODE=8

# 2. Handle Additional Args and Suffix
SUFFIX="${DATASET_NAME}"
export DATASET_NAME

# Default Env Vars for Config
export INTRADOC=false
export IS_DIFFUSION=false
export IS_BLOCK_DIFFUSION=false
export BLOCK_SIZE=16
export MASK_TOKEN_ID=-1
export ZERO_STAGE=0
export ACCUMULATE_GRAD_IN_FP32=true
export IS_SCRATCH=false
export LEARNING_RATE=5.0e-05  # default learning rate
export USE_QKV_PACKED=true
export SHIFT=false
export TP=1  # default tensor parallelism
# Parse Tensor Parallelism FIRST (needed for DP calculation)
if [[ "$ADDITIONAL_ARGS" =~ --tp[[:space:]]+([0-9]+) ]]; then
    export TP="${BASH_REMATCH[1]}"
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    echo "  Tensor Parallelism: $TP"
fi

# Recalculate DP based on TP
TOTAL_GPUS=$(($NNODES * $GPUS_PER_NODE))
export DP=$(($TOTAL_GPUS / $TP))
export BATCH_ACCUM=$((512 / $DP * $SCALING_FACTOR))

echo "Final Parallelism Configuration:"
echo "  Total GPUs: $TOTAL_GPUS"
echo "  TP: $TP"
echo "  DP: $DP"
echo "  Batch Accumulation: $BATCH_ACCUM"

# Parse Additional Args
if [[ "$ADDITIONAL_ARGS" == *"--diffusion"* ]]; then
    export IS_DIFFUSION=true
    SUFFIX="${SUFFIX}_diff"
    export MASK_TOKEN_ID=128255
    echo "  Mode: Diffusion"
fi

# Parse Block Size (implies Block Diffusion)
if [[ "$ADDITIONAL_ARGS" =~ --block-size[[:space:]]+([0-9]+) ]]; then
    export BLOCK_SIZE="${BASH_REMATCH[1]}"
    export IS_BLOCK_DIFFUSION=true
    export USE_QKV_PACKED=true
    echo "  Mode: Block Diffusion (Size: $BLOCK_SIZE), USE_QKV_PACKED: $USE_QKV_PACKED"
    # Ensure Mask Token is set if not already (implies diffusion mode usually)
    if [ "$MASK_TOKEN_ID" -eq -1 ]; then
        export MASK_TOKEN_ID=128255
    fi
    SUFFIX="${SUFFIX}_blk${BLOCK_SIZE}"
    echo "  Mode: Block Diffusion (Size: $BLOCK_SIZE)"
fi

if [[ "$ADDITIONAL_ARGS" == *"--scratch"* ]]; then
    export IS_SCRATCH=true
    SUFFIX="${SUFFIX}-scratch"
    echo "  Mode: Scratch"
fi

if [[ "$ADDITIONAL_ARGS" == *"--intradoc"* ]]; then
    export INTRADOC=true
    SUFFIX="${SUFFIX}_intra"
    echo "  Mode: Intradoc"
fi

if [[ "$ADDITIONAL_ARGS" == *"--zero1"* ]]; then
    export ZERO_STAGE=1
    export ACCUMULATE_GRAD_IN_FP32=true
    SUFFIX="${SUFFIX}_zero1"
    echo "  Mode: Zero1"
fi

if [[ "$ADDITIONAL_ARGS" == *"--shift"* ]]; then
    export SHIFT=true
    SUFFIX="${SUFFIX}_shift"
    echo "  Mode: Autoregressive Shift (only predict masked tokens)"
fi

# Add TP to suffix if not default
if [ "$TP" -gt 1 ]; then
    SUFFIX="${SUFFIX}_tp${TP}"
fi

# Parse --init_ckpt (initialize from a specific nanotron checkpoint)
INIT_CKPT_PATH=""
if [[ "$ADDITIONAL_ARGS" =~ --init_ckpt[[:space:]]+([^[:space:]]+) ]]; then
    INIT_CKPT_PATH="${BASH_REMATCH[1]}"
    
    # Check if the checkpoint path ends with _hf (HuggingFace format)
    if [[ "$INIT_CKPT_PATH" == *_hf ]]; then
        CONVERTED_PATH="${INIT_CKPT_PATH}_converted"
        
        # Check if conversion is needed
        if [ ! -d "$CONVERTED_PATH" ] || [ -z "$(ls -A "$CONVERTED_PATH" 2>/dev/null)" ]; then
            echo "================================"
            echo "HuggingFace checkpoint detected: $INIT_CKPT_PATH"
            echo "Converting to Nanotron format: $CONVERTED_PATH"
            echo "================================"
            
            # Run conversion
            torchrun --nproc_per_node=1 examples/llama/convert_hf_to_nanotron.py \
                --checkpoint_path="$INIT_CKPT_PATH" \
                --save_path="$CONVERTED_PATH"
            
            if [ $? -ne 0 ]; then
                echo "Error: Failed to convert HuggingFace checkpoint to Nanotron format"
                exit 1
            fi
            
            echo "Conversion completed successfully!"
            echo "================================"
        else
            echo "  Using existing converted checkpoint: $CONVERTED_PATH"
        fi
        
        # Use the converted path
        INIT_CKPT_PATH="$CONVERTED_PATH"
    fi
    
    # Derive suffix from last two path components, removing "checkpoints-" prefix
    INIT_CKPT_PARENT=$(basename "$(dirname "$INIT_CKPT_PATH")")
    INIT_CKPT_STEP=$(basename "$INIT_CKPT_PATH")
    INIT_CKPT_SUFFIX="${INIT_CKPT_PARENT}_${INIT_CKPT_STEP}"
    INIT_CKPT_SUFFIX="${INIT_CKPT_SUFFIX#checkpoints-}"  # Remove "checkpoints-" prefix
    SUFFIX="${SUFFIX}_from_${INIT_CKPT_SUFFIX}"
    echo "  Init Checkpoint: $INIT_CKPT_PATH"
    echo "  Init Checkpoint Suffix: $INIT_CKPT_SUFFIX"
fi

# Parse Learning Rate
if [[ "$ADDITIONAL_ARGS" =~ --lr[[:space:]]+([0-9.eE+-]+) ]]; then
    export LEARNING_RATE="${BASH_REMATCH[1]}"
    echo "  Learning Rate: $LEARNING_RATE"
fi

# Append sequence length to suffix if not default (8k)
if [ "$SEQ_LENGTH_ARG" != "8k" ]; then
    SUFFIX="${SUFFIX}-${SEQ_LENGTH_ARG}"
    echo "  Sequence Length: ${SEQ_LENGTH_ARG}"
fi

# Append learning rate to suffix if not default (5.0e-05)
if [ "$LEARNING_RATE" != "5.0e-05" ]; then
    SUFFIX="${SUFFIX}_lr${LEARNING_RATE}"
    echo "  Custom Learning Rate: ${LEARNING_RATE}"
fi

# Append node count to suffix if > 1 for clarity
if [ "$NNODES" -gt 1 ]; then
    SUFFIX="${SUFFIX}_${NNODES}node"
fi

export SUFFIX
echo "Run Suffix: $SUFFIX"



# 3. Setup Config and Data based on Model Size
case "$MODEL_SIZE" in
    1b)
        CONFIG_FILE="examples/config_llama32_1b_continual_template.yaml"
        DEFAULT_CHECKPOINT="/home/aiops/zhuty/nanotron/checkpoints/llama32-1b-nt"
        DEFAULT_MAX_STEP=25000
        ;;
    3b)
        CONFIG_FILE="examples/config_llama32_3b_continual_template.yaml"
        DEFAULT_CHECKPOINT="/home/aiops/zhuty/nanotron/checkpoints/llama32-3b-nt"
        DEFAULT_MAX_STEP=12500
        ;;
    *)
        echo "Error: Invalid model size '$MODEL_SIZE'. Must be one of: 1b, 3b"
        exit 1
        ;;
esac

# Allow override via environment variable, otherwise use model-specific default
if [ -z "$MAX_STEP" ]; then
    export MAX_STEP=$DEFAULT_MAX_STEP
fi
echo "Max Training Steps: $MAX_STEP"

DATASET_FOLDER="/home/aiops/zhuty/cont_data/${DATASET_NAME}/llama_tokenized"

# if from scratch set the key and value
if [ "$IS_SCRATCH" = true ]; then
    export INIT_KEY="std"
    export INIT_VALUE=0.025
elif [ -n "$INIT_CKPT_PATH" ]; then
    export INIT_KEY="path"
    export INIT_VALUE="$INIT_CKPT_PATH"
    echo "  Initializing from checkpoint: $INIT_CKPT_PATH"
else
    export INIT_KEY="path"
    export INIT_VALUE="$DEFAULT_CHECKPOINT"
fi


if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$DATASET_FOLDER" ]; then
    echo "Error: Dataset folder not found: $DATASET_FOLDER"
    exit 1
fi

# 4. Distributed Setup
TORCHRUN_ARGS="--nproc_per_node=$GPUS_PER_NODE"

# Allow override via environment variable, otherwise use defaults
if [ -z "$CHECKPOINT_INTERVAL" ]; then
    export CHECKPOINT_INTERVAL=500
    
    if [ "$NNODES" -gt 1 ]; then
        # SCALE UP the checkpoint interval for multi-node
        export CHECKPOINT_INTERVAL=$((500 * $NNODES))
    fi
fi

echo "  Checkpoint Interval: $CHECKPOINT_INTERVAL"

if [ "$NNODES" -gt 1 ]; then
    echo "--------------------------------"
    echo "Setting up Multi-Node Environment"

    # Assume existing env vars or set defaults
    : "${MASTER_ADDR:=localhost}"
    : "${MASTER_PORT:=29500}"
    : "${RANK:=0}"

    # Set NCCL network interface if not already set
    if [ -z "$NCCL_SOCKET_IFNAME" ]; then
        # Auto-detect the network interface for inter-node communication
        # Try common interface names in order of preference
        if ip link show eth0 &>/dev/null; then
            export NCCL_SOCKET_IFNAME=eth0
            echo "  Auto-detected network interface: eth0"
        elif ip link show ib0 &>/dev/null; then
            export NCCL_SOCKET_IFNAME=ib0
            echo "  Auto-detected network interface: ib0"
        elif ip link show bond0 &>/dev/null; then
            export NCCL_SOCKET_IFNAME=bond0
            echo "  Auto-detected network interface: bond0"
        else
            echo "  WARNING: Could not auto-detect network interface. NCCL may fail."
            echo "  Please set NCCL_SOCKET_IFNAME environment variable manually."
        fi
    else
        echo "  Using NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
    fi

    # Set GLOO interface to match NCCL
    if [ -n "$NCCL_SOCKET_IFNAME" ]; then
        export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
        export TP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
    fi

    # Resolve hostname to IP. Only run this in MY
    if [ -f "/home/aiops/zhuty/THIS_IS_MY.txt" ]; then
        echo "  Resolving master hostname to IP address..."
        RESOLVED_IP=$(python3 -c "import socket; print(socket.gethostbyname('$MASTER_ADDR'))" 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$RESOLVED_IP" ]; then
            echo "  Resolved $MASTER_ADDR -> $RESOLVED_IP"
            MASTER_ADDR=$RESOLVED_IP
        else
            echo "  WARNING: Could not resolve hostname, using as-is: $MASTER_ADDR"
        fi
    fi

    echo "  Master: $MASTER_ADDR:$MASTER_PORT"
    echo "  Rank: $RANK"

    # Connectivity Check
    if [ "$RANK" -ne 0 ]; then
        echo "  Checking connectivity to master $MASTER_ADDR:$MASTER_PORT..."
        if python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(10); result = s.connect_ex(('$MASTER_ADDR', int('$MASTER_PORT'))); exit(result)"; then
             echo "  SUCCESS: Connected to master $MASTER_ADDR:$MASTER_PORT"
        else
             echo "  ERROR: Could not connect to master $MASTER_ADDR:$MASTER_PORT"
             echo "         Possible causes:"
             echo "         1. Master node is not running yet."
             echo "         2. Firewall is blocking port $MASTER_PORT."
             echo "         3. Master is listening on localhost instead of $MASTER_ADDR (check MASTER_ADDR on master)."
             # We don't exit here to allow torchrun to retry, but this gives a clear warning.
        fi
    fi

    TORCHRUN_ARGS="$TORCHRUN_ARGS --nnodes=$NNODES --node_rank=$RANK --rdzv_id=nanotron_job --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    # if on MY use different setups 
    if [ -f "/home/aiops/zhuty/THIS_IS_MY.txt" ]; then
        TORCHRUN_ARGS="$TORCHRUN_ARGS --nnodes=$NNODES --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
    
        # Force IPv4 to avoid connectivity issues
        export NCCL_SOCKET_FAMILY=AF_INET
        export GLOO_SOCKET_FAMILY=AF_INET
    fi

    # Additional NCCL settings for stability
    export NCCL_DEBUG=INFO
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
    export NCCL_NET_GDR_LEVEL=2  # Enable GPU Direct RDMA
    export NCCL_ASYNC_ERROR_HANDLING=1  # Better error handling
    export NCCL_TIMEOUT=1800  # 30 minutes timeout for slow networks
    
    # Force IPv4 for all multi-node setups to avoid ambiguity
    export NCCL_SOCKET_FAMILY=AF_INET
    export GLOO_SOCKET_FAMILY=AF_INET
fi

echo "--------------------------------"
echo "Launching torchrun..."
echo "Command: torchrun $TORCHRUN_ARGS run_train.py --config-file $CONFIG_FILE"
echo "--------------------------------"

torchrun $TORCHRUN_ARGS run_train.py --config-file $CONFIG_FILE
