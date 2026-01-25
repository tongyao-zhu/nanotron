#!/bin/bash

# Usage: ./train_unified.sh <DATASET_NAME> [MODEL_SIZE] [NNODES] [ADDITIONAL_ARGS]
# Example: ./train_unified.sh my_dataset 1b 4 "--zero1 --length 2k --lr 1e-4"

DATASET_NAME=$1
MODEL_SIZE=${2:-1b}
ADDITIONAL_ARGS=${3:-}

if [ -z "$DATASET_NAME" ]; then
    echo "Error: Dataset name is required."
    echo "Usage: $0 <DATASET_NAME> [MODEL_SIZE] [NNODES] [ADDITIONAL_ARGS]"
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

if [ -f "/home/aiops/zhuty/THIS_IS_MY.txt" ] && [ $num_nodes -gt 1 ]; then
    echo "THIS_IS_MY.txt exists, setting NCCL_SOCKET_IFNAME to bond0"
    export NCCL_SOCKET_IFNAME=bond0
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

# 1. Calculate DP and Batch Accumulation to maintain Global Batch Size = 512
# Global BS (512) = DP * Micro_Batch * Accumulation
GPUS_PER_NODE=8
export DP=$(($NNODES * $GPUS_PER_NODE))
export BATCH_ACCUM=$((512 / $DP * $SCALING_FACTOR))

echo "Calculated Configuration:"
echo "  DP Size: $DP ($NNODES x $GPUS_PER_NODE)"
echo "  Batch Accumulation: $BATCH_ACCUM"

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
    export ACCUMULATE_GRAD_IN_FP32=false
    SUFFIX="${SUFFIX}_zero1"
    echo "  Mode: Zero1"
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



# 3. Setup Config and Data
CONFIG_FILE="examples/config_llama32_1b_continual_template.yaml"
DATASET_FOLDER="/home/aiops/zhuty/cont_data/${DATASET_NAME}/llama_tokenized"

# if from scratch set the key and value
if [ "$IS_SCRATCH" = true ]; then
    export INIT_KEY="std"
    export INIT_VALUE=0.025
else
    export INIT_KEY="path"
    export INIT_VALUE="/home/aiops/zhuty/nanotron/checkpoints/llama32-1b-nt"
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

if [ "$NNODES" -gt 1 ]; then
    echo "--------------------------------"
    echo "Setting up Multi-Node Environment"
    
    # Assume existing env vars or set defaults
    : "${MASTER_ADDR:=localhost}"
    : "${MASTER_PORT:=29500}"
    : "${RANK:=0}"

    echo "  Master: $MASTER_ADDR:$MASTER_PORT"
    echo "  Rank: $RANK"

    # Connectivity Check
    if [ "$RANK" -ne 0 ]; then
        echo "  Checking connectivity to master..."
        ping -c 3 $MASTER_ADDR || echo "  WARNING: Ping to master failed"
    fi

    TORCHRUN_ARGS="$TORCHRUN_ARGS --nnodes=$NNODES --node_rank=$RANK --rdzv_id=nanotron_job --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
    
    export NCCL_DEBUG=INFO
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
fi

echo "--------------------------------"
echo "Launching torchrun..."
echo "Command: torchrun $TORCHRUN_ARGS run_train.py --config-file $CONFIG_FILE"
echo "--------------------------------"

torchrun $TORCHRUN_ARGS run_train.py --config-file $CONFIG_FILE
