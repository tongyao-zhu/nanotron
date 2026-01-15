#!/bin/bash

echo $MASTER_ADDR "is the master address"
echo $MASTER_PORT "is the master port"
echo $RANK "is the rank"

echo "Testing connectivity to master:"
ping -c 3 $MASTER_ADDR || echo "Ping failed"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_SOCKET_IFNAME=eth0  # Uncomment and set if needed
model_size=$1
dataset=$2

if [ -z "$model_size" ]; then
    echo "Model size is required"
    exit 1
fi
config=examples/config_llama32_${model_size}_continual_4node.yaml
export CONFIG_FILE=$config
echo "Using config file: $CONFIG_FILE"
# assert file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file does not exist: $CONFIG_FILE"
    exit 1
fi

if [ -z "$dataset" ]; then
    echo "Dataset is required"
    exit 1
fi

export DATASET_NAME=$dataset

torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$RANK \
    --rdzv_id=nanotron_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    run_train.py --config-file $CONFIG_FILE
