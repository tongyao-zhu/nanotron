#!/bin/bash

echo $MASTER_ADDR "is the master address"
echo $MASTER_PORT "is the master port"
echo $RANK "is the rank"

echo "Testing connectivity to master:"
ping -c 3 $MASTER_ADDR || echo "Ping failed"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_SOCKET_IFNAME=eth0  # Uncomment and set if needed

torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$RANK \
    --rdzv_id=nanotron_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    run_train.py --config-file examples/config_llama32_3b_continual.yaml
