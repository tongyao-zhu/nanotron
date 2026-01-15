DATASET_NAME=$1 
export DATASET_NAME
MODEL_SIZE=${2:-1b}
additional_args=${3:-}

source /home/aiops/zhuty/nano_start.sh 
cd /home/aiops/zhuty/nanotron
echo "Using dataset: $DATASET_NAME"
echo "Using model size: $MODEL_SIZE"
config=examples/config_llama32_${MODEL_SIZE}_continual_template.yaml
SUFFIX=${DATASET_NAME}


# if additional args contains '--diffusion', set IS_DIFFUSION to true
if [[ "$additional_args" == *"--diffusion"* ]]; then
    export IS_DIFFUSION=true
    SUFFIX=${SUFFIX}_diff
    export MASK_TOKEN_ID=128255
else
    export IS_DIFFUSION=false
    export MASK_TOKEN_ID=-1
fi

# if additional args contains '--intradoc', set INTRADOC to true
if [[ "$additional_args" == *"--intradoc"* ]]; then
    export INTRADOC=true
    SUFFIX=${SUFFIX}_intra
else
    export INTRADOC=false
fi
    
# if zero1, set zero_stage to 1
if [[ "$additional_args" == *"--zero1"* ]]; then
    export ZERO_STAGE=1
    export ACCUMULATE_GRAD_IN_FP32=false
    SUFFIX=${SUFFIX}_zero1
else
    export ZERO_STAGE=0
    export ACCUMULATE_GRAD_IN_FP32=true
fi

echo "Using suffix: $SUFFIX"
echo "Using config file: $config"
# assert file exists
if [ ! -f "$config" ]; then
    echo "Config file does not exist: $config"
    exit 1
fi



dataset_folder=/home/aiops/zhuty/cont_data/${DATASET_NAME}/llama_tokenized
# if dataset_folder does not exist, exit
if [ ! -d "$dataset_folder" ]; then
    echo "Dataset folder does not exist: $dataset_folder"
    exit 1
fi
export SUFFIX=$SUFFIX
export DP=8
export BATCH_ACCUM=64
torchrun --nproc_per_node=8 run_train.py --config-file $config