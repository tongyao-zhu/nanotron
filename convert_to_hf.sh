export LOCAL_RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=14432

path=$1

# if save_path exists and is not empty, skip
if [ -d "$path\_hf" ] && [ -n "$(ls -A $path\_hf)" ]; then
    echo "Skipping $path because it already exists"
    exit 0
fi

if [[ "$path" == *opencoder* ]]; then
    tokenizer_name="tyzhu/opencoder484"
    echo "Converting opencoder484 model"
else
    tokenizer_name="meta-llama/Llama-3.2-1B"
    echo "Converting llama3.2-1B model"
fi

python -m examples.llama.convert_nanotron_to_hf --checkpoint_path=$path --save_path=$path\_hf --tokenizer_name=$tokenizer_name --config_cls=Qwen2Config
