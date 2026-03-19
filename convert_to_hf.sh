export LOCAL_RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=14432

path=$1
device=${2:-cuda}  # Default to cuda if not specified

# if save_path exists and is not empty, skip

hf_path="${path}_hf"

if [ -d "$hf_path" ] && [ -n "$(ls -A "$hf_path")" ]; then
    folder_size=$(du -sb "$hf_path" | cut -f1)
    if [ "$folder_size" -gt 1073741824 ]; then
        echo "Skipping $path because $hf_path already exists and is larger than 1GB (size: $folder_size bytes)"
        exit 0
    else
        echo "Warning: $hf_path exists but is only $folder_size bytes (less than 1GB), will reconvert"
    fi
fi

echo "Converting $path to $hf_path because output directory $hf_path is empty or does not exist"


if [[ "$path" == *opencoder* ]]; then
    tokenizer_name="tyzhu/opencoder484"
    echo "Converting opencoder484 model"
else
    tokenizer_name="meta-llama/Llama-3.2-1B"
    echo "Converting llama3.2-1B model"
fi

python -m examples.llama.convert_nanotron_to_hf --checkpoint_path=$path --save_path=$path\_hf --tokenizer_name=$tokenizer_name --config_cls=Qwen2Config --device=$device
