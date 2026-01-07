short_name=$1

if [ -z "$short_name" ]; then
    echo "Usage: $0 <short_name>"
    exit 1
elif [ "$short_name" == "llama3-1b" ]; then
    model_name="meta-llama/Llama-3.2-1B"
elif [ "$short_name" == "llama3-3b" ]; then
    model_name="meta-llama/Llama-3.2-3B"
elif [ "$short_name" == "opencoder484" ]; then
    model_name="tyzhu/opencoder484"
else 
    echo "Invalid short name"
    exit 1
fi

python evaluate_ppl.py \
    --test-file /home/aiops/zhuty/cont_data/opptsfp1/train_local_2/shuffled_batch_066.jsonl \
    --model $model_name  \
    --context-length 8192 \
    --batch-size 64 \
    --output-file evaluate_results/$short_name-batch_066.jsonl