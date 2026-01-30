
for path in /home/aiops/zhuty/nanotron/checkpoints-llama32-1b-*; do
    echo "Converting $path"
    bash /home/aiops/zhuty/nanotron/all_convert_to_hf.sh $path true
done