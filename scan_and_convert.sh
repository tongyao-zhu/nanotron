period=$1

# if period is set, run this for every period seconds
if [ -n "$period" ]; then
    while true; do
        for path in /home/aiops/zhuty/nanotron/checkpoints-llama32-*; do
            echo "Converting $path"
            bash /home/aiops/zhuty/nanotron/all_convert_to_hf.sh $path true 
        done
        sleep $period
    done
else
    for path in /home/aiops/zhuty/nanotron/checkpoints-llama32-*; do
        echo "Converting $path"
        bash /home/aiops/zhuty/nanotron/all_convert_to_hf.sh $path true 
    done
fi