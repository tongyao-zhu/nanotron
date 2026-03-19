input_path=$1
delete_old=$2
# monitor every 3600 seconds and convert the model to hf
while true; do
    if [ -d "$input_path" ]; then
        echo "Converting $input_path to hf"
        bash /home/aiops/zhuty/nanotron/all_convert_to_hf.sh $input_path  $delete_old
    else
        echo "Input path does not exist: $input_path"
    fi
    sleep 3600
done

