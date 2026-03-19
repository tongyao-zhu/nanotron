#!/bin/bash
source /home/aiops/zhuty/nano_start.sh
cd /home/aiops/zhuty/nanotron

base_path=$1
delete_old=$2
device=${3:-cuda}  # Default to cuda if not specified

for sub_path in $base_path/*; do
    if [ -d "$sub_path" ]; then
        dir_name=$(basename "$sub_path")
        if [[ "$dir_name" =~ ^[0-9]+$ ]]; then
            echo "Converting $sub_path using device: $device"
            bash /home/aiops/zhuty/nanotron/convert_to_hf.sh $sub_path $device
        else
            echo "Skipping $sub_path"
        fi
    fi
done

if [ "$delete_old" == "true" ]; then
    echo "[INFO] Requested deletion of old checkpoints in $base_path"
    # Collect all subdirectories with numeric names
    numeric_dirs=()
    for sub_path in $base_path/*; do
        if [ -d "$sub_path" ]; then
            dir_name=$(basename "$sub_path")
            if [[ "$dir_name" =~ ^[0-9]+$ ]]; then
                echo "[DEBUG] Found checkpoint dir: $dir_name"
                numeric_dirs+=("$sub_path")
            else
                echo "[DEBUG] Skipping non-numeric dir: $dir_name"
            fi
        fi
    done

    echo "[INFO] Numeric checkpoint dirs found: ${#numeric_dirs[@]}"
    if [ ${#numeric_dirs[@]} -gt 2 ]; then
        # Sort the directories numerically and keep the last two
        IFS=$'\n' sorted_dirs=($(for d in "${numeric_dirs[@]}"; do basename "$d"; done | sort -n))
        last_dir=${sorted_dirs[-1]}
        second_last_dir=${sorted_dirs[-2]}
        echo "[INFO] Keeping the two newest checkpoints: $second_last_dir and $last_dir"
        for dir in "${sorted_dirs[@]:0:${#sorted_dirs[@]}-2}"; do
            full_path="$base_path/$dir"
            hf_path="${full_path}_hf"
            
            # Check if _hf directory exists and is of reasonable size (> 3GB)
            if [ -d "$hf_path" ]; then
                hf_size=$(du -sb "$hf_path" 2>/dev/null | cut -f1)
                min_size=$((2 * 1024 * 1024 * 1024))  # 3GB in bytes
                
                if [ -n "$hf_size" ] && [ "$hf_size" -gt "$min_size" ]; then
                    echo "[INFO] Deleting old checkpoint $full_path (HF version exists and is ${hf_size} bytes)"
                    rm -rf "$full_path"
                else
                    echo "[WARNING] Skipping deletion of $full_path - HF version ($hf_path) is missing or too small (${hf_size:-0} bytes, need > $min_size bytes)"
                fi
            else
                echo "[WARNING] Skipping deletion of $full_path - HF version ($hf_path) does not exist"
            fi
        done
    elif [ ${#numeric_dirs[@]} -le 2 ]; then
        echo "[INFO] Two or fewer checkpoint dirs found. Nothing to delete."
    else
        echo "[INFO] No checkpoint directories found to delete."
    fi
fi
