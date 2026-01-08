#!/bin/bash
source /home/aiops/zhuty/nano_start.sh
cd /home/aiops/zhuty/nanotron

base_path=$1
delete_old=$2

for sub_path in $base_path/*; do
    if [ -d "$sub_path" ]; then
        dir_name=$(basename "$sub_path")
        if [[ "$dir_name" =~ ^[0-9]+$ ]]; then
            echo "Converting $sub_path"
            bash /home/aiops/zhuty/nanotron/convert_to_hf.sh $sub_path
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
    if [ ${#numeric_dirs[@]} -gt 1 ]; then
        # Sort the directories numerically and keep the last one
        IFS=$'\n' sorted_dirs=($(for d in "${numeric_dirs[@]}"; do basename "$d"; done | sort -n))
        last_dir=${sorted_dirs[-1]}
        echo "[INFO] Keeping the newest checkpoint: $last_dir"
        for dir in "${sorted_dirs[@]:0:${#sorted_dirs[@]}-1}"; do
            full_path="$base_path/$dir"
            echo "[INFO] Deleting old checkpoint $full_path"
            rm -rf "$full_path"
        done
    elif [ ${#numeric_dirs[@]} -eq 1 ]; then
        only_dir=$(basename "${numeric_dirs[0]}")
        echo "[INFO] Only one checkpoint dir ($only_dir) found. Nothing to delete."
    else
        echo "[INFO] No checkpoint directories found to delete."
    fi
fi
