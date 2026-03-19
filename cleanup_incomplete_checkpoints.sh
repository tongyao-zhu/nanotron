#!/bin/bash
# Script to find and clean up incomplete checkpoints
# A checkpoint is considered complete if it has both:
#   - model/ directory
#   - checkpoint_metadata.json file

set -euo pipefail

CHECKPOINT_BASE="${1:-.}"  # Default to current directory

echo "Scanning for checkpoint directories in: $CHECKPOINT_BASE"
echo ""

for dir in "$CHECKPOINT_BASE"/checkpoint*; do
    [ -d "$dir" ] || continue
    
    # Find all numeric checkpoint folders (without _hf suffix)
    mapfile -t checkpoints < <(ls -1 "$dir" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
    
    [ ${#checkpoints[@]} -eq 0 ] && continue
    
    last_idx=$((${#checkpoints[@]} - 1))
    last_ckpt="${checkpoints[$last_idx]}"
    last_path="$dir/$last_ckpt"
    
    # Check if last checkpoint is complete
    if [ ! -d "$last_path/model" ] || [ ! -f "$last_path/checkpoint_metadata.json" ]; then
        echo "=== $(basename "$dir") ==="
        echo "  Last checkpoint $last_ckpt is INCOMPLETE"
        
        # Check if there's a valid fallback
        if [ $last_idx -gt 0 ]; then
            prev_ckpt="${checkpoints[$((last_idx - 1))]}"
            prev_path="$dir/$prev_ckpt"
            
            if [ -d "$prev_path/model" ] && [ -f "$prev_path/checkpoint_metadata.json" ]; then
                echo "  ✓ Valid fallback available: $prev_ckpt"
                
                if [ "${DRY_RUN:-1}" = "0" ]; then
                    echo "  Deleting incomplete checkpoint: $last_ckpt"
                    rm -rf "$last_path"
                    echo "  ✓ Deleted"
                    
                    # Update latest.txt
                    echo "$prev_ckpt" > "$dir/latest.txt"
                    echo "  ✓ Updated latest.txt to $prev_ckpt"
                else
                    echo "  [DRY RUN] Would delete: $last_path"
                    echo "  [DRY RUN] Would update latest.txt to: $prev_ckpt"
                fi
            else
                echo "  ✗ No valid fallback (previous checkpoint also incomplete)"
            fi
        else
            echo "  ✗ No fallback available (only one checkpoint exists)"
        fi
        echo ""
    fi
done

echo "Done."
if [ "${DRY_RUN:-1}" = "1" ]; then
    echo ""
    echo "This was a dry run. To actually delete incomplete checkpoints, run:"
    echo "  DRY_RUN=0 $0 $CHECKPOINT_BASE"
fi
