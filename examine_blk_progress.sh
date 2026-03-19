#!/bin/bash
# Examine training progress for blk{N}-2k checkpoint runs

PREFIXES=("mathprosfeos" "mathpros1sfeos" "mathpros10sfeos")
LABELS=("mathprosfeos" "s1sfeos" "s10sfeos")
BLKS=(16 32 64 128 256 512)
BASE_DIR="$(dirname "$(realpath "$0")")"

for i in "${!PREFIXES[@]}"; do
    prefix="${PREFIXES[$i]}"
    label="${LABELS[$i]}"
    echo "=== ${label} (${prefix}) ==="
    for blk in "${BLKS[@]}"; do
        dir="${BASE_DIR}/checkpoints-llama32-1b-${prefix}_blk${blk}-2k"
        if [ -d "$dir" ]; then
            latest=$(ls "$dir" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
            count=$(ls "$dir" 2>/dev/null | grep -E '^[0-9]+$' | wc -l)
            hf_count=$(ls -d "$dir"/*-hf 2>/dev/null | wc -l)
            printf "  blk%-4s  %2d ckpts  latest=%-6s  hf_conversions=%d\n" \
                "$blk" "$count" "${latest:-N/A}" "$hf_count"
        else
            printf "  blk%-4s  NOT FOUND\n" "$blk"
        fi
    done
    echo
done
