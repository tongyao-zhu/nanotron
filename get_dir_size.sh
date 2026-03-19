#!/bin/bash
# List checkpoints* directories with last update date and size, oldest first

printf "%-80s %-20s %s\n" "DIRECTORY" "LAST UPDATED" "SIZE"
printf "%-80s %-20s %s\n" "$(printf '%0.s-' {1..80})" "$(printf '%0.s-' {1..20})" "$(printf '%0.s-' {1..10})"

find . -maxdepth 1 -type d -name 'checkpoints*' -printf '%T+ %p\n' | sort | while read -r ts dir; do
    date_part="${ts%%.*}"
    date_fmt="${date_part/+/ }"
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    printf "%-80s %-20s %s\n" "$dir" "$date_fmt" "$size"
done
