#!/bin/bash

# 1. Check if a directory argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <parent_directory> <command>"
    exit 1
fi

PARENT_DIR="$1"

# 2. Iterate through every subdirectory within the parent directory
# -maxdepth 1 ensures we only look at immediate subdirectories
# -type d ensures we only pick directories
for dir_path in "$PARENT_DIR"/*/; do
    
    # Check if any directories exist to avoid errors in empty folders
    [ -e "$dir_path" ] || continue

    # 3. Extract just the name of the subdirectory (remove trailing slash)
    s=$(basename "$dir_path")

    echo "Processing subdirectory: $s"

    # 4. Execute your command
    ./algo-tester-bin folder "$PARENT_DIR/$s" "$2"

done
