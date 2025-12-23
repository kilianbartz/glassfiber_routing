#!/bin/bash

# Ensure an argument was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <number of test instances per net count>"
    exit 1
fi

for k in {3..19}; do uv run gen_instance.py 20 $k -o testcases/density20/$k -i 100; done
